from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

from .io_tiff import read_tiff_stack, write_tiff_stack
from .stack_split import split_tiff_stack_by_z
from .ambiguity import plausible_options, PlausibleOption
from .axes_utils import group_stacks_from_axes
from .layout_infer import generate_interpretations, apply_interpretation_to_groups, ShapeInterpretation

OrderChoice = Literal["s-z-o", "s-o-z", "z-s-o", "z-o-s", "o-s-z", "o-z-s"]


@dataclass(frozen=True)
class UIState:
	input_path: str
	output_dir: str
	num_shifts: Literal[3, 5]
	order: OrderChoice


class MainWindow(QtWidgets.QMainWindow):
	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("TIFF Stack Splitter (by Z)")
		self.setMinimumWidth(720)

		central = QtWidgets.QWidget()
		self.setCentralWidget(central)

		layout = QtWidgets.QVBoxLayout()
		central.setLayout(layout)

		# Input file
		in_row = QtWidgets.QHBoxLayout()
		layout.addLayout(in_row)

		self.input_line = QtWidgets.QLineEdit()
		self.input_line.setPlaceholderText("Select input .tif/.tiff (or .npy)")
		in_btn = QtWidgets.QPushButton("Choose Input…")
		in_btn.clicked.connect(self.choose_input)

		in_row.addWidget(QtWidgets.QLabel("Input:"))
		in_row.addWidget(self.input_line, 1)
		in_row.addWidget(in_btn)

		# Output folder
		out_row = QtWidgets.QHBoxLayout()
		layout.addLayout(out_row)

		self.output_line = QtWidgets.QLineEdit()
		self.output_line.setPlaceholderText("Select output folder")
		out_btn = QtWidgets.QPushButton("Choose Folder…")
		out_btn.clicked.connect(self.choose_output_dir)

		out_row.addWidget(QtWidgets.QLabel("Output:"))
		out_row.addWidget(self.output_line, 1)
		out_row.addWidget(out_btn)

		# Options row
		opts = QtWidgets.QGridLayout()
		layout.addLayout(opts)

		self.mode_combo = QtWidgets.QComboBox()
		self.mode_combo.addItem("2D (s=3 shifts)", 3)
		self.mode_combo.addItem("3D (s=5 shifts)", 5)

		self.order_combo = QtWidgets.QComboBox()
		for opt in ["s-z-o", "s-o-z", "z-s-o", "z-o-s", "o-s-z", "o-z-s"]:
			self.order_combo.addItem(opt, opt)

		note = QtWidgets.QLabel(
			"Note: z is inferred as n_frames / (3 * s). "
			"Choose 2D if each z has 3*3 frames; choose 3D if each z has 3*5 frames."
		)
		note.setWordWrap(True)

		opts.addWidget(QtWidgets.QLabel("Mode:"), 0, 0)
		opts.addWidget(self.mode_combo, 0, 1)
		opts.addWidget(QtWidgets.QLabel("Order:"), 0, 2)
		opts.addWidget(self.order_combo, 0, 3)
		opts.addWidget(note, 2, 0, 1, 4)

		# Run button
		self.run_btn = QtWidgets.QPushButton("Split and Save")
		self.run_btn.clicked.connect(self.run_split)
		layout.addWidget(self.run_btn)

		# Log/error window
		self.log = QtWidgets.QPlainTextEdit()
		self.log.setReadOnly(True)
		self.log.setPlaceholderText("Messages and errors will appear here…")
		layout.addWidget(self.log, 1)
	
	def choose_shape_interpretation(
		self,
		raw_shape: Tuple[int, ...],
		its: List[ShapeInterpretation],
	) -> ShapeInterpretation | None:
		"""
		If no interpretations exist, return None.
		If more than three are possible, raise an error advising clearer metadata or pre-separating channels.
		Otherwise, open a dialog titled “Ambiguous stack layout (shape-based)” explaining that axes metadata is unreliable and asking the user to pick the correct layout.
		Display each interpretation as a radio button (mutually exclusive), showing:
		which axis is s (with its shift count),
		which axis is o=3,
		which axis (if any) is k (with its size),
		the inferred z.
		Preselect the first option.
		If the user cancels, return None.
		If accepted, return the selected interpretation.
		"""
		if not its:
			return None

		# If there are too many, bail out as you requested
		if len(its) > 3:
			raise ValueError(
				f"Too many plausible interpretations ({len(its)}) for raw shape={raw_shape}. "
				"Please export/convert with clearer metadata (prefer OME-TIFF), or separate channels before using this tool."
			)

		dlg = QtWidgets.QDialog(self)
		dlg.setWindowTitle("Ambiguous stack layout (shape-based)")
		v = QtWidgets.QVBoxLayout(dlg)

		label = QtWidgets.QLabel(
			"Axes metadata is not reliable.\n"
			"Select which interpretation matches the converted file layout:"
		)
		label.setWordWrap(True)
		v.addWidget(label)

		group = QtWidgets.QButtonGroup(dlg)
		group.setExclusive(True)

		buttons: List[Tuple[QtWidgets.QRadioButton, ShapeInterpretation]] = []
		for i, it in enumerate(its):
			o_txt = f"axis#{it.o_axis}"
			s_txt = f"axis#{it.s_axis}"
			k_txt = "none" if it.k_axis is None else f"axis#{it.k_axis} (k={it.k})"
			text = (
				f"Option {i+1}: s={it.num_shifts} on {s_txt}, o=3 on {o_txt}, "
				f"k={k_txt}, inferred z={it.z}"
			)
			rb = QtWidgets.QRadioButton(text)
			if i == 0:
				rb.setChecked(True)
			group.addButton(rb)
			v.addWidget(rb)
			buttons.append((rb, it))

		btns = QtWidgets.QDialogButtonBox(
			QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
		)
		v.addWidget(btns)
		btns.accepted.connect(dlg.accept)
		btns.rejected.connect(dlg.reject)

		if dlg.exec() != QtWidgets.QDialog.Accepted:
			return None

		for rb, it in buttons:
			if rb.isChecked():
				return it
		return None

	def score_interpretation(self, it: ShapeInterpretation) -> int:
		"""
		Assign a heuristic score to an interpretation:
		+10 if s = 5
		+5 if k > 1
		+8 if k = 2
		+4 if k is 3 or 4
		Return the total score (higher is better).

		Higher is better. Simple heuristic:
		- Prefer s=5 over s=3 (often "3D mode")
		- Prefer k=2 (common channels) then k in (3,4)
		- Prefer having an explicit k axis (k>1)
		"""
		score = 0
		if it.num_shifts == 5:
			score += 10
		if it.k > 1:
			score += 5
		if it.k == 2:
			score += 8
		elif it.k in (3, 4):
			score += 4
		return score

	def auto_pick_interpretation(self, its: List[ShapeInterpretation]) -> ShapeInterpretation | None:
		"""
		If no interpretations exist, return None.
		Score and sort them descending.
		If only one exists, return it.
		If the top score exceeds the second by at least 6, return the best.
		Otherwise, return None.
		calls: score_interpretation
		"""
		if not its:
			return None
		# If there is a single best score and it beats the runner-up by a margin, auto-pick
		scored = sorted(((self.score_interpretation(it), it) for it in its), key=lambda x: x[0], reverse=True)
		best_score, best = scored[0]
		if len(scored) == 1:
			return best
		second_score, _ = scored[1]
		if best_score - second_score >= 6:
			return best
		return None

	def choose_ambiguity_options(
		self,
		frames_total: int,
	) -> List[PlausibleOption]:
		"""
			Generate plausible interpretations for frames_total.
			If none exist, raise an error.
			Show a dialog titled “Ambiguous stack layout” explaining that metadata is unreliable and asking the user to select interpretations to export.
			For each plausible option, display an unchecked checkbox labeled with its group count, shifts (s), and z value.
			Provide OK and Cancel buttons.
			If the user cancels, return an empty list.
			If accepted, return the list of options whose checkboxes were selected.
			calls: plausible_options
		"""
		opts = plausible_options(frames_total)
		if not opts:
			raise ValueError(
				f"Cannot find any plausible interpretations for frames_total={frames_total} with o=3 and s in (3,5)."
			)

		dlg = QtWidgets.QDialog(self)
		dlg.setWindowTitle("Ambiguous stack layout")
		v = QtWidgets.QVBoxLayout(dlg)

		label = QtWidgets.QLabel(
			"Metadata does not reliably specify channels/FOV.\n"
			"Select which interpretations you want to export:"
		)
		label.setWordWrap(True)
		v.addWidget(label)

		checks: List[Tuple[QtWidgets.QCheckBox, PlausibleOption]] = []
		for opt in opts:
			text = f"Assume {opt.k_groups} group(s): s={opt.num_shifts}, z={opt.z}"
			cb = QtWidgets.QCheckBox(text)
			cb.setChecked(False)
			v.addWidget(cb)
			checks.append((cb, opt))

		btns = QtWidgets.QDialogButtonBox(
			QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
		)
		v.addWidget(btns)

		btns.accepted.connect(dlg.accept)
		btns.rejected.connect(dlg.reject)

		if dlg.exec() != QtWidgets.QDialog.Accepted:
			return []

		selected: List[PlausibleOption] = []
		for cb, opt in checks:
			if cb.isChecked():
				selected.append(opt)
		return selected

	def append_log(self, msg: str) -> None:
		self.log.appendPlainText(msg)

	def choose_input(self) -> None:
		path, _ = QtWidgets.QFileDialog.getOpenFileName(
			self,
			"Choose input stack",
			"",
			"Stacks (*.tif *.tiff *.ome.tif *.ome.tiff *.npy);;All files (*)",
		)
		if path:
			self.input_line.setText(path)

	def choose_output_dir(self) -> None:
		path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder", "")
		if path:
			self.output_line.setText(path)

	def get_state(self) -> UIState:
		input_path = self.input_line.text().strip()
		output_dir = self.output_line.text().strip()
		num_shifts = int(self.mode_combo.currentData())
		order = str(self.order_combo.currentData())

		if not input_path:
			raise ValueError("No input file selected.")
		if not os.path.exists(input_path):
			raise ValueError(f"Input path does not exist: {input_path}")
		if not output_dir:
			raise ValueError("No output folder selected.")
		if not os.path.isdir(output_dir):
			raise ValueError(f"Output folder does not exist: {output_dir}")

		if num_shifts not in (3, 5):
			raise ValueError("Mode must be 2D (3 shifts) or 3D (5 shifts).")

		return UIState(
			input_path=input_path,
			output_dir=output_dir,
			num_shifts=num_shifts,  # type: ignore[arg-type]
			order=order,  # type: ignore[arg-type]
		)

	def load_stack(self, path: str):
		ext = os.path.splitext(path.lower())[1]

		if ext == ".npy":
			arr = np.load(path, allow_pickle=False)
			if arr.ndim < 2:
				raise ValueError(f".npy must have at least 2 dims (frames, image...). Got {arr.shape}")
			return arr, "npy"

		if ext in (".tif", ".tiff") or path.lower().endswith((".ome.tif", ".ome.tiff")):
			rr = read_tiff_stack(path)
			return rr, "tiff"

		raise ValueError("Unsupported input type for MVP. Use .tif/.tiff/.ome.tif/.ome.tiff or .npy")
	
	def split_into_k_groups(self, stack: np.ndarray, k: int) -> List[np.ndarray]:
		"""
		Splits (frames, ...) into k contiguous groups of equal size.
		Assumes stack[0] is the frames axis.
		"""
		if k <= 0:
			raise ValueError("k must be > 0")
		n = int(stack.shape[0])
		if n % k != 0:
			raise ValueError(f"Cannot split {n} frames into k={k} equal groups.")
		step = n // k
		return [stack[i * step:(i + 1) * step] for i in range(k)]

	def export_stack(
		self,
		*,
		base: str,
		out_dir: str,
		tag: str,
		stack: np.ndarray,
		num_shifts: Literal[3, 5],
		order: str,
	) -> None:
		
		if not stack.flags["C_CONTIGUOUS"]:
			self.append_log(f"Making stack contiguous (was non-contiguous): shape={stack.shape}")
			stack = np.ascontiguousarray(stack)

		self.append_log(f"Exporting tag={tag} stack_shape={stack.shape}")
		"""
		split_tiff_stack_by_z
		Splits a flattened SIM frame stack into per-Z stacks, inferring
		the number of Z slices from the total frame count.
		Overview
		The input is assumed to be a single flattened frame sequence
		representing exactly three logical SIM axes:
			- orientations (o)
			- shifts/phases (s)
			- z slices (z)
		The function groups frames by z and returns one output stack per z slice.
		Frames within each z output preserve their original order from the input.
		No semantic reordering of (o, s) is performed.
		Assumptions
		- Number of orientations is fixed at o = 3.
		- Number of shifts is specified per call: s ∈ {3, 5}.
		- Total frames must satisfy:
				n_frames = o * s * z
		- Frame order is a permutation of (s, z, o),
		where `order` specifies fastest → slowest varying axis.
		- The input stack contains only SIM frames
		(no timepoints, preview frames, extra channels, or mixed FOVs).
		If these assumptions are violated (e.g., dropped frames, extra
		dimensions flattened into the sequence, multiple FOVs not separated),
		Z inference or grouping may be incorrect.
		Inputs
		stack: np.ndarray
			Shape (n_frames, *image_dims).
			Axis 0 must be the flattened SIM frame sequence.
		num_shifts: Literal[3, 5]
			Number of SIM phase/shift positions.
		order: str | Sequence["s","z","o"]
			Permutation describing frame order,
			fastest → slowest (e.g. "s-z-o").
		Outputs
		Returns StackSplitResult containing:
		per_z: list[np.ndarray]
			List of length z.
			Each element has shape:
				(o*s, *image_dims)
			and contains all frames belonging to that z slice.
			Frame order within each element matches the input order.
		z: int
			Inferred number of Z slices.
		order: tuple[str, str, str]
			Parsed axis order (fastest → slowest).
		Notes
		- Raises an error if n_frames == 0.
		- Raises an error if n_frames is not divisible by (o*s).
		- No frame reordering is performed; grouping is index-based.
		- Output frame order within each Z stack is not guaranteed
		to follow any specific (o, s) nesting and should not be
		semantically interpreted.
		"""
		res = split_tiff_stack_by_z(
			stack,
			num_shifts=num_shifts,
			order=order,
		)
		self.append_log(f"[{tag}] Inferred z={res.z}")

		for zi, block in enumerate(res.per_z):
			out_name = f"{base}_{tag}_z{zi:03d}.tif"
			out_path = os.path.join(out_dir, out_name)
			write_tiff_stack(out_path, block)
			self.append_log(f"[{tag}] Wrote: {out_path} shape={block.shape} dtype={block.dtype}")
	
	def run_split(self) -> None:
		self.log.clear()
		try:
			state = self.get_state()
			self.append_log(f"Reading: {state.input_path}")
			loaded, kind = self.load_stack(state.input_path)

			base = os.path.splitext(os.path.basename(state.input_path))[0]

			# --- Reliable metadata path (TIFF with trustworthy axes) ---
			if kind == "tiff":
				rr = loaded
				# rr is ReadResult (after your io_tiff.py changes)
				self.append_log(
					f"TIFF series axes={rr.axes!r} shape={rr.metadata.get('series_shape')} reliable={rr.axes_reliable}"
				)

				if rr.axes_reliable and rr.axes:
					# Group by position/FOV (if any) and channel (if any), then flatten to frames
					groups = group_stacks_from_axes(rr.raw, rr.axes)
					self.append_log(f"Detected {len(groups)} group(s) from axes metadata.")

					for g in groups:
						# Use the mode selected in UI (s/order/flatten), just like before
						self.append_log(
							f"Splitting group {g.tag} with s={state.num_shifts}, o=3, order={state.order},"
						)
						self.export_stack(
							base=base,
							out_dir=state.output_dir,
							tag=g.tag,
							stack=g.stack,
							num_shifts=state.num_shifts,
							order=state.order,
						)

					self.append_log("Done.")
					return

				raw = rr.raw
				raw_shape = tuple(int(x) for x in raw.shape)
				# Axes unreliable: do NOT assume `raw` is (frames,Y,X[,C]).
				# Prefer the frames-first view produced by read_tiff_stack().
				stack = rr.stack
				self.append_log(
					f"WARNING: Axes not reliable; attempting shape-based inference from raw shape={raw_shape}."
				)
				self.append_log(
					f"Using frames-first stack from reader: frames_total={int(stack.shape[0])}, stack_shape={stack.shape}"
				)

				# For FOV=1 validation, go through the same ambiguity logic used for NPY:
				frames_total = int(stack.shape[0])
				options = plausible_options(frames_total)
				if not options:
					raise ValueError(
						f"No plausible interpretations found for frames_total={frames_total}. "
						"Try a different export from Fiji or ensure frames are (z*o*s)."
					)

				selected = self.choose_ambiguity_options(frames_total)
				if not selected:
					self.append_log("Cancelled.")
					return

				for opt in selected:
					self.append_log(f"Selected: k={opt.k_groups}, s={opt.num_shifts}, inferred_z={opt.z}")
					substacks = self.split_into_k_groups(stack, opt.k_groups)
					for gi, sub in enumerate(substacks):
						tag = f"k{opt.k_groups:02d}_g{gi:02d}"
						self.append_log(
							f"Splitting {tag} with s={opt.num_shifts}, o=3, order={state.order}"
						)
						self.export_stack(
							base=base,
							out_dir=state.output_dir,
							tag=tag,
							stack=sub,
							num_shifts=opt.num_shifts,
							order=state.order,
						)

				self.append_log("Done.")
				return

			# --- NPY path (no metadata), or TIFF fallback path ---
			if kind == "npy":
				stack = loaded
				self.append_log(f"Loaded NPY stack shape={stack.shape}, dtype={stack.dtype}")

			self.append_log(
				f"Ambiguity handling: frames_total={int(stack.shape[0])}, using o=3 and s in (3,5)."
			)

			frames_total = int(stack.shape[0])
			options = plausible_options(frames_total)

			if not options:
				raise ValueError(
					f"No plausible interpretations found for frames_total={frames_total}. "
					"Try a different export from Fiji or ensure frames are (z*o*s)."
				)

			selected = self.choose_ambiguity_options(frames_total)
			if not selected:
				self.append_log("Cancelled.")
				return

			# Export each selected interpretation
			for opt in selected:
				# Split into k contiguous groups (best effort assumption)
				self.append_log(f"Selected: k={opt.k_groups}, s={opt.num_shifts}, inferred_z={opt.z}")
				substacks = self.split_into_k_groups(stack, opt.k_groups)

				for gi, sub in enumerate(substacks):
					tag = f"k{opt.k_groups:02d}_g{gi:02d}"
					self.append_log(
						f"Splitting {tag} with s={opt.num_shifts}, o=3, order={state.order}"
					)
					self.export_stack(
						base=base,
						out_dir=state.output_dir,
						tag=tag,
						stack=sub,
						num_shifts=opt.num_shifts,
						order=state.order,
					)

			self.append_log("Done.")

		except Exception as e:
			self.append_log("ERROR:")
			self.append_log(str(e))
			self.append_log("")
			self.append_log("Traceback:")
			self.append_log(traceback.format_exc())


def main() -> None:
	app = QtWidgets.QApplication([])
	w = MainWindow()
	w.show()
	app.exec()


if __name__ == "__main__":
	main()
