from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Optional, Sequence, Union

import numpy as np
from PySide6 import QtWidgets

import tifffile

AxisName = Literal["s", "z", "o"]
OrderSpec = Union[str, Sequence[AxisName]]

@dataclass(frozen=True)
class StackSplitResult:
	# One item per z: each contains all frames belonging to that z,
	# shaped (o*s, *image_dims), preserving original frame order.
	per_z: List[np.ndarray]
	# The inferred z count
	z: int
	# Echo back the interpreted order (fastest -> slowest)
	order: Tuple[AxisName, AxisName, AxisName]


def _parse_order(order: OrderSpec) -> Tuple[AxisName, AxisName, AxisName]:
	if isinstance(order, str):
		parts = tuple(p.strip() for p in order.split("-") if p.strip())
	else:
		parts = tuple(order)

	if len(parts) != 3:
		raise ValueError(f"order must have 3 axes (some permutation of s,z,o). Got: {order!r}")

	valid = {"s", "z", "o"}
	if set(parts) != valid:
		raise ValueError(f"order must be a permutation of ('s','z','o'). Got: {parts!r}")

	# Type narrow
	return parts[0], parts[1], parts[2]  # type: ignore[return-value]

def split_tiff_stack_by_z(
	stack: np.ndarray,
	*,
	num_shifts: Literal[3, 5],
	order: OrderSpec,
) -> StackSplitResult:
	"""
	Splits a TIFF-like frame stack into per-z stacks, given:
	- o (orientations/angles) is always 3
	- s (phases/shifts) is 3 or 5
	- z inferred from n_frames / (o*s)
	- 'order' specifies how frames are arranged in the flattened sequence,
	  e.g. "s-z-o" means s varies fastest, then z, then o.

	- For each z, we simply grab all frames that belong to that z from the
	  original flattened sequence, preserving their original order.
	- The order of frames *within* each per-z output does not matter.

	Parameters
	----------
	stack:
		np.ndarray shaped like (n_frames, *image_dims).
	num_shifts:
		3 (2D option) or 5 (3D option).
	order:
		Permutation of 's', 'z', 'o' describing the frame order (FASTEST -> SLOWEST).

	Returns
	-------
	StackSplitResult with:
	- per_z: list length z, each shaped (o*s, *image_dims)
	"""
	if not isinstance(stack, np.ndarray):
		raise TypeError("stack must be a numpy array")

	if stack.ndim < 2:
		# A stack of images should have at least (frames, pixels...)
		raise ValueError(f"stack must have at least 2 dims (frames, image...). Got shape: {stack.shape}")

	o = 3
	s = int(num_shifts)
	n_frames = int(stack.shape[0])
	denom = o * s

	if n_frames == 0:
		raise ValueError("stack has 0 frames; cannot infer z")

	if n_frames % denom != 0:
		raise ValueError(
			"Cannot infer z because n_frames is not divisible by (o*s). "
			f"Got n_frames={n_frames}, o={o}, s={s}, o*s={denom}."
		)

	z = n_frames // denom

	# User 'order' is interpreted as FASTEST -> SLOWEST (inner loop -> outer loop),
	# matching your example: order="s-z-o" => o slowest, z middle, s fastest.
	user_order = _parse_order(order)

	# Build stride (in "flattened-frame index space") for each logical axis,
	# based on fastest->slowest convention:
	#
	# Example user_order = ('s','z','o')
	# sizes: s=5, z=10, o=3
	# stride[s]=1
	# stride[z]=size[s]
	# stride[o]=size[s]*size[z]
	size_by_axis: Dict[AxisName, int] = {"s": s, "z": z, "o": o}
	stride_by_axis: Dict[AxisName, int] = {}

	stride = 1
	for a in user_order:
		stride_by_axis[a] = stride
		stride *= int(size_by_axis[a])

	# Given a frame index i (0..n_frames-1), recover the z coordinate without reshaping/transposing:
	# coord[a] = (i // stride[a]) % size[a]
	z_stride = int(stride_by_axis["z"])
	z_size = int(size_by_axis["z"])

	# Collect indices for each z slice. This is O(n_frames) and avoids any global reordering.
	indices_by_z: List[List[int]] = [[] for _ in range(z)]
	for i in range(n_frames):
		zi = (i // z_stride) % z_size
		indices_by_z[int(zi)].append(int(i))

	# Build per-z stacks by fancy indexing. This preserves the original frame order within each z.
	# Each block contains exactly o*s frames (unless the input is malformed).
	per_z: List[np.ndarray] = []
	for zi in range(z):
		idxs = indices_by_z[zi]

		# Sanity check: each z should contain exactly denom frames for SIM (o*s).
		# If not, something about the input/order assumptions is off.
		if len(idxs) != denom:
			raise ValueError(
				"Input/order mapping produced an unexpected number of frames for a z-slice. "
				f"z={z}, expected_per_z={denom}, got_per_z={len(idxs)} for zi={zi}. "
				"If this is multi-FOV or includes extra frames, split channels/FOVs first."
			)

		block = stack[idxs]  # shape: (o*s, *image_dims)
		per_z.append(block)

	# NOTE:
	# If your StackSplitResult currently requires an ndarray here, change it to Optional[np.ndarray].
	return StackSplitResult(
		per_z=per_z,
		z=z,
		order=user_order,  # keep reporting what the user provided
	)


@dataclass(frozen=True)
class ReadResult:
	raw: np.ndarray				# First tifffile series array exactly as stored (no reshape/flatten)
	is_ome: bool
	ome_xml: Optional[str]
	axes: Optional[str]			# Series axes if available (OME/tifffile), else None
	axes_reliable: bool			# True only if axes exists, matches raw.ndim, and includes at least X/Y
	metadata: Dict[str, Any]	# dtype, shape, ome_present, plus anything else useful


def read_tiff_stack(path: str) -> ReadResult:
	"""
	Load TIFF via tifffile, read OME-XML if present.

	Return:
	- raw: the first series array exactly as stored (no flattening, no reshape).
	- axes: axes string if available (OME / tifffile), else None.
	- axes_reliable: True only if axes exists, matches raw.ndim, and contains at least X/Y.
	- metadata: dtype, shape, ome_present, plus anything else you already stored.

	Do not produce a flattened/normalized stack anymore; grouping/normalization happens later.
	"""
	with tifffile.TiffFile(path) as tf:
		is_ome = bool(getattr(tf, "is_ome", False))
		ome_xml: Optional[str] = None

		# OME-XML (when present) is usually available via tf.ome_metadata for OME-TIFFs.
		if is_ome:
			try:
				ome_xml = tf.ome_metadata
			except Exception:
				# Be permissive: some files claim OME-ish properties but parsing can fail.
				ome_xml = None

		# First series exactly as stored
		series = tf.series[0]
		raw = series.asarray()

		# Axes / shape as reported by tifffile (if available)
		axes = getattr(series, "axes", None)
		series_shape = getattr(series, "shape", None)

		# Reliability rule: axes exists, matches raw.ndim, and includes at least X and Y.
		# Note: we intentionally compare to raw.ndim (not series_shape length), because raw is the truth.
		axes_reliable = bool(
			axes
			and isinstance(axes, str)
			and (len(axes) == int(raw.ndim))
			and ("X" in axes and "Y" in axes)
		)

		metadata: Dict[str, Any] = {}
		metadata["ome_present"] = bool(is_ome)
		metadata["dtype"] = str(raw.dtype)
		metadata["shape"] = tuple(int(x) for x in raw.shape)

		# Preserve the “series_*” info you were already storing (these describe tifffile’s view).
		metadata["series_axes"] = axes
		metadata["series_shape"] = tuple(int(x) for x in series_shape) if series_shape is not None else None
		metadata["axes_reliable"] = bool(axes_reliable)

		# A couple of extra fields that are often handy and cheap to include.
		metadata["path"] = path
		try:
			metadata["n_series"] = int(len(tf.series))
		except Exception:
			metadata["n_series"] = None

		return ReadResult(
			raw=raw,
			is_ome=is_ome,
			ome_xml=ome_xml,
			axes=axes,
			axes_reliable=axes_reliable,
			metadata=metadata,
		)

def write_tiff_stack(path: str, stack: np.ndarray) -> None:
	tifffile.imwrite(path, stack)

@dataclass(frozen=True)
class GroupedStack:
	tag: str				# e.g. "c00", "p01_c00"
	stack: np.ndarray		# (frames, Y, X) or (frames, ...)

@dataclass(frozen=True)
class SimpleLayout:
	has_fov: bool
	has_rgb: bool
	fov_axis: Optional[int]		# None if no FOV
	frames_axis: int
	y_axis: int
	x_axis: int
	c_axis: Optional[int]		# None if no RGB
	num_fov: int				# 1 if no FOV


def _is_plausible_xy_dim(n: int) -> bool:
	return int(n) >= 64


def _is_plausible_fov_dim(n: int) -> bool:
	n = int(n)
	return 0 < n <= 18


def _is_plausible_rgb_dim(n: int) -> bool:
	return int(n) in (3, 4)

def detect_nice_1d_layout(raw, log_cb):
	"""
	Detect the six “easy” layouts:
		(F, Y, X)
		(P, F, Y, X)
		(F, P, Y, X)
		(F, Y, X, C)
		(P, F, Y, X, C)
		(F, P, Y, X, C)

	Where:
		- Y and X are large (>=64)
		- C is 3 or 4 (RGB/RGBA) and must be last
		- P (FOV) is small (<=18) and must be first if present, *unless* we detect (F,P,...) by rule below
		- F is the SIM frames axis (already flattened SIM sequence)

	Swapped-axis preference rule:
		Prefer (F,P,...) over (P,F,...) only if F > 18 and P <= 18.
	"""
	shape = tuple(int(s) for s in raw.shape)
	nd = raw.ndim

	if nd < 3:
		_log(log_cb, "stack has less than 3 dimensions")

	if nd == 3:
		f, y, x = shape
		if _is_plausible_xy_dim(y) and _is_plausible_xy_dim(x):
			return SimpleLayout(
				has_fov=False,
				has_rgb=False,
				fov_axis=None,
				frames_axis=0,
				y_axis=1,
				x_axis=2,
				c_axis=None,
				num_fov=1,
			)
		return None

	if nd == 4:
		# Either (P,F,Y,X) or (F,P,Y,X) or (F,Y,X,C)
		a, b, c, d = shape

		# (F,Y,X,C)
		if _is_plausible_xy_dim(b) and _is_plausible_xy_dim(c) and _is_plausible_rgb_dim(d):
			return SimpleLayout(
				has_fov=False,
				has_rgb=True,
				fov_axis=None,
				frames_axis=0,
				y_axis=1,
				x_axis=2,
				c_axis=3,
				num_fov=1,
			)

		# For (P,F,Y,X) and (F,P,Y,X), Y,X are always the last two dims here
		if _is_plausible_xy_dim(c) and _is_plausible_xy_dim(d):
			pf_ok = _is_plausible_fov_dim(a)				# (P,F,Y,X)
			fp_ok = _is_plausible_fov_dim(b)				# (F,P,Y,X)

			# Prefer swapped only if it clearly looks like (F,P,...) by your rule
			if fp_ok and (int(a) > 18) and (int(b) <= 18):
				return SimpleLayout(
					has_fov=True,
					has_rgb=False,
					fov_axis=1,
					frames_axis=0,
					y_axis=2,
					x_axis=3,
					c_axis=None,
					num_fov=int(b),
				)

			# Otherwise fall back to the canonical (P,F,Y,X) if it matches
			if pf_ok:
				return SimpleLayout(
					has_fov=True,
					has_rgb=False,
					fov_axis=0,
					frames_axis=1,
					y_axis=2,
					x_axis=3,
					c_axis=None,
					num_fov=int(a),
				)
			_log(log_cb, "5 dimensions. fp_ok: " + str(fp_ok) + ". pf_ok: " + str(pf_ok))

		return None

	if nd == 5:
		# Either (P,F,Y,X,C) or (F,P,Y,X,C)
		a, b, y, x, ch = shape
		if _is_plausible_xy_dim(y) and _is_plausible_xy_dim(x) and _is_plausible_rgb_dim(ch):
			pf_ok = _is_plausible_fov_dim(a)				# (P,F,Y,X,C)
			fp_ok = _is_plausible_fov_dim(b)				# (F,P,Y,X,C)

			# Prefer swapped only if it clearly looks like (F,P,...) by your rule
			if fp_ok and (int(a) > 18) and (int(b) <= 18):
				return SimpleLayout(
					has_fov=True,
					has_rgb=True,
					fov_axis=1,
					frames_axis=0,
					y_axis=2,
					x_axis=3,
					c_axis=4,
					num_fov=int(b),
				)

			# Otherwise fall back to the canonical (P,F,Y,X,C) if it matches
			if pf_ok:
				return SimpleLayout(
					has_fov=True,
					has_rgb=True,
					fov_axis=0,
					frames_axis=1,
					y_axis=2,
					x_axis=3,
					c_axis=4,
					num_fov=int(a),
				)
			_log(log_cb, "5 dimensions. fp_ok: " + str(fp_ok) + ". pf_ok: " + str(pf_ok))
		return None
	
	if(nd > 5):
		_log(log_cb, "stack has more than 5 dimensions")

	return None


def _move_axes_to_frames_yx(raw: np.ndarray, *, frames_axis: int, y_axis: int, x_axis: int, c_axis: Optional[int]) -> np.ndarray:
	"""
	Reorder axes to:
		(frames, Y, X) or (frames, Y, X, C)

	No flattening. Only axis moves.
	"""
	if c_axis is None:
		return np.moveaxis(raw, (frames_axis, y_axis, x_axis), (0, 1, 2))

	# Ensure C is last in the result
	return np.moveaxis(raw, (frames_axis, y_axis, x_axis, c_axis), (0, 1, 2, 3))

def _log(log_cb, msg: str) -> None:
	if log_cb is not None:
		log_cb(msg)

def _axis_after_drop(ax: int, drop_ax: int) -> int:
	# If we remove one axis, all axes after it shift left by 1.
	return ax - 1 if ax > drop_ax else ax

def group_stacks_nice_first(
	raw: np.ndarray,
	*,
	log_cb=None,
) -> Optional[List["GroupedStack"]]:
	"""
	If raw matches a “nice” 1D stack layout, return grouped stacks:
	- one per FOV (if present), else one group
	- each stack is (frames, Y, X) or (frames, Y, X, C)
	Tags:
		- p00_c00 style if FOV present (p??), otherwise c00 only.
	We do NOT interpret multi-channel SIM here. Only FOV first or second and RGB-last.
	"""
	layout = detect_nice_1d_layout(raw, log_cb)
	if layout is None:
		return None

	stacks: List[GroupedStack] = []

	if layout.has_fov:
		_log(log_cb, f"Detected nice layout with FOV axis: raw_shape={raw.shape} num_fov={layout.num_fov}")
		
		if int(raw.shape[layout.fov_axis]) == 1:
			_log(log_cb, f"FOV axis is singleton (size=1). Treating as no-FOV by dropping axis {layout.fov_axis}.")
			raw2 = np.take(raw, 0, axis=layout.fov_axis)

			frames_axis = _axis_after_drop(layout.frames_axis, layout.fov_axis)
			y_axis = _axis_after_drop(layout.y_axis, layout.fov_axis)
			x_axis = _axis_after_drop(layout.x_axis, layout.fov_axis)
			c_axis = (_axis_after_drop(layout.c_axis, layout.fov_axis) if layout.c_axis is not None else None)

			stack = _move_axes_to_frames_yx(raw2, frames_axis=frames_axis, y_axis=y_axis, x_axis=x_axis, c_axis=c_axis)
			stacks.append(GroupedStack(tag="c00", stack=stack))
			return stacks

		for pi in range(layout.num_fov):
			sub = np.take(raw, pi, axis=layout.fov_axis)
			# After slicing off FOV, axis indices shift down by 1 for axes > 0
			frames_axis = _axis_after_drop(layout.frames_axis, layout.fov_axis)
			y_axis = _axis_after_drop(layout.y_axis, layout.fov_axis)
			x_axis = _axis_after_drop(layout.x_axis, layout.fov_axis)
			c_axis = (_axis_after_drop(layout.c_axis, layout.fov_axis) if layout.c_axis is not None else None)

			stack = _move_axes_to_frames_yx(sub, frames_axis=frames_axis, y_axis=y_axis, x_axis=x_axis, c_axis=c_axis)
			stacks.append(GroupedStack(tag=f"p{pi:02d}_c00", stack=stack))
		return stacks

	_log(log_cb, f"Detected nice layout without FOV: raw_shape={raw.shape}")
	stack = _move_axes_to_frames_yx(raw, frames_axis=layout.frames_axis, y_axis=layout.y_axis, x_axis=layout.x_axis, c_axis=layout.c_axis)
	stacks.append(GroupedStack(tag="c00", stack=stack))
	return stacks

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
		self._overwrite_ok_from_now_on = False

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
	
	def _confirm_overwrite_if_needed(self, path: str) -> bool:
		"""
		Returns True if we should proceed with writing, False if the user cancelled.
		"""
		if self._overwrite_ok_from_now_on:
			return True
		if not os.path.exists(path):
			return True

		box = QtWidgets.QMessageBox(self)
		box.setIcon(QtWidgets.QMessageBox.Warning)
		box.setWindowTitle("File exists")
		box.setText("The output file already exists.")
		box.setInformativeText(path)
		# Buttons
		btn_cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
		btn_overwrite = box.addButton("Overwrite", QtWidgets.QMessageBox.AcceptRole)
		btn_overwrite_all = box.addButton("Overwriting is OK from now on", QtWidgets.QMessageBox.AcceptRole)
		box.setDefaultButton(btn_cancel)

		box.exec_()
		clicked = box.clickedButton()

		if clicked == btn_cancel:
			return False
		if clicked == btn_overwrite_all:
			self._overwrite_ok_from_now_on = True
		return True

	def _trim_frames_to_sim_multiple(self, stack: np.ndarray, *, num_shifts: int, o: int = 3) -> np.ndarray:
		"""
		If stack.shape[0] isn't divisible by (o*num_shifts), trim trailing frames
		down to the nearest divisible count and warn.

		No reordering, no padding, no flattening.
		"""
		n = int(stack.shape[0])
		den = int(o) * int(num_shifts)
		if den <= 0 or n <= 0:
			return stack

		rem = n % den
		if rem == 0:
			return stack

		new_n = n - rem
		if new_n <= 0:
			self.append_log(f"Frame count {n} not divisible by {den} and cannot trim safely (new_n={new_n}). Proceeding untrimmed.")
			return stack

		self.append_log(f"WARNING: frames_total={n} not divisible by (o*s)={den}. Trimming trailing {rem} frames -> {new_n}.")
		return stack[:new_n]

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
		if ext in (".tif", ".tiff") or path.lower().endswith((".ome.tif", ".ome.tiff")):
			rr = read_tiff_stack(path)
			return rr

		raise ValueError("Unsupported input type for MVP. Use .tif/.tiff/.ome.tif/.ome.tiff or .npy")
	
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

			if not self._confirm_overwrite_if_needed(out_path):
				self.append_log("Cancelled by user (overwrite prompt).")
				return	# stops continuing/exporting anything else

			write_tiff_stack(out_path, block)
			self.append_log(f"[{tag}] Wrote: {out_path} shape={block.shape} dtype={block.dtype}")
			
	def run_split(self) -> None:
		self.log.clear()
		try:
			state = self.get_state()
			loaded = self.load_stack(state.input_path)
			base = os.path.splitext(os.path.basename(state.input_path))[0]

			rr = loaded
			raw = rr.raw

			# 1) Strict “nice layout” detection by shape alone
			groups = group_stacks_nice_first(raw, log_cb=self.append_log)

			# 3) Export each group
			for g in groups:
				stack = g.stack

				# Ensure we have a frames axis at dim 0 for split_tiff_stack_by_z.
				# group_stacks_* should already guarantee this, but keep a guard.
				if stack.ndim < 3:
					self.append_log(f"ERROR: Group {g.tag} stack has unexpected ndim={stack.ndim}, shape={stack.shape}")
					continue

				# If frames_total doesn't fit (o*s), trim trailing frames (your “dropped off end” idea)
				stack = self._trim_frames_to_sim_multiple(
					stack,
					num_shifts=state.num_shifts,
					o=3,
				)

				self.export_stack(
					base=base,
					out_dir=state.output_dir,
					tag=g.tag,
					stack=stack,
					num_shifts=state.num_shifts,
					order=state.order,
				)

			self.append_log("Done.")
			return

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
