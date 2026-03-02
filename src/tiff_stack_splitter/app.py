from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

from .io_tiff import read_tiff_stack, write_tiff_stack
from .stack_split import split_tiff_stack_by_z

OrderChoice = Literal["s-z-o", "s-o-z", "z-s-o", "z-o-s", "o-s-z", "o-z-s"]


@dataclass(frozen=True)
class UIState:
	input_path: str
	output_dir: str
	num_shifts: Literal[3, 5]
	order: OrderChoice
	flatten: bool


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

		self.flatten_check = QtWidgets.QCheckBox("Flatten (o,s) into frames (o*s, …)")
		self.flatten_check.setChecked(True)

		note = QtWidgets.QLabel(
			"Note: z is inferred as n_frames / (3 * s). "
			"Choose 2D if each z has 3*3 frames; choose 3D if each z has 3*5 frames."
		)
		note.setWordWrap(True)

		opts.addWidget(QtWidgets.QLabel("Mode:"), 0, 0)
		opts.addWidget(self.mode_combo, 0, 1)
		opts.addWidget(QtWidgets.QLabel("Order:"), 0, 2)
		opts.addWidget(self.order_combo, 0, 3)
		opts.addWidget(self.flatten_check, 1, 0, 1, 4)
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
		flatten = bool(self.flatten_check.isChecked())

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
			flatten=flatten,
		)

	def load_stack(self, path: str) -> Tuple[np.ndarray, str]:
		ext = os.path.splitext(path.lower())[1]
		if ext == ".npy":
			arr = np.load(path, allow_pickle=False)
			if arr.ndim < 2:
				raise ValueError(f".npy must have at least 2 dims (frames, image...). Got {arr.shape}")
			return arr, "npy"
		if ext in (".tif", ".tiff"):
			rr = read_tiff_stack(path)
			return rr.stack, "tiff"
		if path.lower().endswith(".ome.tif") or path.lower().endswith(".ome.tiff"):
			rr = read_tiff_stack(path)
			return rr.stack, "tiff"
		raise ValueError("Unsupported input type for MVP. Use .tif/.tiff/.ome.tif/.ome.tiff or .npy")

	def run_split(self) -> None:
		self.log.clear()
		try:
			state = self.get_state()
			self.append_log(f"Reading: {state.input_path}")
			stack, kind = self.load_stack(state.input_path)

			self.append_log(f"Loaded stack shape={stack.shape}, dtype={stack.dtype}")
			self.append_log(f"Splitting with s={state.num_shifts}, o=3, order={state.order}, flatten={state.flatten}")

			res = split_tiff_stack_by_z(
				stack,
				num_shifts=state.num_shifts,
				order=state.order,
				flatten_og=state.flatten,
			)

			self.append_log(f"Inferred z={res.z}")
			self.append_log(f"Canonical shape (z,o,s,...)={res.canonical.shape}")

			# Save per-z stacks
			base = os.path.splitext(os.path.basename(state.input_path))[0]
			for zi, block in enumerate(res.per_z):
				out_name = f"{base}_z{zi:03d}.tif"
				out_path = os.path.join(state.output_dir, out_name)
				write_tiff_stack(out_path, block)
				self.append_log(f"Wrote: {out_path} shape={block.shape} dtype={block.dtype}")

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
