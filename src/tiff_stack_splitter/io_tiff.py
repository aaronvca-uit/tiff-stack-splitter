from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tifffile


@dataclass(frozen=True)
class ReadResult:
	stack: np.ndarray
	# Keep some info for writing
	is_ome: bool
	ome_xml: Optional[str]
	axes: Optional[str]
	metadata: Dict[str, Any]


def read_tiff_stack(path: str) -> ReadResult:
	with tifffile.TiffFile(path) as tf:
		is_ome = bool(getattr(tf, "is_ome", False))
		ome_xml = None
		axes = None
		metadata: Dict[str, Any] = {}

		# tifffile can parse OME-XML if present
		if is_ome:
			try:
				ome_xml = tf.ome_metadata
			except Exception:
				ome_xml = None

		# Read as numpy without dtype conversion
		arr = tf.asarray()

		# We want (frames, *image_dims). If a file is already a stack, great.
		# If it’s a single image, make it a 1-frame stack.
		if arr.ndim == 2 or arr.ndim == 3:
			# Could be (Y,X) or (Y,X,C) single image
			arr = arr[np.newaxis, ...]
		elif arr.ndim >= 4:
			# Could be (T,Z,C,Y,X) etc. We do NOT try to interpret scientific axes in MVP.
			# Instead, we flatten leading dims into "frames" while preserving the trailing image dims.
			# Heuristic: treat the last 2 dims as Y,X; if last 3 dims looks like color, keep it.
			# This keeps dtype intact but makes the notion of "frame" explicit.
			pass

		# If arr is something like (A,B,Y,X) or (A,B,Y,X,C), flatten A,B -> frames.
		if arr.ndim >= 4:
			# Decide if last dim is color-like
			last_dim = arr.shape[-1]
			looks_color = last_dim in (3, 4) and arr.ndim >= 3

			if looks_color:
				# Keep (..., Y, X, C)
				lead = arr.shape[:-3]
				img = arr.shape[-3:]
				arr = arr.reshape((int(np.prod(lead)),) + img)
			else:
				# Keep (..., Y, X)
				lead = arr.shape[:-2]
				img = arr.shape[-2:]
				arr = arr.reshape((int(np.prod(lead)),) + img)

		# Minimal metadata: dtype and shape (useful for debugging)
		metadata["dtype"] = str(arr.dtype)
		metadata["shape"] = tuple(int(x) for x in arr.shape)

		return ReadResult(
			stack=arr,
			is_ome=is_ome,
			ome_xml=ome_xml,
			axes=axes,
			metadata=metadata,
		)


def write_tiff_stack(path: str, stack: np.ndarray, *, ome_xml: Optional[str] = None) -> None:
	# Do not change dtype. tifffile will write using the array’s dtype.
	# We keep it simple in MVP: write as a standard TIFF stack.
	#
	# If you want OME-TIFF writing later, we can add tifffile.imwrite(..., metadata={"axes": ...}, ome=True)
	# but that requires we know the correct axes string.
	if ome_xml:
		# We’re not rebuilding OME metadata in MVP; keeping this placeholder avoids giving a false sense
		# of correctness. We intentionally ignore ome_xml for now.
		pass

	tifffile.imwrite(path, stack)
