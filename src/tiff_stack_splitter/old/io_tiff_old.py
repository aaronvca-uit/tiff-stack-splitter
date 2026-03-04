from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import tifffile


@dataclass(frozen=True)
class ReadResult:
	raw: np.ndarray				# Raw N-D array from tifffile series
	stack: np.ndarray			# Your existing flattened (frames, ...) view
	is_ome: bool
	ome_xml: Optional[str]
	axes: Optional[str]			# Series axes if available
	axes_reliable: bool
	metadata: Dict[str, Any]


def read_tiff_stack(path: str) -> ReadResult:
	with tifffile.TiffFile(path) as tf:
		is_ome = bool(getattr(tf, "is_ome", False))
		ome_xml = None
		metadata: Dict[str, Any] = {}

		if is_ome:
			try:
				ome_xml = tf.ome_metadata
			except Exception:
				ome_xml = None

		series = tf.series[0]
		raw = series.asarray()
		axes = getattr(series, "axes", None)
		series_shape = getattr(series, "shape", None)

		axes_reliable = bool(
			axes
			and series_shape
			and len(axes) == len(series_shape)
			and ("X" in axes and "Y" in axes)
		)

		metadata["series_axes"] = axes
		metadata["series_shape"] = tuple(int(x) for x in series_shape) if series_shape else None
		metadata["axes_reliable"] = bool(axes_reliable)

		arr = raw

		def _as_frames_using_axes(a: np.ndarray, ax: str) -> np.ndarray:
			y_i = ax.index("Y")
			x_i = ax.index("X")

			# Treat C as image channel if it looks like RGB(A) and is adjacent to X/Y in a typical way.
			c_i = ax.index("C") if "C" in ax else -1
			c_is_rgb = False
			if c_i != -1:
				c = int(a.shape[c_i])
				y = int(a.shape[y_i])
				x = int(a.shape[x_i])
				# Heuristic: small channel count + reasonably image-like XY + C at end is common.
				c_is_rgb = (c in (3, 4) and x >= 64 and y >= 64 and c_i == (len(ax) - 1))

			if c_is_rgb:
				# Keep (Y,X,C) as the image plane; flatten everything else into frames.
				keep = {y_i, x_i, c_i}
				perm = [i for i in range(a.ndim) if i not in keep] + [y_i, x_i, c_i]
				a2 = np.transpose(a, axes=perm)
				lead = a2.shape[:-3]
				img = a2.shape[-3:]  # (Y,X,C)
				frames = int(np.prod(lead)) if lead else 1
				a2 = np.ascontiguousarray(a2)
				return a2.reshape((frames,) + img)

			# Default: treat image as (Y,X); flatten everything else into frames.
			perm = [i for i in range(a.ndim) if i not in (y_i, x_i)] + [y_i, x_i]
			a2 = np.transpose(a, axes=perm)
			lead = a2.shape[:-2]
			img = a2.shape[-2:]
			frames = int(np.prod(lead)) if lead else 1
			a2 = np.ascontiguousarray(a2)
			return a2.reshape((frames,) + img)

		def _as_frames_heuristic(a: np.ndarray) -> np.ndarray:
			"""
			Axes-unaware fallback.
			Produces (frames, Y, X) or (frames, Y, X, C) when the last dim looks like channels.
			"""
			if a.ndim == 2:
				return a[np.newaxis, ...]

			if a.ndim == 3:
				# If it looks like (Y, X, C) with C=3/4, treat it as a single frame.
				if int(a.shape[-1]) in (3, 4) and int(a.shape[-2]) >= 64 and int(a.shape[-3]) >= 64:
					return a[np.newaxis, ...]
				# Otherwise assume (frames, Y, X)
				return a

			# ndim >= 4: decide whether the image payload is (Y,X) or (Y,X,C)
			last = int(a.shape[-1])
			y = int(a.shape[-3]) if a.ndim >= 3 else 0
			x = int(a.shape[-2]) if a.ndim >= 2 else 0

			looks_chlast = (last in (3, 4) and x >= 64 and y >= 64)
			img_nd = 3 if looks_chlast else 2

			lead = a.shape[:-img_nd]
			img = a.shape[-img_nd:]
			frames = int(np.prod(lead)) if lead else 1

			# Make it reshape-safe and predictable downstream.
			a = np.ascontiguousarray(a)
			return a.reshape((frames,) + img)

		# Prefer axes-driven normalization when it’s plausibly trustworthy.
		if axes_reliable and axes and ("Y" in axes) and ("X" in axes) and (len(axes) == arr.ndim):
			arr = _as_frames_using_axes(arr, axes)
		else:
			arr = _as_frames_heuristic(arr)

		metadata["dtype"] = str(arr.dtype)
		metadata["shape"] = tuple(int(x) for x in arr.shape)

		return ReadResult(
			raw=raw,
			stack=arr,
			is_ome=is_ome,
			ome_xml=ome_xml,
			axes=axes,
			axes_reliable=axes_reliable,
			metadata=metadata,
		)


def write_tiff_stack(path: str, stack: np.ndarray) -> None:
	tifffile.imwrite(path, stack)
