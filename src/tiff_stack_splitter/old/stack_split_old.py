from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple, Union

import numpy as np

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
