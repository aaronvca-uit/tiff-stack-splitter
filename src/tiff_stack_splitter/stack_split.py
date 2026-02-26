from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple, Union

import numpy as np

AxisName = Literal["s", "z", "o"]
OrderSpec = Union[str, Sequence[AxisName]]


@dataclass(frozen=True)
class StackSplitResult:
	# Canonical view: (z, o, s, *image_dims)
	canonical: np.ndarray
	# One item per z: each is either (o, s, *image_dims) or flattened to (o*s, *image_dims)
	per_z: List[np.ndarray]
	# The inferred z count
	z: int
	# Echo back the interpreted order
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
	flatten_og: bool = True,
) -> StackSplitResult:
	"""
	Splits a TIFF-like frame stack into per-z stacks, given:
	- o (orientations/angles) is always 3
	- s (phases/shifts) is 3 or 5
	- z inferred from n_frames / (o*s)
	- 'order' specifies how frames are arranged in the flattened sequence,
	  e.g. "s-z-o" means s varies fastest, then z, then o.

	Parameters
	----------
	stack:
		np.ndarray shaped like (n_frames, *image_dims).
	num_shifts:
		3 (2D option) or 5 (3D option).
	order:
		Permutation of 's', 'z', 'o' describing the frame order.
	flatten_og:
		If True, each per-z item is returned flattened to (o*s, *image_dims).
		If False, each per-z item is (o, s, *image_dims).

	Returns
	-------
	StackSplitResult with:
	- canonical: (z, o, s, *image_dims)
	- per_z: list length z
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

	if n_frames % denom != 0:
		raise ValueError(
			"Cannot infer z because n_frames is not divisible by (o*s). "
			f"Got n_frames={n_frames}, o={o}, s={s}, o*s={denom}."
		)

	z = n_frames // denom

	# User 'order' is interpreted as FASTEST -> SLOWEST (inner loop -> outer loop),
	# matching your example: order="s-z-o" => o slowest, z middle, s fastest.
	user_order = _parse_order(order)

	reshape_order = tuple(reversed(user_order))  # SLOWEST -> FASTEST for np.reshape
	size_by_axis: Dict[AxisName, int] = {"s": s, "z": z, "o": o}
	reshape_dims = tuple(size_by_axis[a] for a in reshape_order) + tuple(stack.shape[1:])

	reshaped = stack.reshape(reshape_dims)

	# Now transpose to canonical (z, o, s, ...)
	idx_by_axis = {a: reshape_order.index(a) for a in ("s", "z", "o")}
	transpose_axes = (idx_by_axis["z"], idx_by_axis["o"], idx_by_axis["s"]) + tuple(range(3, reshaped.ndim))
	canonical = np.transpose(reshaped, axes=transpose_axes)

	# Split into per-z stacks
	per_z: List[np.ndarray] = []
	for zi in range(z):
		block = canonical[zi]  # (o, s, *imgdims)
		if flatten_og:
			# Flatten (o,s) -> (o*s) in a consistent order: o major, s minor
			block = block.reshape((o * s,) + block.shape[2:])
		per_z.append(block)

	return StackSplitResult(
		canonical=canonical,
		per_z=per_z,
		z=z,
		order=user_order,  # keep reporting what the user provided
	)
