# new file: layout_infer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np

@dataclass(frozen=True)
class SpatialSpec:
	yx_axes: Tuple[int, int]			# indices in raw
	samples_axis: Optional[int]			# 3/4 channel samples, if present
	lead_axes: Tuple[int, ...]			# all other axes (excluding Y/X and samples)

def infer_spatial_axes(raw: np.ndarray) -> SpatialSpec:
	"""
	Given a NumPy array, determine which axes represent spatial dimensions.
	Require at least 2 dimensions; otherwise raise an error.
	If the last axis has size 3 or 4 and the two preceding axes are large (≥64), treat it as a color samples axis.
	From the remaining axes, select the two largest dimensions as spatial (Y, X).
	If either spatial dimension is smaller than 32, raise an error.
	All other non-spatial (and non-sample) axes are treated as leading axes.
	Return a SpatialSpec containing:
	the spatial axes (y, x),
	the optional samples axis,
	the remaining leading axes.
	"""
	shape = tuple(int(x) for x in raw.shape)
	if raw.ndim < 2:
		raise ValueError(f"raw must have >=2 dims, got shape={shape}")

	# Detect color samples: (..., Y, X, S) where S in {3,4} and Y/X look "big"
	samples_axis: Optional[int] = None
	if raw.ndim >= 3 and shape[-1] in (3, 4):
		# "two dims before it look spatial (big)"
		if shape[-2] >= 64 and shape[-3] >= 64:
			samples_axis = raw.ndim - 1

	# Candidate axes for spatial selection (exclude samples if present)
	candidate_axes = list(range(raw.ndim))
	if samples_axis is not None:
		candidate_axes.remove(samples_axis)

	# Pick the two largest dims among candidates as spatial
	sorted_axes = sorted(candidate_axes, key=lambda i: shape[i], reverse=True)
	y_i, x_i = sorted_axes[0], sorted_axes[1]

	# Basic sanity: spatial dims should be "big-ish"
	if shape[y_i] < 32 or shape[x_i] < 32:
		raise ValueError(f"Could not confidently identify spatial axes in shape={shape}")

	lead_axes = tuple(i for i in range(raw.ndim) if i not in (y_i, x_i) and i != samples_axis)
	return SpatialSpec(yx_axes=(y_i, x_i), samples_axis=samples_axis, lead_axes=lead_axes)

# layout_infer.py continued

@dataclass(frozen=True)
class ShapeInterpretation:
	num_shifts: int
	o_axis: int					# axis index in raw
	s_axis: int					# axis index in raw
	k_axis: Optional[int]		# axis index in raw (channels/FOV grouping), optional
	k: int
	z: int

def _prod(vals: Sequence[int]) -> int:
	out = 1
	for v in vals:
		out *= int(v)
	return int(out)

def generate_interpretations(
	raw: np.ndarray,
	*,
	o: int = 3,
	shifts: Sequence[int] = (3, 5),
	k_candidates: Sequence[int] = (2, 3, 4, 5, 6),
) -> List[ShapeInterpretation]:
	"""
	Using inferred spatial axes, consider only the remaining “lead” axes for layout interpretation.
	Among lead axes:
	Find axes of size o (default 3) as possible o axes.
	For each allowed shift count (3 or 5), find axes matching that size as possible s axes.
	For every distinct (o, s) pair:
	Treat all other lead axes as remaining dimensions.
	Optionally choose one remaining axis as a k axis if its size is in k_candidates; otherwise assume k = 1.
	Compute z as the product of all remaining sizes not used for o, s, or k.
	For each valid combination (with z > 0), create a ShapeInterpretation recording:
	shift count,
	raw indices of o, s, and optional k,
	k size,
	inferred z.
	Return all generated interpretations.
	calls: infer_spatial_axes, _prod
	"""
	spec = infer_spatial_axes(raw)
	shape = tuple(int(x) for x in raw.shape)

	lead_axes = spec.lead_axes
	lead_shape = [shape[i] for i in lead_axes]

	# Map "lead index" -> "raw axis index"
	raw_axis_for_lead = list(lead_axes)

	# Find lead indices with sizes that can serve as o or s
	o_lead_idxs = [li for li, sz in enumerate(lead_shape) if sz == o]

	out: List[ShapeInterpretation] = []

	for s in shifts:
		s_lead_idxs = [li for li, sz in enumerate(lead_shape) if sz == int(s)]
		# If s=3, s candidates overlap o candidates; that's OK but we must pick distinct axes
		for o_li in o_lead_idxs:
			for s_li in s_lead_idxs:
				if s_li == o_li:
					continue

				remaining_lis = [li for li in range(len(lead_shape)) if li not in (o_li, s_li)]
				remaining_sizes = [lead_shape[li] for li in remaining_lis]
				total_remaining = _prod(remaining_sizes) if remaining_sizes else 1

				# k-axis optional: choose none, or choose one remaining axis with size in k_candidates
				k_choices: List[Optional[int]] = [None]
				for li in remaining_lis:
					if lead_shape[li] in set(int(x) for x in k_candidates):
						k_choices.append(li)

				for k_li in k_choices:
					if k_li is None:
						k = 1
						z_total = total_remaining
					else:
						k = int(lead_shape[k_li])
						other_sizes = [lead_shape[li] for li in remaining_lis if li != k_li]
						z_total = _prod(other_sizes) if other_sizes else 1

					# z_total is "whatever else"; accept any positive integer z
					if z_total <= 0:
						continue

					out.append(
						ShapeInterpretation(
							num_shifts=int(s),
							o_axis=int(raw_axis_for_lead[o_li]),
							s_axis=int(raw_axis_for_lead[s_li]),
							k_axis=None if k_li is None else int(raw_axis_for_lead[k_li]),
							k=int(k),
							z=int(z_total),
						)
					)

	return out

def apply_interpretation_to_groups(raw: np.ndarray, it: ShapeInterpretation) -> List[np.ndarray]:
	"""
	Using inferred spatial axes and a chosen interpretation:
	Identify all axes used for o, s, spatial (Y, X), optional k, and optional samples.
	Treat all remaining axes as “other lead axes” and verify their size product equals the expected z.
	If not, raise an error.
	If a k axis is defined, verify its size matches the expected k.
	Reorder the array to:
	optional k
	collapsed z
	o
	s
	Y, X
	optional samples
	Validate that:
	o has size 3
	s matches the expected shift count
	Finally:
	Collapse z × o × s into a single frame dimension.
	If no k, return one reshaped array.
	If k exists, reshape to (k, frames, …) and return one array per k group.
	"""
	spec = infer_spatial_axes(raw)
	y_i, x_i = spec.yx_axes
	samp_i = spec.samples_axis
	shape = tuple(int(x) for x in raw.shape)

	# Build list of axes in desired order:
	# k (optional), remaining lead axes (collapsed into z), o, s, Y, X, samples(optional)
	used = {it.o_axis, it.s_axis, y_i, x_i}
	if it.k_axis is not None:
		used.add(it.k_axis)
	if samp_i is not None:
		used.add(samp_i)

	# "Other lead axes" are everything not used and not spatial/samples
	other_lead_axes = [ax for ax in range(raw.ndim) if ax not in used]

	# We collapse other_lead_axes into z; ensure product matches it.z
	other_prod = _prod([shape[ax] for ax in other_lead_axes]) if other_lead_axes else 1
	if other_prod != it.z:
		raise ValueError(
			f"Interpretation mismatch: expected z={it.z} but product(other_lead_axes)={other_prod} "
			f"for shape={shape} it={it}"
		)

	# k axis handling
	if it.k_axis is None:
		k_axes = []
		k = 1
	else:
		k_axes = [it.k_axis]
		k = int(shape[it.k_axis])
		if k != it.k:
			raise ValueError(f"Interpretation mismatch: expected k={it.k} but axis size is {k}")

	# Transpose
	order = k_axes + other_lead_axes + [it.o_axis, it.s_axis, y_i, x_i]
	if samp_i is not None:
		order.append(samp_i)

	arr = np.transpose(raw, axes=order)

	# Now arr has shape: (k?, z, o, s, Y, X[,S])
	# Collapse "z,o,s" into frames
	if it.k_axis is None:
		lead = (it.z, 3, it.num_shifts)
		start = 0
	else:
		lead = (it.k, it.z, 3, it.num_shifts)
		start = 1

	# Validate expected o/s sizes
	if arr.shape[start + 1] != 3:
		raise ValueError(f"Expected o=3 but got {arr.shape[start + 1]} after transpose")
	if arr.shape[start + 2] != it.num_shifts:
		raise ValueError(f"Expected s={it.num_shifts} but got {arr.shape[start + 2]} after transpose")

	# Reshape to groups
	if it.k_axis is None:
		frames = it.z * 3 * it.num_shifts
		return [arr.reshape((frames,) + arr.shape[3:])]
	else:
		frames = it.z * 3 * it.num_shifts
		arr2 = arr.reshape((it.k, frames) + arr.shape[4:])
		return [arr2[gi] for gi in range(it.k)]