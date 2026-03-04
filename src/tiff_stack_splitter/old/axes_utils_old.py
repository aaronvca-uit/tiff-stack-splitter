from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

@dataclass(frozen=True)
class GroupedStack:
	tag: str				# e.g. "c00", "p01_c00"
	stack: np.ndarray		# (frames, Y, X) or (frames, ...)

def _move_axis_to_front(arr: np.ndarray, axis: int) -> np.ndarray:
	if axis == 0:
		return arr
	return np.moveaxis(arr, axis, 0)

def _flatten_to_frames(arr: np.ndarray, axes: str) -> np.ndarray:
	"""
	Given an array and an axis-label string:
	Find the indices of Y and X.
	If there’s a C axis, treat it as RGB(A) only when:
	C size is 3 or 4,
	Y and X are “large” (≥64),
	and C is the last axis.
	Then:
	If RGB(A): transpose so all non-(Y,X,C) axes come first, followed by (Y,X,C), flatten the leading axes into a single frames dimension, and reshape to (frames, Y, X, C).
	Otherwise: transpose so all non-(Y,X) axes come first, followed by (Y,X), flatten the leading axes into frames, and reshape to (frames, Y, X).
	In both cases, make the transposed array contiguous before reshaping.
	"""
	y_i = axes.index("Y")
	x_i = axes.index("X")

	c_i = axes.index("C") if "C" in axes else -1
	c_is_rgb = False
	if c_i != -1:
		c = int(arr.shape[c_i])
		y = int(arr.shape[y_i])
		x = int(arr.shape[x_i])
		c_is_rgb = (c in (3, 4) and x >= 64 and y >= 64 and c_i == (len(axes) - 1))

	if c_is_rgb:
		keep = {y_i, x_i, c_i}
		perm = [i for i in range(arr.ndim) if i not in keep] + [y_i, x_i, c_i]
		arr2 = np.transpose(arr, axes=perm)
		lead = arr2.shape[:-3]
		img = arr2.shape[-3:]  # (Y,X,C)
		frames = int(np.prod(lead)) if lead else 1
		arr2 = np.ascontiguousarray(arr2)
		return arr2.reshape((frames,) + img)

	perm = [i for i in range(arr.ndim) if i not in (y_i, x_i)] + [y_i, x_i]
	arr2 = np.transpose(arr, axes=perm)
	lead = arr2.shape[:-2]
	img = arr2.shape[-2:]
	frames = int(np.prod(lead)) if lead else 1
	arr2 = np.ascontiguousarray(arr2)
	return arr2.reshape((frames,) + img)

def _axis_looks_like_rgb_samples(*, raw: np.ndarray, axes: str, key: str) -> bool:
	# Treat "S" or "C" of size 3/4 at the end as image samples (RGB/RGBA), not a grouping axis.
	if key not in axes:
		return False
	i = axes.index(key)
	if i != (len(axes) - 1):
		return False
	c = int(raw.shape[i])
	if c not in (3, 4):
		return False
	if "Y" not in axes or "X" not in axes:
		return False
	y = int(raw.shape[axes.index("Y")])
	x = int(raw.shape[axes.index("X")])
	return (y >= 64 and x >= 64)

def _pick_confident_fov_key(
	raw: np.ndarray,
	axes: str,
	*,
	fov_axis_keys: Sequence[str],
	channel_axis_keys: Sequence[str],
) -> str | None:
	# Goal: protect the easy path (single group). Only split if we're reasonably sure.
	#
	# Rules:
	# - Never use "S" as FOV (often samples/RGB). (Even if passed in, this rejects it.)
	# - Never use an axis that looks like RGB(A) samples.
	# - Only split if the axis exists AND its size > 1.
	# - Prefer earlier keys in fov_axis_keys order.
	for k in fov_axis_keys:
		if k == "S":
			continue
		if k not in axes:
			continue
		try:
			n = int(raw.shape[axes.index(k)])
		except Exception:
			continue
		if n <= 1:
			continue
		# If this key looks like RGB samples, do NOT treat it as FOV.
		if _axis_looks_like_rgb_samples(raw=raw, axes=axes, key=k):
			continue
		# Also: don't treat the chosen FOV key as a channel key.
		if k in channel_axis_keys:
			continue
		return k
	return None

def _axis_len(raw: np.ndarray, axes: str, key: str) -> int:
	# Returns axis length if present, else 1
	if key not in axes:
		return 1
	return int(raw.shape[axes.index(key)])

def _is_rgb_like_c(raw: np.ndarray, axes: str, c_key: str) -> bool:
	# Decide if C is actually RGB(A) samples and should be treated as image dims
	if c_key not in axes:
		return False
	c_i = axes.index(c_key)

	# Require XY to exist
	if "Y" not in axes or "X" not in axes:
		return False

	c = int(raw.shape[c_i])
	y = int(raw.shape[axes.index("Y")])
	x = int(raw.shape[axes.index("X")])

	# Match your heuristic: small C + image-like XY + C last
	return (c in (3, 4) and x >= 64 and y >= 64 and c_i == (len(axes) - 1))

def _should_fast_path_single_group(
	raw: np.ndarray,
	axes: str,
	*,
	channel_axis_keys: Sequence[str],
	fov_axis_keys: Sequence[str],
) -> bool:
	# 1) If any FOV axis exists with length > 1, not a single-group easy case
	for k in fov_axis_keys:
		if k in axes and _axis_len(raw, axes, k) > 1:
			return False

	# 2) If any channel axis exists with length > 1, it is multi-channel
	#    EXCEPT: allow RGB(A) channel-last to still count as “single group”
	for k in channel_axis_keys:
		if k in axes:
			nc = _axis_len(raw, axes, k)
			if nc > 1 and not _is_rgb_like_c(raw, axes, k):
				return False

	return True

def group_stacks_from_axes(
	raw: np.ndarray,
	axes: str,
	*,
	channel_axis_keys: Sequence[str] = ("C",),
	fov_axis_keys: Sequence[str] = ("P", "S", "V"),
) -> List[GroupedStack]:
	"""
	Given an array and its axis labels, split it into one or more GroupedStack outputs based on channel and field-of-view axes.
	Fast path: if the data has no confident multi-FOV and no real multi-channel (per _should_fast_path_single_group), flatten everything into (frames, Y, X[, C]) and return a single group tagged "c00".
	Otherwise:
	Identify a channel axis key present in axes (from channel_axis_keys).
	Pick a confident FOV axis key (from fov_axis_keys) via _pick_confident_fov_key.
	If an FOV key is found:
	Move that axis to the front.
	For each FOV index, remove the FOV label from the axis string and group channels within that sub-array via _group_channels, tagging groups with a prefix like "p00_", "p01_", etc.
	Return all collected groups.
	If no FOV key is found:
	Group channels on the full array via _group_channels (no prefix) and return the result.
	calls: _should_fast_path_single_group, _flatten_to_frames, 
	_pick_confident_fov_key, _move_axis_to_front, _group_channels
	"""
	# --- FAST PATH: treat as a single group if no multi-FOV and no true multi-channel ---
	if _should_fast_path_single_group(
		raw,
		axes,
		channel_axis_keys=channel_axis_keys,
		fov_axis_keys=fov_axis_keys,
	):
		frames = _flatten_to_frames(raw, axes)
		return [GroupedStack(tag="c00", stack=frames)]

	# --- Existing behavior below (multi-FOV or multi-channel) ---
	c_key = next((k for k in channel_axis_keys if k in axes), None)
	p_key = _pick_confident_fov_key(
		raw,
		axes,
		fov_axis_keys=fov_axis_keys,
		channel_axis_keys=channel_axis_keys,
	)

	stacks: List[GroupedStack] = []
	if p_key:
		p_i = axes.index(p_key)
		raw_p = _move_axis_to_front(raw, p_i)
		for pi in range(raw_p.shape[0]):
			sub = raw_p[pi]
			sub_axes = axes.replace(p_key, "")
			stacks.extend(_group_channels(sub, sub_axes, c_key=c_key, prefix=f"p{pi:02d}_"))
		return stacks
	return _group_channels(raw, axes, c_key=c_key, prefix="")

def _group_channels(raw: np.ndarray, axes: str, *, c_key: str | None, prefix: str) -> List[GroupedStack]:
	stacks: List[GroupedStack] = []

	# If C is actually RGB(A), do NOT split into channel groups.
	if c_key and c_key in axes:
		c_i = axes.index(c_key)
		c = int(raw.shape[c_i])
		y = int(raw.shape[axes.index("Y")]) if "Y" in axes else 0
		x = int(raw.shape[axes.index("X")]) if "X" in axes else 0
		c_is_rgb = (c in (3, 4) and x >= 64 and y >= 64 and c_i == (len(axes) - 1))
		if c_is_rgb:
			frames = _flatten_to_frames(raw, axes)
			stacks.append(GroupedStack(tag=f"{prefix}c00", stack=frames))
			return stacks

	if c_key and c_key in axes:
		c_i = axes.index(c_key)
		raw_c = _move_axis_to_front(raw, c_i)
		sub_axes = axes.replace(c_key, "")
		for ci in range(raw_c.shape[0]):
			frames = _flatten_to_frames(raw_c[ci], sub_axes)
			stacks.append(GroupedStack(tag=f"{prefix}c{ci:02d}", stack=frames))
	else:
		frames = _flatten_to_frames(raw, axes)
		stacks.append(GroupedStack(tag=f"{prefix}c00", stack=frames))
	return stacks