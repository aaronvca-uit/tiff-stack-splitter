from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

@dataclass(frozen=True)
class PlausibleOption:
	num_shifts: Literal[3, 5]
	k_groups: int
	z: int

def plausible_options(
	frames_total: int,
	*,
	o: int = 3,
	shifts: Sequence[int] = (3, 5),
	k_candidates: Sequence[int] = (1, 2, 3, 4, 5, 6),
) -> List[PlausibleOption]:
	opts: List[PlausibleOption] = []
	for s in shifts:
		denom = o * int(s)
		for k in k_candidates:
			if k <= 0:
				continue
			if frames_total % k != 0:
				continue
			ft = frames_total // k
			if ft % denom != 0:
				continue
			z = ft // denom
			opts.append(PlausibleOption(num_shifts=s, k_groups=int(k), z=int(z)))
	return opts