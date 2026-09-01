"""Resolve PDD sign codes → embedding ids for PlanT2 (route path / env).

Index 0 is reserved for unknown / missing. Does not read boxes or mutate cache.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Codes used in plant2_l1_fv_experts_split (+ common eval signs).
SIGN_CODES: tuple[str, ...] = (
    "2.1",
    "2.3.1",
    "2.3.2",
    "2.3.3",
    "2.4",
    "2.5",
    "3.1",
    "3.2",
    "3.24",
    "4.2.1",
    "4.2.2",
    "4.2.3",
    "4.3",
    "4.6",
    "5.7.1",
    "5.7.2",
    "5.15.1",
    "5.15.2",
    "5.19",
    "5.21",
    "5.31",
)

# Plates that carry a number, and the numbers they can carry (km/h). The code
# alone cannot say WHICH limit applies: 3.24 at 20 and 3.24 at 40 demand
# opposite behaviour from the same class, and the spatial token only says so
# while the sign is in frame. The route-level id therefore distinguishes them.
SIGN_VALUE_CODES: tuple[str, ...] = ("3.24", "5.31", "4.6")
SIGN_VALUES_KMH: tuple[int, ...] = (20, 30, 40, 50, 60)


def qualified_sign_code(code: Optional[str], value_kmh=None) -> Optional[str]:
    """`"3.24"` + 20 -> `"3.24@20"`; anything without a number stays as it is."""
    if not code:
        return code
    code = str(code).strip()
    if value_kmh is None or code not in SIGN_VALUE_CODES:
        return code
    try:
        v = int(round(float(value_kmh)))
    except (TypeError, ValueError):
        return code
    if v not in SIGN_VALUES_KMH:
        v = min(SIGN_VALUES_KMH, key=lambda k: abs(k - v))
    return f"{code}@{v}"


# The id vocabulary: every plain code, then one entry per (valued code, value).
SIGN_ID_VOCAB: tuple[str, ...] = SIGN_CODES + tuple(
    f"{c}@{v}" for c in SIGN_VALUE_CODES for v in SIGN_VALUES_KMH
)

# 0 = unknown; 1..N = SIGN_ID_VOCAB
SIGN_CATS: dict[str, int] = {code: i + 1 for i, code in enumerate(SIGN_ID_VOCAB)}
NUM_SIGN_CLASSES: int = 1 + len(SIGN_ID_VOCAB)

# Plates that prescribe a floor rather than a ceiling: the driver must stay
# ABOVE the number. Every other speed plate is a ceiling, and the two demand
# opposite margins from the same kind of label.
MIN_SPEED_CODES: tuple[str, ...] = ("4.6",)


def base_sign_code(sign_id: int) -> Optional[str]:
    """The plain code behind an embedding index, without the plate value."""
    try:
        idx = int(sign_id)
    except (TypeError, ValueError):
        return None
    if idx <= 0 or idx > len(SIGN_ID_VOCAB):
        return None
    return SIGN_ID_VOCAB[idx - 1].split("@", 1)[0]

_EXPERTS_ROOT = Path("/home/jovyan/shares/SR006.nfs3/shepelev/collected_trajectories")
_EXPERT_JSONLS = (
    _EXPERTS_ROOT / "traj-priority-signs/traj_yield_2_4_train80/experts/experts_scene_uid_top1.jsonl",
    _EXPERTS_ROOT / "traj-priority-signs/traj_stop_2_5_train80/experts/experts_scene_uid_top1.jsonl",
    _EXPERTS_ROOT / "traj-priority-signs/traj_main_2_1_train80/experts/experts_scene_uid_top1.jsonl",
    _EXPERTS_ROOT / "traj-priority-signs/traj_secondary_2_3_train80/experts/experts_scene_uid_top1.jsonl",
    _EXPERTS_ROOT / "traj-priority-signs/traj_roundabout_4_3_train80/experts/experts_scene_uid_top1.jsonl",
    _EXPERTS_ROOT / "traffic-rule-bench-traj/experts_detour_train80/experts_scene_uid_top1.jsonl",
    _EXPERTS_ROOT / "traj-priority-signs/traj_lane_5_15_train80/experts/experts_scene_uid_top1.jsonl",
)

_SUMO_SIGN_RE = re.compile(
    r"sumo_(2\.1|2\.3\.[123]|2\.4|2\.5|3\.1|3\.2|3\.24|4\.2\.[123]|4\.3|4\.6|"
    r"5\.7\.[12]|5\.15\.[12]|5\.19|5\.21|5\.31)_"
)


def sign_code_to_id(code: Optional[str], value_kmh=None) -> int:
    """Embedding index for a code, optionally qualified by the plate's number.

    An unknown pairing falls back to the plain code rather than to 0: losing the
    value is a loss of precision, losing the sign is a loss of the sign.
    """
    if not code:
        return 0
    q = qualified_sign_code(code, value_kmh)
    return SIGN_CATS.get(q, SIGN_CATS.get(str(code).strip(), 0))


def sign_of_route_name(name: str) -> Optional[str]:
    """Cheap parse from route / scene_uid string (no disk I/O)."""
    if "_rb_" in name or re.match(r"sign_\d+_rb_", name):
        return "4.3"
    m = _SUMO_SIGN_RE.search(name)
    if m:
        return m.group(1)
    m = re.match(r"sumo_(3\.24|4\.6|5\.21|5\.31|4\.2\.[123])_", name)
    if m:
        return m.group(1)
    return None


@lru_cache(maxsize=1)
def load_uid2sign() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in _EXPERT_JSONLS:
        if not path.is_file():
            continue
        with path.open() as f:
            for line in f:
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                uid = o.get("scene_uid")
                sign = str(o.get("sign") or "")
                if uid and sign:
                    mapping[str(uid)] = sign
    return mapping


def load_split_meta_route2sign(split_meta_path: Path) -> dict[str, str]:
    """Invert split_meta['val'][sign] → route lists (+ optional train if present)."""
    out: dict[str, str] = {}
    if not split_meta_path.is_file():
        return out
    try:
        meta = json.loads(split_meta_path.read_text())
    except Exception:
        return out
    for bucket in ("val", "train"):
        block = meta.get(bucket) or {}
        if not isinstance(block, dict):
            continue
        for sign, routes in block.items():
            if not isinstance(routes, list):
                continue
            for name in routes:
                out[str(name)] = str(sign)
    return out


def resolve_route_sign(name: str, uid2sign: Optional[dict[str, str]] = None) -> Optional[str]:
    s = sign_of_route_name(name)
    if s:
        return s
    uid2sign = uid2sign if uid2sign is not None else load_uid2sign()
    for var in ("default", "s1", "s2", "s3", "s4"):
        suf = "_" + var
        if name.endswith(suf):
            uid = name[: -len(suf)]
            if uid in uid2sign:
                return uid2sign[uid]
    return uid2sign.get(name)


def route_name_from_label_path(label_path: str) -> str:
    """.../data/<route>/boxes/0003.json.gz → <route>."""
    return Path(label_path).parent.parent.name


def resolve_sign_id_for_route(route_name: str, extra_map: Optional[dict[str, str]] = None) -> int:
    if extra_map and route_name in extra_map:
        return sign_code_to_id(extra_map[route_name])
    return sign_code_to_id(resolve_route_sign(route_name))


def _probe_indices(n: int, probes: int = 60) -> list[int]:
    """Evenly spaced frame indices to search, cheapest-first coverage."""
    if n <= 0:
        return []
    if n <= probes:
        return list(range(n))
    step = n / float(probes)
    return sorted({min(n - 1, int(i * step)) for i in range(probes)})


@lru_cache(maxsize=None)
def sniff_sign_from_route(route_dir: str) -> Optional[str]:
    """Read the PDD code out of a route's own boxes.

    Route naming is not a reliable key: whole generations of dumps use names the
    uid map never learned (every 2.5 route in the mixture split resolved to
    "unknown"), which silently zeroes the sign token for exactly the scenes the
    sign matters in. The boxes carry the code as the object class, so the frames
    can answer when the name cannot. One small gz read per route, cached.
    """
    import gzip

    boxes = sorted(Path(route_dir).glob("boxes/*.json.gz"))
    if not boxes:
        return None
    known = set(SIGN_CODES)
    # A sign occupies a window of the route, not the whole of it, and with a
    # tight visibility radius that window can be one per cent of the frames —
    # five probes would miss it and the route would silently train with no sign.
    # Scan evenly and stop at the first hit; the read ends as soon as it works.
    for idx in _probe_indices(len(boxes)):
        try:
            with gzip.open(boxes[idx], "rt") as fh:
                frame = json.load(fh)
        except Exception:
            continue
        for obj in frame:
            code = obj.get("pdd_code") or obj.get("class")
            if isinstance(code, str) and code in known:
                return code
    return None


@lru_cache(maxsize=None)
def sniff_sign_value_from_route(route_dir: str):
    """The number on the route's plate (km/h), read from its own boxes.

    The route name never carries it — `sumo_3.24_1507406_…` is the same string
    whether the plate says 20 or 40 — so the value has to come from the frames,
    where the dump writes it as `sign_value_kmh`. Returns None when the plate
    carries no number.
    """
    import gzip

    boxes = sorted(Path(route_dir).glob("boxes/*.json.gz"))
    if not boxes:
        return None
    for idx in _probe_indices(len(boxes)):
        try:
            with gzip.open(boxes[idx], "rt") as fh:
                frame = json.load(fh)
        except Exception:
            continue
        for obj in frame:
            code = obj.get("pdd_code") or obj.get("class")
            if isinstance(code, str) and code in SIGN_VALUE_CODES:
                val = obj.get("sign_value_kmh")
                if val is not None:
                    return float(val)
    return None
