import os
import logging
import json
import numpy as np
from pathlib import Path
from beartype import beartype

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

import functools
import mmap
import io
import gzip
import glob

from plant_variables import PlanTVariables
from util.static_extents import CAR_EXTENTS, STATIC_EXTENTS
from util.sign_id import (
    MIN_SPEED_CODES,
    SIGN_ID_VOCAB,
    base_sign_code,
    SIGN_VALUE_CODES,
    sign_of_route_name,
    sniff_sign_value_from_route,
    SIGN_CODES,
    load_split_meta_route2sign,
    load_uid2sign,
    resolve_sign_id_for_route,
    route_name_from_label_path,
    sign_code_to_id,
    sniff_sign_from_route,
)

from scipy.spatial import cKDTree

def normalize_angle_degree(x):
  x = x % 360.0
  if isinstance(x, np.ndarray):
      x[x > 180] -= 360
  elif x > 180.0:
    x -= 360.0
  return x

def rad2deg(theta):
    return normalize_angle_degree(np.rad2deg(theta).item())


# The future window makes one sample read seq_len + future_frames measurement
# files (41 with the 40-frame path target). On a network share at ~6 ms per file
# that is ~250 ms per sample and the loader starves the GPU. A route dumped with
# ``measurements_all.json.gz`` (frame stem -> measurement dict, written by
# scripts/plant2_ft_pipeline/data/consolidate_measurements.py) is read once per
# worker and kept in a small LRU; routes without it fall back to the per-file
# reads, so the two layouts can be mixed.
DETOUR_CODES = ("4.2.1", "4.2.2", "4.2.3")

_MEAS_ALL_NAME = "measurements_all.json.gz"

# A route packed by scripts/plant2_ft_pipeline/data/pack_routes.py carries all
# of its frame files in one pack.bin with a pack.json index. Reading a frame
# then is a slice of a memory-mapped file the page cache keeps in RAM, shared
# by every worker on the node, instead of a small read on the network share.
# Routes without a pack fall back to the plain files.
_PACK_BIN, _PACK_IDX = "pack.bin", "pack.json"


@functools.lru_cache(maxsize=4096)
def _route_pack(route_dir: str):
    idx = os.path.join(route_dir, _PACK_IDX)
    if not os.path.isfile(idx):
        return None
    with open(idx, "r", encoding="utf-8") as fh:
        index = json.load(fh)
    fh = open(os.path.join(route_dir, _PACK_BIN), "rb")
    view = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    return view, index


def _read_frame_bytes(path) -> bytes:
    """Bytes of ``<route>/<sub>/<name>``, from the route pack when there is one."""
    path = os.fsdecode(path)
    sub_dir, name = os.path.split(path)
    route_dir, sub = os.path.split(sub_dir)
    pack = _route_pack(route_dir)
    if pack is not None:
        entry = pack[1].get(f"{sub}/{name}")
        if entry is not None:
            off, length = entry
            return pack[0][off:off + length]
    with open(path, "rb") as fh:
        return fh.read()


def _load_gz_json(path):
    return json.loads(gzip.decompress(_read_frame_bytes(path)))


def _open_image(path):
    return Image.open(io.BytesIO(_read_frame_bytes(path)))


@functools.lru_cache(maxsize=48)
def _route_measurements(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _load_measurement_window(measurements, n_meas: int) -> list:
    first = os.fsdecode(measurements[0])
    route_dir = os.path.dirname(os.path.dirname(first))
    if _route_pack(route_dir) is not None:
        return [_load_gz_json(measurements[i]) for i in range(n_meas)]
    consolidated = os.path.join(route_dir, _MEAS_ALL_NAME)
    if os.path.isfile(consolidated):
        table = _route_measurements(consolidated)
        stems = [os.path.basename(os.fsdecode(m)).split(".")[0] for m in measurements[:n_meas]]
        if all(stem in table for stem in stems):
            return [table[stem] for stem in stems]
    return [_load_gz_json(measurements[i]) for i in range(n_meas)]


class PlanTDataset(Dataset):
    @beartype
    def __init__(self, root: str, cfg, shared_dict=None) -> None:
        self.cfg = cfg
        self.cfg_train = cfg.model.training

        self.plant_vars = PlanTVariables

        self.data_cache = shared_dict

        # Frames are dumped at the MetaDrive decision rate (0.1 s), while the
        # pretrained PlanT and both controllers assume 0.25 s between waypoints
        # (the *4.0 in PlanT_agent / plant2_control is 1/dt). Sampling every
        # `wps_stride`-th frame restores that horizon without a re-dump.
        # 1 = old behaviour (0.8 s of waypoints), 2 = 1.6 s, 3 = 2.4 s.
        self.wps_stride = int(os.environ.get(
            "WPS_STRIDE", self.cfg_train.get("wps_stride", 1)) or 1)
        if self.wps_stride > 1:
            print(f"[PlanTDataset] waypoint frame stride = {self.wps_stride} "
                  f"({0.1 * self.wps_stride:.1f} s between waypoints)")

        # Our dumps label target_speed with the ego's *current* speed, and the
        # model receives no ego-speed input — the label is unobservable, so CE
        # is minimised by the marginal speed distribution and the head never
        # predicts stopping. Upstream CARLA collection labels frames with the
        # *commanded* speed instead (0 from the moment the stop is decided,
        # while still moving). TS_LOOKAHEAD=1 approximates that from what is
        # already on disk: the label becomes the minimum ego speed over the
        # loaded future window (seq_len..seq_len+wps_len*stride frames), which
        # is 0 throughout the approach to a stop — observable from the sign
        # geometry and traffic in the frame.
        self.ts_lookahead = str(os.environ.get(
            "TS_LOOKAHEAD", self.cfg_train.get("ts_lookahead", 0)) or 0) not in ("0", "", "False", "false")
        # Floor plates (4.6): label the in-zone target speed no lower than the
        # floor plus this margin. Rule experts that pass the oracle filter may
        # ride the floor itself (a 4.6 violation needs 10 sustained steps under
        # it), and a policy imitating that with the usual noise dips below.
        # 0 keeps the expert's speed as it is.
        self.min_speed_margin_kmh = float(os.environ.get("MIN_SPEED_MARGIN_KMH", 0) or 0)
        if self.ts_lookahead:
            print("[PlanTDataset] target_speed = min ego speed over the "
                  f"{self.cfg.model.waypoints.wps_len * self.wps_stride}-frame lookahead window")

        # loss_path's target is interpolate_route(route) while the model's route
        # token is route_original, and the dump writes both from one array
        # (plant2_frames.py:610-611). The head that steers is therefore trained
        # to reproduce an input it already holds, which is satisfied without
        # reading the sign or the obstacle -- and closed-loop the route is the
        # centre of the lane the ego is already in, so a lateral offset decays
        # as soon as it appears. "future" retargets the head to the ego's
        # realised trajectory, making a held offset the thing the loss rewards.
        self.path_target = str(os.environ.get(
            "PATH_TARGET", self.cfg_train.get("path_target", "route")) or "route").lower()
        if self.path_target not in ("route", "future"):
            raise ValueError(
                f"path_target must be 'route' or 'future', got {self.path_target!r}")
        self.path_horizon = int(os.environ.get(
            "PATH_HORIZON_FRAMES", self.cfg_train.get("path_horizon_frames", 40)) or 40)
        # Frames a sample needs after the current one: the waypoints need
        # wps_len*stride, the realised-future path needs enough travel to fill
        # path_len metres at the 1 m spacing interpolate_route resamples to.
        self.future_frames = self.cfg.model.waypoints.wps_len * self.wps_stride
        if self.path_target == "future":
            self.future_frames = max(self.future_frames, self.path_horizon)
            print(f"[PlanTDataset] loss_path target = realised future over "
                  f"{self.path_horizon} frames ({0.1 * self.path_horizon:.1f} s)")

        # How far a PDD sign stays in the input. Must match the dump's
        # PLANT2_SIGN_RADIUS_M: the tighter of the two wins, silently.
        self.sign_radius = float(os.environ.get(
            "PLANT2_SIGN_RADIUS_M", self.cfg_train.get("sign_radius", 120.0)) or 120.0)

        self.MAX_DISTANCE = self.cfg_train.range
        self.MAX_DISTANCE_DOUBLE = 2*self.MAX_DISTANCE

        self.bev_colors = torch.tensor(self.plant_vars.bev_colors)

        root = root.rstrip("/")

        self.aug_rate = 0.5
        if self.cfg_train.augment:
            self.transform = self.aug_sample
        else:
            self.transform = None

        if self.cfg_train.augment_parked:
            self.parked_locations = {}
            self.parked_rotations = {}
            self.parked_extents = {}
            self.parked_trees = {}
            parked_cars = np.load("/home/geiger/gwb301/code/PlanT_2_cleanup/PlanT/car_data.npy", allow_pickle=True).item()
            for town, data in parked_cars.items():
                self.parked_locations[town] = data["locations"]
                self.parked_rotations[town] = data["rotations"]
                self.parked_extents[town] = data["extents"]
                self.parked_trees[town] = cKDTree(self.parked_locations[town])

        self.speed_cats = self.plant_vars.speed_cats

        self.car_types = self.plant_vars.car_types
        self.type_nums = self.plant_vars.class_nums

        if not self.cfg_train.get("input_static_cars", False):
            self.type_nums.pop("static_car")

        self.BEV = []
        self.labels = []
        self.measurements = []

        # Fast route list when split is pre-filtered (avoids recursive NFS glob).
        if not self.cfg_train.get("filter_routes", True):
            root_path = Path(root)
            label_raw_path = [
                str(p)
                for p in root_path.iterdir()
                if p.is_dir() and (p / "boxes").is_dir()
            ]
        else:
            # If you're not using a slurm cluster you can use this line instead of the one after
            label_raw_path_all = glob.glob(os.path.join(root, "**/boxes"), recursive=True)
            # NOTE: FOR SLURM CHANGE TO:
            # label_raw_path_all = subprocess.run(["lfs", "find", root, "-type", "d", "-name", "boxes", "--maxdepth", "3"], capture_output=True, text=True, check=True).stdout.splitlines()
            label_raw_path = [p[:-5] for p in label_raw_path_all]  # strip "/boxes"

        logging.info(f"Found {len(label_raw_path)} route dirs.")

        total_routes = 0
        skipped_routes = 0
        trainable_routes = 0

        for route_dir in label_raw_path:

            route = os.path.basename(route_dir)
            total_routes += 1

            if self.cfg_train.get("filter_routes", True): # Can be set to false for visu
                if route.startswith('FAILED_') or not os.path.isfile(route_dir + '/results.json.gz'): # or route_dir in manually_kicked:
                    skipped_routes += 1
                    continue

                # We skip data where the expert did not achieve perfect driving score (except for min speed infractions)
                with gzip.open(route_dir + '/results.json.gz', 'rt', encoding='utf-8') as f:
                    results_route = json.load(f)
                condition1 = (results_route['scores']['score_composed'] < 100.0 and \
                not (results_route['num_infractions'] == len(results_route['infractions']['min_speed_infractions'])))
                condition2 = results_route['status'] == 'Failed - Agent couldn\'t be set up'
                condition3 = results_route['status'] == 'Failed'
                condition4 = results_route['status'] == 'Failed - Simulation crashed'
                condition5 = results_route['status'] == 'Failed - Agent crashed'
                if condition1 or condition2 or condition3 or condition4 or condition5:
                    continue

                # Hacky
                if results_route["timestamp"][:4] == "Town":
                    log_file = "qsub_out" + "_".join(results_route["timestamp"].split("_")[:3]) + ".log"
                else:
                    log_file = "qsub_out" + "_".join(results_route["timestamp"].split("_")[:2]) + ".log"

                log_file = root.rstrip("/")[:-4]+"/slurm/run_files/logs/"+log_file

                silentcrash = False
                with open(log_file, "r", encoding="utf8") as f:
                    lines = f.readlines()
                for line in lines:
                    if "SKIPPED" in line:
                        vehicle = line.split(" ")[-1].strip()
                        
                        if vehicle[:6] != "walker" and vehicle not in ["vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets"]:
                            silentcrash = True
                            print(line)
                            break
                
                if silentcrash:
                    continue

            trainable_routes += 1

            route_dir = Path(route_dir)
            num_seq = len(os.listdir(route_dir / "boxes"))

            # ignore the first 5 and last two frames
            for seq in range(
                5,
                num_seq - self.future_frames - self.cfg_train.seq_len - 2,
            ):
                # Only the first frame of the window is stored; the rest are
                # consecutive frame numbers and are expanded in __getitem__.
                # Storing all seq_len + future_frames paths per sample (41 with
                # the future path target) made the dataset object gigabytes,
                # and every spawned loader worker received a pickled copy of
                # it through a pipe: 16 workers took over ten minutes to come
                # up, one after the other, before the first batch.
                self.BEV.append(route_dir / "bev_no_car_semantics" / f"{seq + self.cfg_train.seq_len-1:04d}.png")
                self.labels.append(route_dir / "boxes" / f"{seq:04d}.json.gz")
                self.measurements.append(route_dir / "measurements" / f"{seq:04d}.json.gz")

        # There is a complex "memory leak"/performance issue when using Python objects like lists in a Dataloader that is loaded with multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects because they only have 1 refcount.
        self.BEV          = np.array(self.BEV         ).astype(np.bytes_)
        self.labels       = np.array(self.labels      ).astype(np.bytes_)
        self.measurements = np.array(self.measurements).astype(np.bytes_)
        self.window = self.cfg_train.seq_len + self.future_frames

        # Route → PDD sign_id (embedding index). Resolved once at init; attached
        # on every __getitem__ without writing into diskcache.
        split_meta = Path(root).resolve().parent.parent / "split_meta.json"
        if not split_meta.is_file():
            # root is .../train/data → parent.parent is split root
            split_meta = Path(root).resolve().parent / "split_meta.json"
        extra_map = load_split_meta_route2sign(split_meta)
        uid2sign = load_uid2sign()
        # merge uid map into extra for resolve_sign_id_for_route
        for uid, sign in uid2sign.items():
            extra_map.setdefault(uid, sign)
            for var in ("default", "s1", "s2", "s3", "s4"):
                extra_map.setdefault(f"{uid}_{var}", sign)

        sign_ids = []
        n_valued = 0
        n_known = 0
        n_sniffed = 0
        route_cache: dict = {}

        def _resolve_route(route_name: str, route_dir: str):
            """(sign_id, how) for one route, with the plate value applied.

            The plate number is in neither the route name nor split_meta, which
            stores the bare code. Resolving by name first therefore gave train
            and val two different tokens for the same sign: a train route, absent
            from split_meta, fell through to the boxes and got "3.24@40", while
            the same sign in val resolved from split_meta as plain "3.24". The
            head then fitted the train tokens and met unseen ones at every
            validation step -- train CE at the two-hot floor, val CE at chance,
            and nothing in the logs naming the cause. The boxes are the only
            source that carries the number, so they decide whenever the code can
            carry one.
            """
            code_in_boxes = sniff_sign_from_route(route_dir)
            if code_in_boxes in SIGN_VALUE_CODES:
                val = sniff_sign_value_from_route(route_dir)
                if val is not None:
                    return sign_code_to_id(code_in_boxes, val), "valued"
            sid = resolve_sign_id_for_route(route_name, extra_map)
            if sid > 0:
                return sid, "name"
            if code_in_boxes:
                sid = sign_code_to_id(code_in_boxes,
                                      sniff_sign_value_from_route(route_dir))
                if sid > 0:
                    return sid, "boxes"
            return 0, "none"

        for lab in self.labels:
            label_path = lab.decode()
            route_name = route_name_from_label_path(label_path)
            route_dir = str(Path(label_path).parent.parent)
            if route_dir not in route_cache:
                route_cache[route_dir] = _resolve_route(route_name, route_dir)
            sid, how = route_cache[route_dir]
            if how == "valued":
                n_valued += 1
            elif how == "boxes":
                n_sniffed += 1
            if sid > 0:
                n_known += 1
            sign_ids.append(sid)
        self.sample_sign_ids = np.asarray(sign_ids, dtype=np.int64)
        self.sample_weights, self.frame_meta = self._load_frame_meta(root)
        print(
            f"sign_id resolve: {n_known}/{len(sign_ids)} samples mapped "
            f"({n_sniffed} from boxes, {n_valued} with the plate value, "
            f"split_meta={split_meta.is_file()})"
        )

        print(f"Loading {len(self.labels)} samples")
        print('Total amount of routes:', total_routes)
        print('Skipped routes:', skipped_routes)
        print('Trainable routes:', trainable_routes)

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.measurements)

    def _load_frame_meta(self, root):
        """Per-sample sampling weight + the metadata the sign metrics need.

        Written by scripts/plant2_ft_pipeline/data/make_sample_weights.py. When
        the file is absent every frame weighs 1 and the metrics see an empty
        denominator -- exactly the behaviour before this existed.
        """
        n = len(self.labels)
        weights = np.ones(n, dtype=np.float32)
        meta = {
            # Carried into the batch so the metrics can see which frames the
            # sampler favoured without reaching back into the dataset.
            "frame_weight": weights,
            "in_zone": np.zeros(n, dtype=np.float32),
            "plate_kmh": np.zeros(n, dtype=np.float32),
            "detour_side": np.zeros(n, dtype=np.float32),
            "cones_ahead": np.zeros(n, dtype=np.float32),
        }
        path = Path(root).parent / "sample_weights.json"
        if not path.is_file():
            print(f"sample weights: {path} absent -- uniform sampling, sign metrics idle")
            return weights, meta

        info = json.loads(path.read_text())
        # 4.2.1 passes the obstacle on the right, 4.2.2 on the left
        # (traffic_signs/detour_sign.py). Sign of the lateral offset, not a
        # class id, so the metric can compare it against a waypoint directly.
        side_num = {"right": 1.0, "left": -1.0}
        sets = {name: (set(e.get("in_zone") or ()), set(e.get("cones") or ()))
                for name, e in info.items()}

        n_hot = n_zone = n_cones = 0
        for i, lab in enumerate(self.labels):
            label_path = Path(lab.decode())
            route = label_path.parent.parent.name
            entry = info.get(route)
            if entry is None:
                continue
            seq = int(label_path.name.split(".")[0])
            in_zone, cones = sets[route]

            transient = entry.get("transient")
            if transient and transient[0] <= seq <= transient[1]:
                weights[i] = float(entry.get("w", 1.0))
                n_hot += 1
            if seq in in_zone:
                meta["in_zone"][i] = 1.0
                n_zone += 1
            if seq in cones:
                meta["cones_ahead"][i] = 1.0
                n_cones += 1
            plate = entry.get("plate")
            if plate:
                meta["plate_kmh"][i] = float(plate)
            meta["detour_side"][i] = side_num.get(entry.get("side"), 0.0)

        share = 100.0 * n_hot / max(1, n)
        print(f"sample weights: {n_hot}/{n} frames upweighted ({share:.1f}% of the split), "
              f"in_zone={n_zone} cones_ahead={n_cones}")
        return weights, meta

    def _attach_sign_id(self, sample, index: int):
        """Shallow-copy and set sign_id without mutating diskcache entries."""
        out = dict(sample)
        out["sign_id"] = int(self.sample_sign_ids[index])
        for key, arr in self.frame_meta.items():
            out[key] = float(arr[index])
        return out


    def _window_paths(self, first) -> list:
        """The seq_len + future_frames consecutive frame paths starting at ``first``."""
        first = os.fsdecode(first)
        head, name = os.path.split(first)
        seq = int(name.split(".")[0])
        return [os.fsencode(os.path.join(head, f"{seq + i:04d}.json.gz")) for i in range(self.window)]

    def _cache_key(self, labels) -> str:
        """Cache key. The stride changes the waypoints stored in a sample and
        the lookahead mode changes target_speed, so both must be part of the
        key — otherwise a cache filled under one mode silently serves its
        targets to a run using the other."""
        key = labels[0].decode()
        if self.wps_stride != 1:
            key = f"{key}|s{self.wps_stride}"
        if self.ts_lookahead:
            key = f"{key}|la"
        if self.min_speed_margin_kmh > 0.0:
            key = f"{key}|m{self.min_speed_margin_kmh:g}"
        if self.path_target != "route":
            key = f"{key}|pf{self.path_horizon}"
        return key

    def _floor_kmh(self, boxes, index):
        """Plate value of a floor sign the ego has passed (box behind the ego),
        from the box itself or from the route-level qualified sign id; None
        when the ego is not in the zone or the value is unknown."""
        for b in boxes:
            if str(b.get("class")) not in MIN_SPEED_CODES:
                continue
            try:
                behind = float(b.get("position", [0.0])[0]) < 0.0
            except (TypeError, ValueError, IndexError):
                behind = False
            if not behind:
                continue
            val = b.get("sign_value_kmh")
            if val is None:
                sid = int(self.sample_sign_ids[index])
                if 0 < sid <= len(SIGN_ID_VOCAB) and "@" in SIGN_ID_VOCAB[sid - 1]:
                    val = SIGN_ID_VOCAB[sid - 1].split("@", 1)[1]
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
        return None

    def _future_path(self, loaded_measurements):
        """The ego's realised future in the current ego frame, arc-length
        resampled the same way the route target is, so pred_path keeps its
        geometric meaning while no longer being a copy of its own input token.

        build_ego_matrix writes LOCAL as (x-forward, y-left), the convention the
        dump flips the route into, so the two targets are directly comparable.
        When the expert travels less than the path length -- slow frames, the
        end of an episode -- the trajectory is extended along its final heading
        rather than clamped, which would otherwise collapse the tail of the
        target onto one point and teach the head to plan a stop.
        """
        i0 = self.cfg_train.seq_len - 1
        mats = np.asarray([m["ego_matrix"] for m in loaded_measurements[i0:]],
                          dtype=np.float64)
        pts = (np.linalg.inv(mats[0]) @ mats[:, :, 3].T).T[:, :2]
        path_len = int(self.cfg.model.waypoints.path_len)
        travelled = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        if travelled < path_len + 1.0 and len(pts) > 2:
            step = pts[-1] - pts[-2]
            norm = float(np.linalg.norm(step))
            if norm > 1e-6:
                pts = np.vstack(
                    [pts, pts[-1] + step / norm * (path_len + 1.0 - travelled)])
        return interpolate_route(pts[1:])

    def add_parked_cars(self, sample):
        if self.cfg_train.augment_parked:
            n = np.random.randint(0, 10)
            if len(sample["parked_cars"]) > n:
                if n > 0:
                    idx = np.random.choice(sample["parked_cars"].shape[0], n, replace=False)  
                    sample["input"] += sample["parked_cars"][idx].tolist()
                    sample["output_floating"] += sample["parked_cars"][idx][:, 1:5].tolist()
                    sample["output"] += sample["parked_cars_quant"][idx].tolist()
            elif len(sample["parked_cars"]) > 0:
                sample["input"] += sample["parked_cars"].tolist()
                sample["output_floating"] += sample["parked_cars"][:, 1:5].tolist()
                sample["output"] += sample["parked_cars_quant"].tolist()
            del sample["parked_cars"]
            del sample["parked_cars_quant"]

    def __getitem__(self, index):
        """Returns the item at index idx."""

        labels = self._window_paths(self.labels[index])
        measurements = self._window_paths(self.measurements[index])

        sample = {
            "input": []
        }

        augment = self.transform is not None and np.random.rand() < self.aug_rate

        # See if we can use the cache
        if augment and self.data_cache is not None:
            if self._cache_key(labels) + "_aug" in self.data_cache:
                sample = self.data_cache[self._cache_key(labels) + "_aug"]
                return self._attach_sign_id(sample, index)

            elif self._cache_key(labels) in self.data_cache:
                sample = self.transform(self.data_cache[self._cache_key(labels)])
                sample.pop("BEV_aug", None)
                sample.pop("output_floating", None)
                self.data_cache[self._cache_key(labels) + "_aug"] = sample
                return self._attach_sign_id(sample, index)

        elif self.data_cache is not None and self._cache_key(labels) in self.data_cache:
                sample = self.data_cache[self._cache_key(labels)]
                sample.pop("BEV_aug", None)
                sample.pop("output_floating", None)
                return self._attach_sign_id(sample, index)

        # Load new sample
        loaded_labels = []
        loaded_measurements = []

        # Measurements are needed across the whole future window: the stride
        # slice would otherwise yield wps_len/stride waypoints and crash the L1
        # loss on shape mismatch, and the realised-future path needs the frames
        # beyond that. Boxes are read only at the current frame and at the
        # forecasting offset (+1), so loading the rest is pure I/O.
        n_meas = min(len(measurements), self.cfg_train.seq_len + self.future_frames)
        n_lab = min(len(labels), self.cfg_train.seq_len + 1)
        loaded_measurements = _load_measurement_window(measurements, n_meas)
        for i in range(n_lab):
            loaded_labels.append(_load_gz_json(labels[i]))

        # Extract ego waypoints
        matrices = [x["ego_matrix"] for x in loaded_measurements[self.cfg_train.seq_len - 1 :]]
        ego_inv = np.linalg.inv(matrices[0])
        wps_len = self.cfg.model.waypoints.wps_len
        points = np.array(matrices[self.wps_stride::self.wps_stride][:wps_len])[:, :, 3]
        points = (ego_inv @ points.T).T[:,:2].tolist()
        sample["waypoints"] = points

        sample["route_original"] = loaded_measurements[self.cfg_train.seq_len - 1]["route_original"][:20]
        if self.path_target == "future":
            sample["route"] = self._future_path(loaded_measurements)
        else:
            sample["route"] = interpolate_route(
                loaded_measurements[self.cfg_train.seq_len - 1]["route"][:20])

        meas_t = loaded_measurements[self.cfg_train.seq_len - 1]
        sample["target_speed"] = meas_t["target_speed"]
        # Prefer explicit dump field; fall back to legacy ``speed``.
        sample["ego_speed"] = meas_t["ego_speed"] if "ego_speed" in meas_t else meas_t["speed"]

        if self.ts_lookahead:
            # Commanded-style label: the minimum ego speed over the loaded
            # future window. A frame 1-2 s before a stop is labelled 0 while
            # the car still moves — matching what upstream autopilot dumps —
            # and, unlike the instantaneous speed, it is predictable from the
            # sign geometry and traffic visible in the frame. Speeds under the
            # dump's brake epsilon collapse to an exact 0.0 so the two-hot
            # encoding puts full mass on bin 0. Only meaningful for routes
            # whose label is the ego speed (priority signs); do not enable for
            # posted-limit routes, where target_speed is the constant limit.
            future = loaded_measurements[self.cfg_train.seq_len - 1:]
            speeds = [float(m["ego_speed"] if "ego_speed" in m else m["speed"])
                      for m in future]
            # The minimum encodes a margin BELOW what the expert drove, which is
            # what a ceiling plate (3.24 / 5.31 / 5.21) needs: labelling the
            # posted number itself made the model sit on the limit and violate
            # about half the in-zone steps. A floor plate (4.6) demands the
            # mirror image -- taking the minimum there points the margin at the
            # violation, and its compliance stayed at 0.231 while the ceiling
            # signs reached 0.95-1.00 on the same run.
            code = base_sign_code(int(self.sample_sign_ids[index]))
            # The window maximum was the obvious mirror, but it raises the whole
            # speed head: with it 4.6 went 0.231 -> 0.346 while the three ceiling
            # plates fell 0.96/0.96/0.99 -> 0.85/0.72/0.89 on the same run, for a
            # worse total. The expert's own speed already satisfies the floor
            # (measured: 0 violating steps in 4405 in-zone frames under the sign's
            # real rule), so imitating it needs no margin in either direction.
            # A detour is driven around cones with continual small brakings,
            # so the window minimum sits well under the expert's speed on every
            # frame: models trained with it crawled the detours at 12-14 km/h
            # against 20-23 km/h for the experts they imitated (v6 test), while
            # the plate-label ablation, whose detour label is the plain speed,
            # drove them at 40 km/h. The side of the manoeuvre is in the path
            # target, not in the speed label, so detours take the plain speed.
            v = speeds[0] if (code in MIN_SPEED_CODES or code in DETOUR_CODES) else min(speeds)
            if code in MIN_SPEED_CODES and self.min_speed_margin_kmh > 0.0:
                floor = self._floor_kmh(loaded_labels[self.cfg_train.seq_len - 1], index)
                if floor is not None:
                    v = max(v, (floor + self.min_speed_margin_kmh) / 3.6)
            sample["target_speed"] = 0.0 if v < 0.5 else v

        speed_limit = loaded_measurements[self.cfg_train.seq_len - 1]["speed_limit"]
        speed_limit = round(speed_limit * 3.6)  # TODO
        # Map unknown PDD limits (20/40/60/...) to nearest known embedding bin.
        if speed_limit not in self.speed_cats:
            known = sorted(self.speed_cats.keys())
            speed_limit = min(known, key=lambda k: abs(k - speed_limit))
        sample["speed_limit"] = self.speed_cats[speed_limit]

        if loaded_measurements[self.cfg_train.seq_len - 1]["brake"]: # Just in case
            sample["target_speed"] = 0.0

        if self.cfg_train.get("input_bev", False):
            bev = _open_image(self.BEV[index].decode())
            bev = pil_to_tensor(bev)
            bev = torch.rot90(bev, dims=(1, 2))
            sample["BEV"] = self.bev_colors[bev[0, 64:-64, 64:-64].to(torch.int)].permute(2, 0, 1)

            if self.cfg_train.augment:
                aug_path = self.BEV[index].decode().replace("bev_no_car_semantics", "bev_no_car_semantics_augmented")
                bev_aug = Image.open(aug_path)
                bev_aug = pil_to_tensor(bev_aug)
                bev_aug = torch.rot90(bev_aug, dims=(1, 2))
                sample["BEV_aug"] = self.bev_colors[bev_aug[0, 64:-64, 64:-64].to(torch.int)].permute(2, 0, 1)

        sample["augmentation_translation"] = loaded_measurements[self.cfg_train.seq_len - 1]["augmentation_translation"]
        sample["augmentation_rotation"] = loaded_measurements[self.cfg_train.seq_len - 1]["augmentation_rotation"]

        # Load Input objects
        measurements_data = loaded_measurements[self.cfg_train.seq_len - 1]
        labels_data_all = loaded_labels[self.cfg_train.seq_len - 1]

        labels_data = labels_data_all[1:] # remove ego car

        ego_matrix = np.array(measurements_data["ego_matrix"])
        ego_yaw = measurements_data["theta"]

        # Only for viz
        sample["ego_pos"] = measurements_data["pos_global"]
        sample["ego_rot"] = ego_yaw

        # Spatial PDD signs dumped as boxes["class"] == "2.1" / "3.24" / ...
        pdd_classes = set(SIGN_CODES)
        sign_like = {"stop_sign"} | pdd_classes

        # Fix static extents and drop irrelevant objects
        for x in labels_data:
            if "position" in x:
                pos_x, pos_y, pos_z = x["position"]

                # A traffic light or a CARLA stop sign matters within 30 m —
                # its rule bites at a point. A PDD sign governs a zone that runs
                # 103 m at the median, and with seq_len=1 there is no memory to
                # carry it, so the sign itself has to stay in frame: the dump
                # writes it out to PLANT2_SIGN_RADIUS_M and this must not throw
                # it away again. The radius was hardcoded in both places, so
                # widening it in the dump alone changed nothing.
                if x["class"] in pdd_classes:
                    if pos_x**2 + pos_y**2 > self.sign_radius**2 or abs(pos_z) > 30:
                        x["class"] = "too far"
                elif x["class"] in (["traffic_light", "stop_sign"]):
                    if pos_x**2 + pos_y**2 > 30**2 or abs(pos_z) > 30:
                        x["class"] = "too far"
                # ellipse for others
                else:
                    x_div = self.cfg_train.range_factor_front**2 if pos_x > 0 else 1
                    if pos_x**2/x_div + pos_y**2 > self.MAX_DISTANCE**2 or abs(pos_z) > 30:
                        x["class"] = "too far"

            # Emergency vehicles
            if x["class"]=="car" and x["type_id"] in ["vehicle.dodge.charger_police",
                                                        "vehicle.dodge.charger_police_2020",
                                                        "vehicle.carlamotors.firetruck",
                                                        "vehicle.ford.ambulance"]:
                x["class"] = "emergency"

            # Filter statics and fix extents
            elif x["class"]=="static":
                if "type_id" in x.keys() and x["type_id"] not in ["static.prop.constructioncone", 
                                                                    "static.prop.trafficwarning"]:
                    x["class"] = "irrelevant_static"
                else:
                    # # update static extent
                    if x["type_id"] in STATIC_EXTENTS:
                        x["extent"] = STATIC_EXTENTS[x["type_id"]]
                    else:
                        print(x["type_id"], "was not found in static extents")

            elif x["class"] == "static_car":
                if x["mesh_path"] in CAR_EXTENTS:
                    x["extent"] = CAR_EXTENTS[x["mesh_path"]]
                    if "scale" in x.keys() and x["scale"] is not None:
                        scale = float(x["scale"])
                        x["extent"] = [a*scale for a in x["extent"]]
                else:
                    print("missing static car:", x["mesh_path"])

        input_objects = [
                [
                    self.type_nums[x["class"].lower()],  # type indicator
                    x["position"][0],
                    x["position"][1],
                    rad2deg(x["yaw"]),  # in degrees
                    x["speed"] * 3.6,  # in km/h
                    x["extent"][1]*2 + (0 if "scenario" not in x.keys() or "Door" not in x["scenario"] else 1),
                    x["extent"][0]*2,
                    x["id"],
                ]
                for x in labels_data
                if x["class"].lower() in self.car_types
            ]

        # Add static cars, static objects, traffic lights, stop / PDD signs
        def _keep_staticish(x) -> bool:
            cls = x["class"].lower() if isinstance(x["class"], str) else x["class"]
            # PDD codes are numeric strings ("2.1"); keep original key for type_nums.
            cls_key = x["class"] if x["class"] in self.type_nums else cls
            if cls_key in self.car_types:
                return False
            if cls_key not in self.type_nums:
                return False
            if cls == "traffic_light":
                return x.get("state") in ["Red", "Yellow"] and x.get("affects_ego")
            if cls_key in sign_like or cls in sign_like:
                return bool(x.get("affects_ego"))
            return True

        input_objects += [[
                self.type_nums[x["class"] if x["class"] in self.type_nums else x["class"].lower()],
                x["position"][0],
                x["position"][1],
                rad2deg(x["yaw"]),  # in degrees
                # Static objects do not move, so this slot was a constant 0.
                # Speed-limit plates (3.24 / 5.31 / 4.6) now carry the number
                # written on them here, in km/h: it is the only channel that
                # tells 20 from 60 apart, since both share one PDD class.
                # Everything else still dumps speed 0 and is unaffected.
                x.get("speed", 0.0) * 3.6,
                x["extent"][1]*2,
                x["extent"][0]*2,
                -1 if (x["class"] if x["class"] in self.type_nums else x["class"].lower()) != "static_car" else -999,
            ]
            for x in labels_data
            if _keep_staticish(x)
        ]

        # Load output (forecasting) objects
        offset = 1
        measurements_data_out = loaded_measurements[self.cfg_train.seq_len - 1 + offset]
        labels_data_all_out = loaded_labels[self.cfg_train.seq_len - 1 + offset]

        output_input_tf =  np.linalg.inv(ego_matrix) @ measurements_data_out["ego_matrix"]
        output_input_yaw = ego_yaw - measurements_data_out["theta"]

        # Generate transformed output objects for forecasting
        output_cars = {x["id"]: [
                                *(output_input_tf @ np.append(x["position"],[1]))[:2],
                                rad2deg(x["yaw"] - output_input_yaw),  # in degrees
                                x["speed"] * 3.6  # in km/h
                                ] for x in labels_data_all_out if "id" in x and "speed" in x}

        output_objects_matched = []
        for x in input_objects:
            object_id = x[-1]
            if object_id == -999: # static_car
                output_objects_matched.append(x[1:5])
            elif object_id in output_cars:
                output_objects_matched.append(output_cars[object_id])
            else:
                output_objects_matched.append([-999., -999., -999., -999.]) # Dummy for traffic lights, statics, etc.

        output_objects_quantized = self.quantize_box(output_objects_matched)

        sample["output_floating"] = output_objects_matched
        sample["output"] = output_objects_quantized

        # remove id 
        input_objects = [x[:-1] for x in input_objects]

        sample["input"] = input_objects

        if self.cfg_train.augment_parked:
            town = labels[0].decode().split("/")[-3].split("_")[0]
            if town == "Town10":
                town = "Town10HD"
            ego_pos = loaded_measurements[self.cfg_train.seq_len - 1]["pos_global"]
            ego_theta = loaded_measurements[self.cfg_train.seq_len - 1]["theta"]

            idxs = self.parked_trees[town].query_ball_point(ego_pos, 30)
            if len(idxs) > 0:
                sample["parked_cars"] = np.array([[self.type_nums["car"], x, y, yaw, 0, y_ex*2, x_ex*2]
                                        for ((x, y), yaw, (x_ex, y_ex, _)) in zip(self.parked_locations[town][idxs], self.parked_rotations[town][idxs], self.parked_extents[town][idxs])])
                sample["parked_cars"][:, 1:3] -= ego_pos
                c, s = np.cos(ego_theta), np.sin(ego_theta)
                R = np.array([[c, -s], [s, c]])
                sample["parked_cars"][:, 1:3] = (R.T @ sample["parked_cars"][:, 1:3].T).T
                sample["parked_cars"][:, 3] = normalize_angle_degree(sample["parked_cars"][:, 3] - np.rad2deg(ego_theta))
                sample["parked_cars_quant"] = np.array(self.quantize_box(sample["parked_cars"][:, 1:5]))
            else:
                sample["parked_cars"] = np.array([])
                sample["parked_cars_quant"] = np.array([])

            # For now i fix the parked augmentation per sample with and without aug, could be unique per call but has performance implications
            self.add_parked_cars(sample)

        # Store in data cache
        if self.data_cache is not None:
            self.data_cache[self._cache_key(labels)] = sample # Save unaugmented sample with BEV_aug so we can use it later for aug

        if augment:
            sample = self.transform(sample)
            if self.data_cache is not None:
                sample.pop("BEV_aug", None) # Augmented sample doesnt need BEV_aug since its the normal BEV
                sample.pop("output_floating", None)
                self.data_cache[self._cache_key(labels) + "_aug"] = sample

        sample.pop("BEV_aug", None)
        sample.pop("output_floating", None)

        return self._attach_sign_id(sample, index)

    def aug_sample(self, sample):
        # Geometric augment using recorded augmentation_translation / rotation.
        # Translation applied first (transfuser convention).
        translate = - np.array([0.0, sample["augmentation_translation"]])
        rot = np.deg2rad(sample["augmentation_rotation"])

        if self.cfg_train.get("input_bev", False):
            sample["BEV"] = sample["BEV_aug"]

        input = np.array(sample["input"])
        if "output" in sample:
            output = np.array(sample["output_floating"])
            dummy_mask = output == -999.
        else:
            output = []
        waypoints = np.array(sample["waypoints"])
        route = np.array(sample["route"])
        route_original = np.array(sample["route_original"])

        # Translation
        if len(input) > 0:
            input[:, 1:3] += translate
        if len(output) > 0:
            output[:, :2] += translate
        waypoints += translate
        route += translate
        route_original += translate

        # Rotation
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])

        if len(input) > 0:
            input[:, 1:3] = (R.T @ input[:, 1:3].T).T
        if len(output) > 0:
            output[:, :2] = (R.T @ output[:, :2].T).T
        waypoints = (R.T @ waypoints.T).T
        route = (R.T @ route.T).T
        route_original = (R.T @ route_original.T).T

        if len(input) > 0:
            input[:, 3] -= np.rad2deg(rot)
        if len(output) > 0:
            output[:, 2] -= np.rad2deg(rot)

        sample["input"] = input.tolist()
        if "output" in sample:
            output[dummy_mask] = -999.
            sample["output"] = self.quantize_box(output.tolist())
        sample["waypoints"] = waypoints.tolist()
        sample["route"] = route.tolist()
        sample["route_original"] = route_original.tolist()

        return sample

    def quantize_box(self, boxes):
        boxes = np.array(boxes)

        if len(boxes)==0:
            return boxes.tolist()

        # range of xy is [-30, 30]
        # range of yaw is [-360, 0]
        # range of speed is [0, 120]
        # range of extent is [0, 30]

        # Dummy mask:
        dummy_mask = boxes==-999

        # quantize xy
        boxes[:, 0] = (boxes[:, 0] + self.MAX_DISTANCE) / self.MAX_DISTANCE_DOUBLE
        boxes[:, 1] = (boxes[:, 1] + self.MAX_DISTANCE) / self.MAX_DISTANCE_DOUBLE

        # quantize yaw
        boxes[:, 2] = (boxes[:, 2] % 360) / 360

        # quantize speed
        boxes[:, 3] = boxes[:, 3] / 120

        boxes[:, 0] = np.clip(boxes[:, 0], 0, (1 + self.cfg.model.training.get("range_factor_front", 1)) / 2)
        boxes[:, 1:] = np.clip(boxes[:, 1:], 0, 1)

        size_pos = pow(2, self.cfg.model.pre_training.precision_pos)
        size_speed = pow(2, self.cfg.model.pre_training.precision_speed)
        size_angle = pow(2, self.cfg.model.pre_training.precision_angle)

        boxes[:, :2] = (boxes[:, :2] * (size_pos - 1)).round()
        boxes[:, 2] = (boxes[:, 2] * (size_angle - 1)).round()
        boxes[:, 3] = (boxes[:, 3] * (size_speed - 1)).round()

        boxes[dummy_mask] = -999

        return boxes.astype(np.int32).tolist()
    
    # This is only used for visualization
    def unquantize_box(self, boxes):
        boxes = np.array(boxes).astype(np.float32)

        if len(boxes)==0:
            return boxes.tolist()
        
        size_pos = pow(2, self.cfg.model.pre_training.precision_pos)
        size_speed = pow(2, self.cfg.model.pre_training.precision_speed)
        size_angle = pow(2, self.cfg.model.pre_training.precision_angle)

        boxes[:, :2] = boxes[:, :2] / (size_pos - 1)
        boxes[:, 2] = boxes[:, 2] / (size_angle - 1)
        boxes[:, 3] = boxes[:, 3] / (size_speed - 1)

        # unquantize xy
        boxes[:, 0] = boxes[:, 0] * self.MAX_DISTANCE_DOUBLE - self.MAX_DISTANCE
        boxes[:, 1] = boxes[:, 1] * self.MAX_DISTANCE_DOUBLE - self.MAX_DISTANCE

        # unquantize yaw
        boxes[:, 2] = normalize_angle_degree(boxes[:, 2] * 360)

        # unquantize speed
        boxes[:, 3] = boxes[:, 3] * 120 #TODO

        return boxes.tolist()


    # def filter_data_by_town(self, label_raw_path_all, split):
    #     # in case we want to train without T2 and T5
    #     label_raw_path = []
    #     if split == "train":
    #         for path in label_raw_path_all:
    #             if "Town02" in path or "Town05" in path:
    #                 continue
    #             label_raw_path.append(path)
    #     elif split == "val":
    #         for path in label_raw_path_all:
    #             if "Town02" in path or "Town05" in path:
    #                 label_raw_path.append(path)
    #     elif split == "all":
    #         label_raw_path = label_raw_path_all
            
    #     return label_raw_path

def interpolate_route(points):
    route = np.concatenate((np.zeros_like(points[:1]),  points)) # Add 0 to front
    shift = np.roll(route, 1, axis=0) # Shift by 1
    shift[0] = shift[1] # Set wraparound value to 0

    dists = np.linalg.norm(route-shift, axis=1)
    dists = np.cumsum(dists)
    dists += np.arange(0, len(dists))*1e-4 # Prevents dists not being strictly increasing

    x = np.arange(0, 20, 1)
    interp_points = np.array([np.interp(x, dists, route[:, 0]), np.interp(x, dists, route[:, 1])]).T

    return interp_points

def generate_batch(data_batch):
    maxseq = max([len(sample["input"]) for sample in data_batch])
    B = len(data_batch)

    x_batch_objs = [[0, 0, 0, 0, 0, 0, 0]]  # Padding at idx 0
    y_batch_objs = [[-999, -999, -999, -999]]  # Padding

    batch_idxs = torch.zeros((B, maxseq), dtype=torch.int32)

    keys = [x for x in data_batch[0] if x not in ["input", "output"]]

    batches = {key: [] for key in keys}

    n = 1  # Padding is 0

    for i, sample in enumerate(data_batch):
        # Input
        n_sample = len(sample["input"])
        batch_idxs[i, :n_sample] = torch.arange(n, n+n_sample)
        n += n_sample

        x_batch_objs.extend(sample["input"])
        y_batch_objs.extend(sample["output"])

        for key in keys:
            if key == "speed_limit" or key == "sign_id":
                batches[key].append(torch.tensor(sample[key], dtype=torch.int))
            else:
                if torch.is_tensor(sample[key]):
                    batches[key].append(sample[key].type(torch.float32))
                else:
                    batches[key].append(torch.tensor(sample[key], dtype=torch.float32))

    batches = {key: torch.stack(value) for key, value in batches.items()}
    batches["idxs"] = batch_idxs
    batches["x_objs"] = torch.tensor(x_batch_objs, dtype=torch.float32)
    batches["y_objs"] = torch.tensor(y_batch_objs, dtype=torch.long)

    return batches


if __name__=="__main__":
    import yaml
    # Read YAML file
    with open("PlanT/config/config.yaml", 'r') as stream:
        cfg = yaml.safe_load(stream)

    with open("PlanT/config/model/PlanT.yaml", 'r') as stream:
        plnt = yaml.safe_load(stream)

    cfg["model"] = plnt

    cfg["visualize"] = False

    cfg["trainset_size"] = 1
    class DictAsMember(dict):
        def __getattr__(self, name):
            value = self[name]
            if isinstance(value, dict):
                value = DictAsMember(value)
            return value

    cfg = DictAsMember(cfg)

    ds = PlanTDataset("/home/simon/PlanT_2_cleanup/PlanT_2_2025_07_24/data", cfg)

    print(generate_batch([ds[255], ds[256], ds[257]]).keys())
