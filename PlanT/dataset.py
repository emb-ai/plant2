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

import gzip
import glob

from plant_variables import PlanTVariables
from util.static_extents import CAR_EXTENTS, STATIC_EXTENTS
from util.sign_id import SIGN_CODES

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

class PlanTDataset(Dataset):
    @beartype
    def __init__(self, root: str, cfg, shared_dict=None) -> None:
        self.cfg = cfg
        self.cfg_train = cfg.model.training

        self.plant_vars = PlanTVariables

        self.data_cache = shared_dict

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
                num_seq - self.cfg.model.waypoints.wps_len - self.cfg_train.seq_len - 2,
            ):
                # load input seq and pred seq jointly
                label = []
                measurement = []
                for idx in range(
                    self.cfg_train.seq_len + self.cfg.model.waypoints.wps_len
                ):
                    labels_file = route_dir / "boxes" / f"{seq + idx:04d}.json.gz"
                    measurements_file = (
                        route_dir / "measurements" / f"{seq + idx:04d}.json.gz"
                    )
                    label.append(labels_file)
                    measurement.append(measurements_file)

                bev_path = route_dir / "bev_no_car_semantics" / f"{seq + self.cfg_train.seq_len-1:04d}.png"
                repeat = self._oversample_repeat(label[self.cfg_train.seq_len - 1])
                for _ in range(repeat):
                    self.BEV.append(bev_path)
                    self.labels.append(label)
                    self.measurements.append(measurement)

        # There is a complex "memory leak"/performance issue when using Python objects like lists in a Dataloader that is loaded with multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects because they only have 1 refcount.
        self.BEV          = np.array(self.BEV         ).astype(np.bytes_)
        self.labels       = np.array(self.labels      ).astype(np.bytes_)
        self.measurements = np.array(self.measurements).astype(np.bytes_)

        print(f"Loading {len(self.labels)} samples")
        print('Total amount of routes:', total_routes)
        print('Skipped routes:', skipped_routes)
        print('Trainable routes:', trainable_routes)

    def _oversample_repeat(self, labels_file) -> int:
        """H5-detour: how many times to duplicate this frame in the dataset.

        Obstacle-avoidance is a handful of frames per route needing a real
        lateral swerve, drowned out by the rest of the route's near-straight
        frames -- loss reweighting alone plateaus under Adam's per-parameter
        gradient normalization (scaling one sample's loss doesn't scale the
        actual step size nearly that much, since the running gradient-
        magnitude estimate for the same parameters adapts alongside it).
        Oversampling changes how often the pattern is seen at all, which
        loss-scaling can't replicate.

        Uses the frame's own ``boxes/*.json.gz`` (ego-frame object list,
        cones are ``class=='static', type_id=='static.prop.constructioncone'``)
        rather than the ground-truth path's lateral extent -- a first
        attempt at the latter found ~59% of ALL frames exceed 0.5m lateral
        deviation (ordinary turns/junctions have similar magnitude to an
        obstacle swerve in this representation), so it can't distinguish
        "near an obstacle" from "route has a bend in it". Cone proximity is
        unambiguous.

        Distances form a WINDOW ``[oversample_cone_distance_min_m,
        oversample_cone_distance_m]``. The window matters: the expert starts
        its lane change a median 51.5 m before the cone (77% of routes finish
        it before the 30 m compliance zone even begins), so the frames that
        actually teach the *decision* are the far ones. An earlier run
        oversampled `<=5 m` -- frames where the maneuver is already over --
        and unsurprisingly changed nothing.

        ``oversample_cone_distance_m`` (default 0 disabled) /
        ``oversample_factor`` (default 1, no duplication even if a distance is
        set) gate this.
        """
        factor = int(self.cfg_train.get("oversample_factor", 1))
        if factor <= 1:
            return 1

        cone_max = float(self.cfg_train.get("oversample_cone_distance_m", 0.0))
        cone_min = float(self.cfg_train.get("oversample_cone_distance_min_m", 0.0))
        # Same window, but measured to the SIGN rather than to a cone. On scene
        # sets where the obstacle is rarely in frame the cone rule is inert --
        # measured on the oracle detour dump, a cone is visible in 4% of frames
        # against 69% on the older one, so cone-keyed oversampling touched 725
        # frames instead of 42050. The sign is present in 87% of those frames,
        # and it is what the decision is actually made on.
        sign_max = float(self.cfg_train.get("oversample_sign_distance_m", 0.0))
        sign_min = float(self.cfg_train.get("oversample_sign_distance_min_m", 0.0))
        if cone_max <= 0.0 and sign_max <= 0.0:
            return 1

        with gzip.open(labels_file, "rt", encoding="utf-8") as f:
            boxes = json.load(f)
        sign_like = set(SIGN_CODES)
        for obj in boxes:
            if not isinstance(obj, dict) or "position" not in obj:
                continue
            x, y = obj["position"][0], obj["position"][1]
            d = (x * x + y * y) ** 0.5
            if cone_max > 0.0 and obj.get("type_id") == "static.prop.constructioncone":
                if cone_min <= d <= cone_max:
                    return factor
            if sign_max > 0.0 and obj.get("class") in sign_like:
                if sign_min <= d <= sign_max:
                    return factor
        return 1

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.measurements)

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

    @staticmethod
    def _sign_in_range(objs, sign_like, sign_range) -> bool:
        for x in objs:
            cls = x.get("class")
            cls_l = cls.lower() if isinstance(cls, str) else cls
            if cls not in sign_like and cls_l not in sign_like:
                continue
            if not x.get("affects_ego") or "position" not in x:
                continue
            pos_x, pos_y, pos_z = x["position"]
            if pos_x**2 + pos_y**2 <= sign_range**2 and abs(pos_z) <= 30:
                return True
        return False

    def _recent_sign_signal(self, cur_boxes_path, cur_labels_data, sign_like, sign_range, lookback) -> float:
        """H3: 1.0 if a relevant sign is in-frame now; else linearly-decayed
        value based on how many frames ago it was last seen (0 beyond lookback)."""
        if self._sign_in_range(cur_labels_data, sign_like, sign_range):
            return 1.0

        boxes_path = Path(cur_boxes_path)
        boxes_dir = boxes_path.parent
        seq = int(boxes_path.name.split(".")[0])
        for back in range(1, lookback + 1):
            prev_seq = seq - back
            if prev_seq < 0:
                break
            prev_path = boxes_dir / f"{prev_seq:04d}.json.gz"
            if not prev_path.is_file():
                break
            prev_boxes = json.load(gzip.open(prev_path))[1:]  # drop ego
            if self._sign_in_range(prev_boxes, sign_like, sign_range):
                return max(0.0, 1.0 - back / lookback)
        return 0.0

    def __getitem__(self, index):
        """Returns the item at index idx."""

        labels = self.labels[index]
        measurements = self.measurements[index]

        sample = {
            "input": []
        }

        augment = self.transform is not None and np.random.rand() < self.aug_rate

        # See if we can use the cache.
        #
        # Uses the atomic Cache.get() rather than `key in cache` followed by
        # `cache[key]`. That check-then-read pattern is a TOCTOU race: when
        # several trainings share one diskcache sitting at its size limit,
        # another process can evict the key between the two calls, and the
        # read then raises KeyError inside a DataLoader worker and kills the
        # run mid-epoch (observed doing exactly that). get() returns None on a
        # miss instead, and we simply fall through and reload from disk.
        # NB: never assign to `sample` here -- it is already initialised to the
        # dict built above, and the load-from-disk path below fills that dict
        # in place. Binding a cache miss (None) to it breaks that fallthrough.
        if self.data_cache is not None:
            cache_key = labels[0].decode()
            if augment:
                cached_aug = self.data_cache.get(cache_key + "_aug")
                if cached_aug is not None:
                    return cached_aug

                cached = self.data_cache.get(cache_key)
                if cached is not None:
                    transformed = self.transform(cached)
                    transformed.pop("BEV_aug", None)
                    transformed.pop("output_floating", None)
                    self.data_cache[cache_key + "_aug"] = transformed
                    return transformed
            else:
                cached = self.data_cache.get(cache_key)
                if cached is not None:
                    cached.pop("BEV_aug", None)
                    cached.pop("output_floating", None)
                    return cached

        # Load new sample
        loaded_labels = []
        loaded_measurements = []

        for i in range(self.cfg_train.seq_len + self.cfg.model.waypoints.wps_len):
            measurements_i = json.load(gzip.open(measurements[i]))
            labels_i = json.load(gzip.open(labels[i]))

            loaded_labels.append(labels_i)
            loaded_measurements.append(measurements_i)

        # Extract ego waypoints
        matrices = [x["ego_matrix"] for x in loaded_measurements[self.cfg_train.seq_len - 1 :]]
        ego_inv = np.linalg.inv(matrices[0])
        points = np.array(matrices[1:])[:,:,3]
        points = (ego_inv @ points.T).T[:,:2].tolist()
        sample["waypoints"] = points

        sample["route_original"] = loaded_measurements[self.cfg_train.seq_len - 1]["route_original"][:20]
        sample["route"] = interpolate_route(loaded_measurements[self.cfg_train.seq_len - 1]["route"][:20])

        meas_t = loaded_measurements[self.cfg_train.seq_len - 1]
        sample["target_speed"] = meas_t["target_speed"]
        # Prefer explicit dump field; fall back to legacy ``speed``.
        sample["ego_speed"] = meas_t["ego_speed"] if "ego_speed" in meas_t else meas_t["speed"]

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
            bev = Image.open(self.BEV[index].decode())
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

        # H1 (sign persistence): radius for TL/sign objects, overridable per
        # experiment (default 30m matches original behaviour).
        sign_range = self.cfg_train.get("sign_range_m", 30)

        # H3 (sign memory): decayed "was a relevant sign seen in the last K
        # frames" scalar, computed from raw (unmutated) labels_data before the
        # in-place "too far" reclassification below. Off by default.
        sign_memory_frames = int(self.cfg_train.get("sign_memory_frames", 0) or 0)
        if sign_memory_frames > 0:
            sample["sign_recent_signal"] = self._recent_sign_signal(
                labels[0].decode(), labels_data, sign_like, sign_range, sign_memory_frames
            )

        # H4 (auxiliary sign-presence loss): ground truth for the presence
        # probe added in model.py when aux_sign_presence_weight > 0.
        if float(self.cfg_train.get("aux_sign_presence_weight", 0.0) or 0.0) > 0:
            sample["sign_present_now"] = float(
                self._sign_in_range(labels_data, sign_like, sign_range)
            )

        # Fix static extents and drop irrelevant objects
        for x in labels_data:
            if "position" in x:
                pos_x, pos_y, pos_z = x["position"]

                # sign_range radius for TL / stop / PDD sign objects
                if x["class"] in (["traffic_light"] + list(sign_like)):
                    if pos_x**2 + pos_y**2 > sign_range**2 or abs(pos_z) > 30:
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

        # H1 (sign persistence): the simulator's "affects_ego" flag can flicker
        # frame-to-frame for sign-class objects even while in range, unlike the
        # unconditional keep used for statics. Gate-able so default behaviour
        # (require affects_ego) is unchanged unless the experiment opts in.
        sign_ignore_affects_ego = self.cfg_train.get("sign_ignore_affects_ego", False)

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
                if sign_ignore_affects_ego:
                    return True
                return bool(x.get("affects_ego"))
            return True

        input_objects += [[
                self.type_nums[x["class"] if x["class"] in self.type_nums else x["class"].lower()],
                x["position"][0],
                x["position"][1],
                rad2deg(x["yaw"]),  # in degrees
                0.0,
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
            self.data_cache[labels[0].decode()] = sample # Save unaugmented sample with BEV_aug so we can use it later for aug

        if augment:
            sample = self.transform(sample)
            if self.data_cache is not None:
                sample.pop("BEV_aug", None) # Augmented sample doesnt need BEV_aug since its the normal BEV
                sample.pop("output_floating", None)
                self.data_cache[labels[0].decode()+"_aug"] = sample

        sample.pop("BEV_aug", None)
        sample.pop("output_floating", None)

        return sample

    def aug_sample(self, sample):
        # Geometric augment using recorded augmentation_translation / rotation.
        # Translation applied first (transfuser convention).
        #
        # The two groups of fields live in MIRRORED lateral frames in this dump:
        # object boxes (`input` / `output`) are y=RIGHT (plant2_frames._ego_xy
        # negates what MetaDrive's convert_to_local_coordinates returns), while
        # route / route_original / waypoints are y=LEFT. Applying one signed
        # shift and one rotation to all of them -- correct upstream, where every
        # field shares a frame -- moves the objects and the route in OPPOSITE
        # physical directions, i.e. it teaches the wrong geometry. Mirroring a
        # frame negates both the lateral shift and the rotation sense, so the
        # y=right group takes +translation and R, the y=left group -translation
        # and R.T.
        translate_left = - np.array([0.0, sample["augmentation_translation"]])
        translate_right = - translate_left
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
            input[:, 1:3] += translate_right
        if len(output) > 0:
            output[:, :2] += translate_right
        waypoints += translate_left
        route += translate_left
        route_original += translate_left

        # Rotation
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])

        if len(input) > 0:
            input[:, 1:3] = (R @ input[:, 1:3].T).T
        if len(output) > 0:
            output[:, :2] = (R @ output[:, :2].T).T
        waypoints = (R.T @ waypoints.T).T
        route = (R.T @ route.T).T
        route_original = (R.T @ route_original.T).T

        if len(input) > 0:
            input[:, 3] += np.rad2deg(rot)
        if len(output) > 0:
            output[:, 2] += np.rad2deg(rot)

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
            if key == "speed_limit":
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
