from util.sign_id import SIGN_CODES, SIGN_CODE_ALIASES

# tok_emb index for the first PDD object class (after emergency=6).
PDD_OBJECT_CLASS_START = 7


class PlanTVariables:
    bev_colors = [[0.485, 0.456, 0.406],# Background: Imagenet mean
                [0.25, 0.25, 0.75], # Street: Blue
                [0.485, 0.456, 0.406],# Sidewalk: Imagenet mean
                [0.75, 0.25, 0.25], # All lines: Red
                [0.25, 0.75, 0.25]] # Broken lines: Green
    
    speed_cats = {50: 0, 80: 1, 100: 2, 120: 3}

    class_nums = {# "ego_car": 1.0,
                    "car": 1.0,
                    "walker": 2.0,
                    "static": 3.0,
                    # "static_trafficwarning": 3.0,
                    "static_car": 1.0,
                    "stop_sign": 4.0,
                    "traffic_light": 5.0,
                    "emergency": 6.0,
                    # One unique tok_emb index per PDD code (boxes["class"] == code).
                    **{
                        code: float(PDD_OBJECT_CLASS_START + i)
                        for i, code in enumerate(SIGN_CODES)
                    },
                 }

    # The 2.3 variants share one box token for the same reason they share one
    # sign id: the side is in the plate's position, not in its class.
    _v23 = class_nums["2.3.1"]
    for _variant in SIGN_CODE_ALIASES:
        class_nums[_variant] = _v23
    class_nums["2.3"] = _v23
    del _v23, _variant

    # Classes written into boxes that are spatial PDD signs (not the global sign_emb).
    pdd_object_classes = frozenset(SIGN_CODES) | {"2.3"}
    
    car_types = ["car", "walker","emergency"]

    target_speeds = [0.0, 4.0, 8.0, 10, 13.88888888, 16, 17.77777777, 20]

    @staticmethod
    def num_object_types() -> int:
        """Size of tok_emb ModuleList (= max class index + 1, including padding 0)."""
        return int(max(PlanTVariables.class_nums.values())) + 1