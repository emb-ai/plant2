"""
PlanT 2.0: convert model predictions into a MetaDrive action.

Control logic mirrors PlanTAgent._get_control (PlanT/PlanT_agent.py) but
outputs a MetaDrive [steer, throttle_brake] pair instead of carla.VehicleControl.

Pipeline (same as _get_control)
---------------------------------
1. desired_speed  = E[speed bins | softmax(pred_speed)]
                     OR waypoint-spacing heuristic when pred_speed is None
2. longitudinal   = LongitudinalLinearRegressionController (pure Python, no CARLA)
3. steering path  = pred_path when available, else pred_wps
4. interpolation  = PchipInterpolator at 0.1 m resolution (same as _get_control)
5. lateral        = LateralPIDController  (lateral_controller.py, no CARLA)
   • when stopped+braking: straight-ahead dummy waypoints (creep heuristic)
6. MetaDrive out  = [-steer, throttle_brake]   (sign flip: MD positive = left)

Speed bins
----------
SPEED_BINS mirrors the user-specified bins used at training time.
"""

import os
import sys
import numpy as np
from scipy.interpolate import PchipInterpolator

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Speed bin centres (m/s) — must match the bins used when training ego_speed_classifier
SPEED_BINS = np.array(
    [0.0, 0.025, 0.05472609, 1.0, 1.5, 2.0, 4.0, 8.0, 10.0, 20.0],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Waypoint interpolation (identical to PlanTAgent.interpolate_waypoints)
# ---------------------------------------------------------------------------

def interpolate_waypoints(waypoints: np.ndarray) -> np.ndarray:
    """
    Resample waypoints at 0.1 m intervals using PCHIP interpolation.

    Args:
        waypoints: (N, 2) array in ego frame.

    Returns:
        (M, 2) densely resampled waypoints, M ≈ total_length / 0.1.
        If all points collapse to one location the last waypoint is returned.
    """
    waypoints = waypoints.copy().astype(np.float64)
    # Prepend ego origin so cumulative distance starts at 0
    waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
    shift = np.roll(waypoints, 1, axis=0)
    shift[0] = shift[1]
    dists = np.linalg.norm(waypoints - shift, axis=1)
    dists = np.cumsum(dists)
    dists += np.arange(len(dists)) * 1e-4   # guarantee strictly increasing

    interp = PchipInterpolator(dists, waypoints, axis=0)
    x = np.arange(0.1, dists[-1], 0.1)
    result = interp(x)

    if result.shape[0] == 0:
        result = waypoints[None, -1]

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def plant2_predictions_to_action(
    pred_plan,
    current_speed: float,
    target_speed_mps: float,
    speed_limit_idx: int = 1,
    speed_limits_kmh: tuple = (50, 80, 100, 120),
    device: str = "cpu",
    return_waypoints: bool = False,
):
    """
    Convert PlanT / HFLM predictions to a MetaDrive action.

    Args:
        pred_plan:        (pred_path, pred_wps, pred_speed) from HFLM.forward
        current_speed:    ego speed in m/s
        target_speed_mps: speed-limit cap (m/s) from get_target_speed_from_limit
        return_waypoints: if True return (action, waypoints_np) tuple

    Returns:
        action: np.ndarray [steer, throttle_brake] in [-1, 1]
        (optional) waypoints_np: (N, 2) float64 ego-frame waypoints
    """
    import torch
    from config import GlobalConfig
    from lateral_controller import LateralPIDController
    from longitudinal_controller import LongitudinalLinearRegressionController

    pred_path, pred_wps, pred_speed = pred_plan

    config  = GlobalConfig()
    lon_pid = LongitudinalLinearRegressionController(config)
    lat_pid = LateralPIDController(config)

    # ------------------------------------------------------------------
    # 1. Desired speed
    # ------------------------------------------------------------------
    if pred_speed is not None:
        logits = pred_speed.detach().float()
        if logits.dim() > 1:
            logits = logits.squeeze(0)          # (C,)
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        desired_speed = float((probs * SPEED_BINS).sum())
    else:
        # Waypoint-spacing heuristic — mirrors _get_control fallback
        _wp = pred_wps if pred_wps is not None else pred_path
        if _wp is not None:
            wp_arr = _wp.detach().cpu().numpy()
            if wp_arr.ndim > 2:
                wp_arr = wp_arr.squeeze(0)
            if len(wp_arr) >= 4:
                desired_speed = float(np.linalg.norm(wp_arr[2] - wp_arr[3]) * 4.0)
                mean_speed = float(np.linalg.norm(wp_arr[:-1] - wp_arr[1:], axis=-1).mean() * 4.0)
                if current_speed < 0.01:
                    desired_speed = min(mean_speed, 0.1)
            else:
                desired_speed = target_speed_mps
        else:
            desired_speed = target_speed_mps

    # Cap by speed limit (safety ceiling — model should respect it already)
    desired_speed = min(desired_speed, target_speed_mps)

    # ------------------------------------------------------------------
    # 2. Longitudinal control
    # ------------------------------------------------------------------
    # Brake whenever desired speed is near zero — mirrors _get_control threshold
    hazard_brake = desired_speed < 0.05
    throttle, brake = lon_pid.get_throttle_and_brake(hazard_brake, desired_speed, current_speed)

    # ------------------------------------------------------------------
    # 3. Select & extract steering waypoints
    # ------------------------------------------------------------------
    # Prefer pred_path for steering, pred_wps as fallback (mirrors _get_control)
    steer_tensor = pred_path if pred_path is not None else pred_wps
    speed_tensor = pred_wps  if pred_wps  is not None else pred_path

    _no_wps = np.array([0.0, 0.5], dtype=np.float32)
    if steer_tensor is None:
        return (_no_wps, np.zeros((0, 2), dtype=np.float64)) if return_waypoints else _no_wps

    steer_np = steer_tensor.detach().cpu().numpy()
    if steer_np.ndim > 2:
        steer_np = steer_np.squeeze(0)          # (N, 2)
    if steer_np.shape[0] == 0:
        return (_no_wps, np.zeros((0, 2), dtype=np.float64)) if return_waypoints else _no_wps

    # Waypoints returned to caller (for visualisation / loss computation)
    if speed_tensor is not None:
        out_wps = speed_tensor.detach().cpu().numpy()
        if out_wps.ndim > 2:
            out_wps = out_wps.squeeze(0)
    else:
        out_wps = steer_np

    # ------------------------------------------------------------------
    # 4. Interpolate steering path at 0.1 m resolution
    # ------------------------------------------------------------------
    interp_wp = interpolate_waypoints(steer_np)

    # ------------------------------------------------------------------
    # 5. Lateral control
    # ------------------------------------------------------------------
    if current_speed < 0.05 and brake:
        # Creep heuristic: prevent integral wind-up when stopped with brake
        # by feeding a straight-ahead dummy path (mirrors _get_control)
        steer_input = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    else:
        steer_input = interp_wp

    steer = lat_pid.step(
        steer_input,
        current_speed,
        np.array([0.0, 0.0]),   # ego at origin in ego frame
        0.0,                     # heading = 0 in ego frame
        False,                   # inference_mode=False (matches PlanTAgent)
    )

    # ------------------------------------------------------------------
    # 6. Assemble MetaDrive action
    # ------------------------------------------------------------------
    steer = np.clip(float(steer), -1.0, 1.0)
    steer = -steer                              # PID: positive=right; MetaDrive: positive=left

    if brake:
        throttle_brake = -0.5
    else:
        throttle_brake = float(np.clip(throttle, 0.0, 1.0))
    throttle_brake = np.clip(throttle_brake, -1.0, 1.0)

    action = np.array([steer, throttle_brake], dtype=np.float32)
    return (action, np.asarray(out_wps, dtype=np.float64)) if return_waypoints else action


# ---------------------------------------------------------------------------
# Speed-limit helper (unchanged)
# ---------------------------------------------------------------------------

def get_target_speed_from_limit(speed_limit_idx, speed_limits_kmh=(50, 80, 100, 120)):
    """Map speed-limit index (0-3) to m/s."""
    idx = min(3, max(0, int(speed_limit_idx)))
    return speed_limits_kmh[idx] / 3.6
