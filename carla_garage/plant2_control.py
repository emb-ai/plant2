"""
PlanT 2.0: преобразование предсказаний модели в action для агента.

PlanT 2.0 output:
- K=5 траекторий, каждая 40 waypoints
- Confidence score для каждой траектории
- Выбирается траектория с максимальным confidence
- Координаты в ego frame
- PID: longitudinal (газ/тормоз) + lateral (руль)

--- Системы координат CARLA vs MetaDrive ---

CARLA (Unreal, left-handed):
  - Мир и ego: X = вперёд, Y = вправо, Z = вверх.
  - Waypoints в датасете и на выходе модели: (x, y) = (forward, right).
  - BEV в датасете: загружается PNG, затем torch.rot90(bev, dims=(1,2)) (90° CCW),
    в итоге "вперёд" в кадре модели = верх изображения.

MetaDrive (Panda3D + обёртка):
  - Мир: x, y в плоскости (right-handed по coordinates_shift).
  - Ego (BaseVehicle): первый компонент = вперёд, второй = вправо.
  - convert_to_local возвращает [ret[1], -ret[0]] от базового объекта (Panda3D),
    heading_theta = base_heading + pi/2 (модель машины смотрит по Y).
  - То есть API агента: (forward, right) в том же смысле, что и CARLA.

Если BEV из MetaDrive при визуализации повернули на -90° чтобы дорога совпала с "верхом":
  - В нашем BEV "вперёд" (ex) = верх, но дорога нарисована вдоль ey (горизонталь).
  - После поворота BEV на -90° "вперёд" в кадре = то, что было "справа" (ey).
  - Модель обучена на CARLA BEV, где вперёд = верх; на наш неповёрнутый BEV она видит
    "вперёд" по ex (верх), а дорога у нас по ey → выход модели "вперёд" может быть
    смещён на 90° относительно реального направления дороги в MetaDrive.
  - Преобразование выхода модели в ego MetaDrive: поворот на 90° CCW,
    (x_carla, y_carla) -> (-y_carla, x_carla), т.е. (forward_md, right_md) = (-model_y, model_x).
  - Если траектория всё ещё перпендикулярна или в зеркале, можно попробовать:
    (y, -x), (y, x) или (-y, -x) — см. блок waypoints_np ниже.
"""
import os
import sys
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# PlanTVariables.target_speeds (m/s) — бины для предсказания скорости при path+2hot
TARGET_SPEEDS = np.array([0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0], dtype=np.float64)
# TARGET_SPEEDS = np.array([0.05472609, 1.83970237, 3.36500943, 4.31875348, 4.71476984,
#        5.01266575, 5.41567874, 6.39454746], dtype=np.float64)


def _interpolate_waypoints(waypoints, step_m=0.1):
    """
    Как в PlanT_agent: дополняем началом (0,0), интерполируем Pchip с шагом step_m.
    waypoints: (N, 2), в ego frame.
    Returns: (M, 2) точки с шагом ~0.1 м.
    """
    from scipy.interpolate import PchipInterpolator
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if waypoints.shape[0] < 2:
        return waypoints
    waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints), axis=0)
    shift = np.roll(waypoints, 1, axis=0)
    shift[0] = shift[1]
    dists = np.linalg.norm(waypoints - shift, axis=1)
    dists = np.cumsum(dists)
    dists = dists + np.arange(len(dists)) * 1e-4
    interp = PchipInterpolator(dists, waypoints, axis=0)
    x = np.arange(step_m, dists[-1], step_m)
    if x.size == 0:
        return waypoints[None, -1]
    return interp(x)


def waypoints_to_control(waypoints_np, current_speed, target_speed_mps, device="cpu"):
    """
    Waypoints (N, 2) в ego frame -> steer, throttle, brake.
    Логика как в PlanT_agent: интерполяция waypoints, brake при target < 0.05,
    при низкой скорости и торможении — фиктивные прямые waypoints для сброса интеграла PID.
    """
    from config import GlobalConfig
    from nav_planner import LateralPIDController, get_throttle

    config = GlobalConfig()
    lat_pid = LateralPIDController(config)
    ego_loc = np.array([0.0, 0.0])
    ego_rot = 0.0
    brake = target_speed_mps < 0.05  # как в PlanT_agent (desired_speed < 0.05)

    if current_speed < 0.05 and brake:
        # Integral accumulation: фиктивные прямые waypoints
        route_for_steer = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    else:
        route_for_steer = _interpolate_waypoints(waypoints_np, step_m=0.1)

    steer = lat_pid.step(route_for_steer, current_speed, ego_loc, ego_rot)
    throttle, control_brake = get_throttle(config, brake, target_speed_mps, current_speed)
    throttle = np.clip(throttle, 0.0, getattr(config, "clip_throttle", 1.0))
    return steer, throttle, control_brake


def _desired_speed_from_pred_speed(pred_speed):
    """
    Как в PlanT_agent: softmax по бинам скорости, затем мат. ожидание по TARGET_SPEEDS.
    pred_speed: tensor (B, bins_speed) или (bins_speed,).
    """
    import torch
    t = pred_speed.detach().cpu().float().squeeze()
    if t.dim() == 0:
        t = t.unsqueeze(0)
    probs = torch.softmax(t, dim=0).numpy()
    return float(np.sum(TARGET_SPEEDS * probs))


def _desired_speed_from_waypoints(waypoints_np, current_speed):
    """
    Как в PlanT_agent при pred_speed is None: скорость из расстояния между 3-м и 4-м waypoint * 4,
    плюс creep heuristic при почти нулевой скорости.
    """
    waypoints_np = np.asarray(waypoints_np, dtype=np.float64)
    if waypoints_np.shape[0] < 4:
        return 0.0
    desired_speed = np.linalg.norm(waypoints_np[2] - waypoints_np[3]) * 4.0
    if current_speed < 0.01:
        diffs = np.linalg.norm(waypoints_np[:-1] - waypoints_np[1:], axis=1)
        if diffs.size > 0:
            mean_speed = float(np.mean(diffs)) * 4.0
            desired_speed = min(mean_speed, 0.1)
    return desired_speed


def plant2_predictions_to_action(
    pred_plan,
    current_speed,
    target_speed_mps,
    speed_limit_idx=1,
    speed_limits_kmh=(50, 80, 100, 120),
    num_modes=5,
    waypoints_per_trajectory=40,
    device="cpu",
    return_waypoints=False,
):
    """
    Преобразовать предсказания PlanT/PlanT2 в action [steer, throttle_brake].
    Логика как в PlanT_agent: желаемая скорость из pred_speed (softmax) или из waypoints + creep;
    brake при desired_speed < 0.05; интерполяция waypoints и низкоскоростной фиктивный путь для steer.

    Returns:
        action: np.ndarray [steer, throttle_brake], range [-1, 1]
        waypoints_np: (N, 2) в ego frame, если return_waypoints=True (возвращается кортеж)

    Note: pred_wps are already in ego frame — do NOT add ego world position.
    """
    pred_path, pred_wps, pred_speed = pred_plan
    # pred_wps is already in ego frame (model output = displacements from current position).
    # Using pred_path (20 pts) if pred_wps is absent; prefer pred_wps (8 pts) otherwise.
    if pred_wps is not None:
        waypoints = pred_wps.detach().cpu().numpy()
    elif pred_path is not None:
        waypoints = pred_path.detach().cpu().numpy()
    else:
        out = np.array([0.0, 0.5], dtype=np.float32)
        return (out, np.zeros((0, 2))) if return_waypoints else out
    if waypoints.ndim == 4:
        confidences = getattr(pred_plan, "confidences", None)
        print(f"confidences: {confidences}")
        if confidences is not None:
            conf = confidences.detach().cpu().numpy()
            best_k = np.argmax(conf[0])
        else:
            best_k = 0
        waypoints_np = waypoints[0, best_k]
    else:
        waypoints_np = waypoints[0]

    if waypoints_np.shape[0] == 0:
        out = np.array([0.0, 0.5], dtype=np.float32)
        return (out, np.zeros((0, 2))) if return_waypoints else out

    # noisy_waypoints = waypoints_np + np.random.normal(0, 0.2, waypoints_np.shape)
    # waypoints_np = noisy_waypoints
    # Желаемая скорость: из pred_speed (path+2hot) или из waypoints + creep (как PlanT_agent)
    if pred_speed is not None:
        desired_speed = _desired_speed_from_pred_speed(pred_speed)
    else:
        desired_speed = _desired_speed_from_waypoints(waypoints_np, current_speed)
    target_speed_mps = desired_speed

    # CARLA -> MetaDrive ego: при необходимости раскомментировать преобразование осей
    # waypoints_np = np.column_stack([-waypoints_np[:, 1], waypoints_np[:, 0]]).astype(np.float64)

    steer, throttle, brake = waypoints_to_control(
        waypoints_np, current_speed, target_speed_mps, device
    )
    throttle_brake = -0.5 if brake else throttle
    steer = np.clip(steer, -1.0, 1.0)
    steer = -steer  # CARLA: positive=right, MetaDrive: positive=left → negate
    throttle_brake = np.clip(throttle_brake, -1.0, 1.0)
    action = np.array([steer, throttle_brake], dtype=np.float32)
    return (action, np.asarray(waypoints_np, dtype=np.float64)) if return_waypoints else action


def get_target_speed_from_limit(speed_limit_idx, speed_limits_kmh=(50, 80, 100, 120)):
    """Целевая скорость в m/s по индексу лимита."""
    idx = min(3, max(0, int(speed_limit_idx)))
    return speed_limits_kmh[idx] / 3.6
