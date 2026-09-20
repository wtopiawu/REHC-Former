import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-8
LENGTH_UNIT_TO_METERS = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
}

TEC_LABELS = {"tec", "t_ec", "eye_in_hand", "camera_to_ee", "camera_to_end_effector"}
TCE_LABELS = {"tce", "t_ce", "ee_to_camera", "end_effector_to_camera"}


def normalize_quaternion_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q)
    if n < EPS:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return q / n


def quat_xyzw_to_matrix_np(q: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def matrix_to_quat_xyzw_np(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float32)
    trace = np.trace(R)

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float32)
    return normalize_quaternion_xyzw(q)


def pose_to_matrix_np(t: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quat_xyzw_to_matrix_np(q_xyzw)
    T[:3, 3] = np.asarray(t, dtype=np.float32)
    return T


def matrix_to_pose_np(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=np.float32)
    t = T[:3, 3].astype(np.float32)
    q = matrix_to_quat_xyzw_np(T[:3, :3])
    return t, q


def invert_transform_np(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float32)
    R = T[:3, :3]
    t = T[:3, 3]
    inv_T = np.eye(4, dtype=np.float32)
    inv_T[:3, :3] = R.T
    inv_T[:3, 3] = -R.T @ t
    return inv_T


def transform_points_np(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    R = np.asarray(T[:3, :3], dtype=np.float32)
    t = np.asarray(T[:3, 3], dtype=np.float32)
    return points @ R.T + t


def compute_add_from_poses(points: np.ndarray, pred_T: np.ndarray, gt_T: np.ndarray) -> float:
    pred_points = transform_points_np(points, pred_T)
    gt_points = transform_points_np(points, gt_T)
    distances = np.linalg.norm(pred_points - gt_points, axis=1)
    return float(distances.mean())


def compute_add_from_keypoints(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
    pred_points = np.asarray(pred_points, dtype=np.float32)
    gt_points = np.asarray(gt_points, dtype=np.float32)
    if pred_points.shape != gt_points.shape:
        raise ValueError(f"预测/真值关键点维度不一致: pred={pred_points.shape}, gt={gt_points.shape}")
    distances = np.linalg.norm(pred_points - gt_points, axis=1)
    return float(distances.mean())


def convert_length_array(values: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    if from_unit not in LENGTH_UNIT_TO_METERS:
        raise ValueError(f"不支持的长度单位: {from_unit}")
    if to_unit not in LENGTH_UNIT_TO_METERS:
        raise ValueError(f"不支持的长度单位: {to_unit}")
    values = np.asarray(values, dtype=np.float32)
    if from_unit == to_unit:
        return values.astype(np.float32)
    scale = LENGTH_UNIT_TO_METERS[from_unit] / LENGTH_UNIT_TO_METERS[to_unit]
    return (values * scale).astype(np.float32)


def convert_transform_translation_unit(T: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    out = np.asarray(T, dtype=np.float32).copy()
    out[:3, 3] = convert_length_array(out[:3, 3], from_unit=from_unit, to_unit=to_unit)
    return out


def _parse_matrix_unit(unit_text: str, default: str = "mm") -> str:
    unit_text = str(unit_text or "").lower()
    if "mm" in unit_text:
        return "mm"
    if "cm" in unit_text:
        return "cm"
    if "meter" in unit_text or unit_text == "m" or unit_text.startswith("m_"):
        return "m"
    return default


def infer_target_length_unit(meta: Dict, label_mode: str = "tec") -> str:
    label = str(label_mode).strip().lower()
    if label == "auto":
        label = "tec" if "T_EC" in meta else "tce"
    if label in TEC_LABELS:
        return _parse_matrix_unit(meta.get("T_EC_unit", ""), default="mm")
    if label in TCE_LABELS:
        return _parse_matrix_unit(meta.get("T_CE_unit", ""), default="mm")
    raise ValueError(f"不支持的 label_mode: {label_mode}. 建议使用 tec 或 tce。")


def _get_transform_matrix_from_meta(meta: Dict, key: str, output_unit: str) -> np.ndarray:
    if key in meta:
        T = np.asarray(meta[key], dtype=np.float32)
        from_unit = _parse_matrix_unit(meta.get(f"{key}_unit", ""), default="mm")
        return convert_transform_translation_unit(T, from_unit=from_unit, to_unit=output_unit)

    debug_key = f"{key}_m_debug"
    if debug_key in meta:
        T = np.asarray(meta[debug_key], dtype=np.float32)
        return convert_transform_translation_unit(T, from_unit="m", to_unit=output_unit)

    raise KeyError(f"JSON 缺少 {key} 或 {debug_key}")


def extract_target_from_json(
    meta: Dict,
    label_mode: str = "tec",
    output_unit: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Extract the eye-in-hand training target.

    Default target is T_EC, which maps camera-frame points C into the end-effector frame E:
    X_E = R_EC * X_C + t_EC.
    Translation is normally stored in millimeters in this project's JSON files.
    """
    label = str(label_mode).strip().lower()
    if label == "auto":
        label = "tec" if "T_EC" in meta or "T_EC_m_debug" in meta else "tce"

    target_unit = output_unit or infer_target_length_unit(meta, label_mode=label)

    if label in TEC_LABELS:
        T = _get_transform_matrix_from_meta(meta, "T_EC", output_unit=target_unit)
        t, q = matrix_to_pose_np(T)
        return t, q, "T_EC_camera_to_end_effector"

    if label in TCE_LABELS:
        T = _get_transform_matrix_from_meta(meta, "T_CE", output_unit=target_unit)
        t, q = matrix_to_pose_np(T)
        return t, q, "T_CE_end_effector_to_camera"

    raise ValueError(f"不支持的 label_mode: {label_mode}. 建议使用 tec 或 tce。")


def pose_represents_camera_to_ee(label_mode: str = "", pose_name: str = "") -> Optional[bool]:
    label = str(label_mode).strip().lower()
    pose = str(pose_name).strip().lower()

    if label == "auto":
        # Check pose name when checkpoint was created with auto.
        label = "tec" if "t_ec" in pose or "camera_to_end" in pose else "tce"

    if label in TEC_LABELS:
        return True
    if label in TCE_LABELS:
        return False

    if pose in {"t_ec", "tec", "t_ec_camera_to_end_effector"} or "camera_to_end_effector" in pose:
        return True
    if pose in {"t_ce", "tce", "t_ce_end_effector_to_camera"} or "end_effector_to_camera" in pose:
        return False
    return None


def pose_to_T_EC_transform(T: np.ndarray, label_mode: str = "", pose_name: str = "") -> np.ndarray:
    direction = pose_represents_camera_to_ee(label_mode=label_mode, pose_name=pose_name)
    if direction is None:
        raise ValueError(f"无法判断位姿方向: label_mode={label_mode}, pose_name={pose_name}")
    if direction:
        return np.asarray(T, dtype=np.float32)
    return invert_transform_np(np.asarray(T, dtype=np.float32))


def pose_to_T_CE_transform(T: np.ndarray, label_mode: str = "", pose_name: str = "") -> np.ndarray:
    T_EC = pose_to_T_EC_transform(T, label_mode=label_mode, pose_name=pose_name)
    return invert_transform_np(T_EC)


# Backward-compatible name for older scripts. In this eye-in-hand project this returns T_EC.
def pose_to_eye_in_hand_transform(T: np.ndarray, label_mode: str = "", pose_name: str = "") -> np.ndarray:
    return pose_to_T_EC_transform(T, label_mode=label_mode, pose_name=pose_name)


def rotation_6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    return R[..., :2, :].reshape(R.shape[:-2] + (6,))


def project_matrix_to_rotation_torch(R: torch.Tensor) -> torch.Tensor:
    """Project arbitrary 3x3 matrices to the closest proper rotations."""
    R = R.float()
    U, _, Vh = torch.linalg.svd(R)
    R_projected = U @ Vh
    det = torch.linalg.det(R_projected)
    if torch.any(det < 0):
        U = U.clone()
        U[..., :, -1] = torch.where(
            (det < 0).unsqueeze(-1),
            -U[..., :, -1],
            U[..., :, -1],
        )
        R_projected = U @ Vh
    return R_projected


def rotation_9d_to_matrix(x: torch.Tensor, project: bool = True) -> torch.Tensor:
    R = x.reshape(x.shape[:-1] + (3, 3))
    if project:
        return project_matrix_to_rotation_torch(R)
    return R


def matrix_to_rotation_9d(R: torch.Tensor) -> torch.Tensor:
    return R.reshape(R.shape[:-2] + (9,))


def quat_xyzw_to_matrix_torch(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    x, y, z, w = q.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)

    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)

    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    return torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )


def geodesic_distance_from_two_matrices(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    relative = R1 @ R2.transpose(-1, -2)
    trace = relative[..., 0, 0] + relative[..., 1, 1] + relative[..., 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos_theta)


def matrix_to_quat_xyzw_torch(R: torch.Tensor) -> torch.Tensor:
    Rs = R.detach().cpu().numpy()
    quats = [matrix_to_quat_xyzw_np(r) for r in Rs]
    return torch.from_numpy(np.stack(quats, axis=0))


def build_transform_matrix_from_prediction(t: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    return pose_to_matrix_np(t, q_xyzw)
