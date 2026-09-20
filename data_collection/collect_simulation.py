import argparse
import json
import time

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ============================================================
# 1. 配置区
# ============================================================

OUT_DIR = Path("data/simulation")

CAMERA_PATH = "/UR5/Vision_sensor"
MASK_CAMERA_PATH = "/UR5/Vision_sensor_mask"
EE_PATH = "/UR5/connection"
UR5_PATH = "/UR5"

# Example gripper root; override through --scene-config.
END_EFFECTOR_ROOT_PATH = "/UR5/connection/EG2_4CX_step"
END_EFFECTOR_MODEL_NAME = "EG2_4CX_step_tree"

# 100 组不同 T_EC，每组 15 张
NUM_TEC_GROUPS = 100
IMAGES_PER_GROUP = 15

# 每组 T_EC 的随机范围：
# 以当前 Vision_sensor 相对 connection 的姿态为标准，
# 每组随机一个新的 T_EC。
# 位置在 3cm x 3cm x 3cm 体积内随机，即每轴 ±15mm；
# 姿态绕 x/y/z 各轴 ±5°。
TEC_TRANS_HALF_RANGE_M = 0.015
TEC_ROT_HALF_RANGE_DEG = 5.0

# 每组内部，机械臂末端变换范围：
# 以当前末端姿态为基准，3cm x 3cm x 3cm 体积内随机，即每轴 ±15mm；
# 末端姿态绕 x/y/z 各轴 ±5°。
# Reduce the motion range if IK frequently fails in the configured scene.
EE_TRANS_HALF_RANGE_M = 0.015
EE_ROT_HALF_RANGE_DEG = 5.0

# 图像方向设置
# FLIP_UP_DOWN=True   : 上下翻转，相当于 np.flipud(img)
# FLIP_LEFT_RIGHT=True: 左右翻转，相当于 np.fliplr(img)
# Example rendering convention; verify against the configured scene.
FLIP_UP_DOWN = False
FLIP_LEFT_RIGHT = True

# 是否先安全打开夹爪
OPEN_GRIPPER_FIRST = False
GRIPPER_OPEN_CLOSE_JOINT = "/UR5/RG2/openCloseJoint"
GRIPPER_OPEN_TARGET = 0.0538
# 如果夹爪方向反了，把 GRIPPER_OPEN_TARGET 改成 0.0

BLACK_MEAN_THRESHOLD = 1.0

# Mask 输出设置
SAVE_MASK_IMAGE = True
REQUIRE_COMPLETE_SAMPLE_TRIPLET = True
MASK_READ_RETRIES = 4
MASK_THRESHOLD = 10
MASK_MORPHOLOGY_ENABLED = False
AUTO_SETUP_MASK_CAMERA_RENDER_ONLY_EE = True
MIN_MASK_AREA_RATIO = 0.002
PNG_COMPRESSION = 1

# IK 参数
IK_MAX_ITERS = 80
IK_POS_TOL_M = 0.0015
IK_ROT_TOL_RAD = np.deg2rad(0.8)
IK_DAMPING = 1e-3
IK_NUMERIC_EPS = 1e-4
IK_MAX_STEP_RAD = 0.06

# 每张图最多尝试多少次 IK / 采图
MAX_ATTEMPTS_PER_IMAGE = 100

# 采完后是否恢复初始状态
RESTORE_INITIAL_STATE = True

# ============================================================
# 1.1 EG2 整体颜色配置
# ============================================================

SET_EE_BLACK_MATERIAL = True
SET_EE_WHITE_MATERIAL_FOR_MASK_READ = True


# ============================================================
# 2. 基础工具函数
# ============================================================

def ensure_dirs():
    (OUT_DIR / "rgb").mkdir(parents=True, exist_ok=True)
    if SAVE_MASK_IMAGE:
        (OUT_DIR / "mask").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "anno").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "groups").mkdir(parents=True, exist_ok=True)


def pose_to_T(pose):
    """
    CoppeliaSim pose:
        [x, y, z, qx, qy, qz, qw]

    sim.getObjectPose(obj, ref) 表示 obj frame -> ref frame。
    例如 sim.getObjectPose(camera, ee) 得到 camera -> ee，即 T_EC。
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array(pose[:3], dtype=np.float64)
    T[:3, :3] = R.from_quat(pose[3:7]).as_matrix()
    return T


def T_to_pose(T):
    q = R.from_matrix(T[:3, :3]).as_quat()
    t = T[:3, 3]
    return [
        float(t[0]), float(t[1]), float(t[2]),
        float(q[0]), float(q[1]), float(q[2]), float(q[3])
    ]


def set_object_pose(sim, obj, ref, pose):
    """
    兼容不同 CoppeliaSim 版本的 setObjectPose 参数顺序。
    """
    try:
        sim.setObjectPose(obj, ref, pose)
    except Exception:
        sim.setObjectPose(obj, pose, ref)


def T_m_to_T_mm(T_m):
    """
    只把平移从 m 转成 mm，旋转矩阵不变。
    """
    T_mm = np.array(T_m, dtype=np.float64).copy()
    T_mm[:3, 3] *= 1000.0
    return T_mm


def matrix_to_list(T):
    return np.asarray(T, dtype=np.float64).tolist()


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_write_image(path, image, params=None):
    if params is None:
        ok = cv2.imwrite(str(path), image)
    else:
        ok = cv2.imwrite(str(path), image, params)
    return bool(ok and path.exists() and path.stat().st_size > 0)


def cleanup_partial_sample_files(*paths):
    for path in paths:
        if path is None:
            continue
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def transform_error(T_target, T_current):
    """
    返回 6D 误差：
        前 3 维：位置误差，单位 m
        后 3 维：旋转误差，axis-angle rotvec，单位 rad
    """
    dp = T_target[:3, 3] - T_current[:3, 3]

    R_err = T_target[:3, :3] @ T_current[:3, :3].T
    rotvec = R.from_matrix(R_err).as_rotvec()

    return np.concatenate([dp, rotvec], axis=0)


# ============================================================
# 3. T_EC 分组采样
# ============================================================

def sample_group_TEC(T_EC_base):
    """
    以当前标准 T_EC 为中心，为一组数据随机生成一个固定 T_EC。

    这个 T_EC 在该组 15 张图中保持不变。
    """
    delta_t = np.random.uniform(
        low=-TEC_TRANS_HALF_RANGE_M,
        high=TEC_TRANS_HALF_RANGE_M,
        size=3
    )

    delta_rpy_deg = np.random.uniform(
        low=-TEC_ROT_HALF_RANGE_DEG,
        high=TEC_ROT_HALF_RANGE_DEG,
        size=3
    )

    R_delta = R.from_euler("xyz", delta_rpy_deg, degrees=True).as_matrix()

    T_group = np.array(T_EC_base, dtype=np.float64).copy()
    T_group[:3, 3] = T_EC_base[:3, 3] + delta_t
    T_group[:3, :3] = T_EC_base[:3, :3] @ R_delta

    return T_group, delta_t, delta_rpy_deg


# ============================================================
# 4. UR5 关节控制与数值 IK
# ============================================================

def get_ur5_arm_joints(sim):
    """
    自动找到 UR5 的 6 个主关节，排除 RG2 内部关节。
    """
    ur5 = sim.getObject(UR5_PATH)
    all_joints = sim.getObjectsInTree(ur5, sim.object_joint_type, 0)

    arm_joints = []
    for h in all_joints:
        name = sim.getObjectAlias(h, 1)
        if "RG2" in name:
            continue
        arm_joints.append(h)

    arm_joints = arm_joints[:6]

    print("UR5 arm joints:")
    for h in arm_joints:
        print("  ", sim.getObjectAlias(h, 1))

    if len(arm_joints) != 6:
        print("[WARN] 找到的 UR5 主关节不是 6 个，请检查对象树。")

    return arm_joints


def get_joint_positions(sim, joint_handles):
    return np.array([sim.getJointPosition(h) for h in joint_handles], dtype=np.float64)


def set_joint_positions(sim, joint_handles, q):
    for h, qi in zip(joint_handles, q):
        try:
            sim.setJointTargetPosition(h, float(qi))
        except Exception:
            pass

        try:
            sim.setJointPosition(h, float(qi))
        except Exception:
            pass


def get_ee_T_base(sim, ee, base):
    return pose_to_T(sim.getObjectPose(ee, base))


def compute_numeric_jacobian(sim, joint_handles, ee, base, q_current):
    """
    通过有限差分计算末端 6D 数值雅可比。
    """
    set_joint_positions(sim, joint_handles, q_current)
    time.sleep(0.002)

    T0 = get_ee_T_base(sim, ee, base)
    J = np.zeros((6, len(joint_handles)), dtype=np.float64)

    for i in range(len(joint_handles)):
        q_plus = q_current.copy()
        q_plus[i] += IK_NUMERIC_EPS

        set_joint_positions(sim, joint_handles, q_plus)
        time.sleep(0.002)

        T_plus = get_ee_T_base(sim, ee, base)

        dp = (T_plus[:3, 3] - T0[:3, 3]) / IK_NUMERIC_EPS

        R_diff = T_plus[:3, :3] @ T0[:3, :3].T
        drot = R.from_matrix(R_diff).as_rotvec() / IK_NUMERIC_EPS

        J[:3, i] = dp
        J[3:, i] = drot

    set_joint_positions(sim, joint_handles, q_current)
    time.sleep(0.002)

    return J


def solve_ik_numeric(sim, joint_handles, ee, base, T_target, q_seed):
    """
    用 damped least squares 数值 IK 求解目标末端位姿。
    返回:
        success, q_solution, final_pos_error_m, final_rot_error_deg
    """
    q = q_seed.copy()

    for _ in range(IK_MAX_ITERS):
        set_joint_positions(sim, joint_handles, q)
        time.sleep(0.003)

        T_current = get_ee_T_base(sim, ee, base)
        err = transform_error(T_target, T_current)

        pos_err = np.linalg.norm(err[:3])
        rot_err = np.linalg.norm(err[3:])

        if pos_err < IK_POS_TOL_M and rot_err < IK_ROT_TOL_RAD:
            return True, q, pos_err, np.rad2deg(rot_err)

        J = compute_numeric_jacobian(sim, joint_handles, ee, base, q)

        A = J @ J.T + (IK_DAMPING ** 2) * np.eye(6)

        try:
            dq = J.T @ np.linalg.solve(A, err)
        except np.linalg.LinAlgError:
            return False, q, pos_err, np.rad2deg(rot_err)

        max_abs = np.max(np.abs(dq))
        if max_abs > IK_MAX_STEP_RAD:
            dq = dq / max_abs * IK_MAX_STEP_RAD

        q = q + dq

    set_joint_positions(sim, joint_handles, q)

    T_current = get_ee_T_base(sim, ee, base)
    err = transform_error(T_target, T_current)

    pos_err = np.linalg.norm(err[:3])
    rot_err = np.linalg.norm(err[3:])

    success = pos_err < IK_POS_TOL_M * 2 and rot_err < IK_ROT_TOL_RAD * 2
    return success, q, pos_err, np.rad2deg(rot_err)


def sample_target_ee_pose(T_BE_base):
    """
    Sample around the current pose using the configured translation/rotation ranges.
    """
    delta_t = np.random.uniform(
        low=-EE_TRANS_HALF_RANGE_M,
        high=EE_TRANS_HALF_RANGE_M,
        size=3
    )

    delta_rpy_deg = np.random.uniform(
        low=-EE_ROT_HALF_RANGE_DEG,
        high=EE_ROT_HALF_RANGE_DEG,
        size=3
    )

    R_delta = R.from_euler("xyz", delta_rpy_deg, degrees=True).as_matrix()

    T_target = np.array(T_BE_base, dtype=np.float64).copy()
    T_target[:3, 3] = T_BE_base[:3, 3] + delta_t
    T_target[:3, :3] = T_BE_base[:3, :3] @ R_delta

    return T_target, delta_t, delta_rpy_deg


# ============================================================
# 5. 安全打开 RG2：只控制 openCloseJoint
# ============================================================

def open_rg2_safely(sim):
    """
    只设置 /UR5/RG2/openCloseJoint。
    不设置 prismJoint、leftJoint、rightJoint 等内部联动关节，避免夹爪变形。
    """
    print("\nOpening RG2 safely by openCloseJoint only...")

    status = {}

    try:
        joint = sim.getObject(GRIPPER_OPEN_CLOSE_JOINT)
    except Exception:
        print(f"[WARN] Cannot find joint: {GRIPPER_OPEN_CLOSE_JOINT}")
        return status

    target = float(GRIPPER_OPEN_TARGET)

    for _ in range(5):
        try:
            sim.setJointTargetPosition(joint, target)
        except Exception:
            pass

        try:
            sim.setJointPosition(joint, target)
        except Exception:
            pass

        time.sleep(0.05)

    status["joint_path"] = GRIPPER_OPEN_CLOSE_JOINT
    status["target"] = target

    print(f"  {GRIPPER_OPEN_CLOSE_JOINT} -> {target:.6f}")
    print("RG2 safe open command finished.\n")

    return status



# ============================================================
# 5.1 EG2 整体黑色材质
# ============================================================

def get_end_effector_shape_handles(sim):
    """
    获取当前 EG2_4CX_step 夹爪树下所有 shape。
    """
    try:
        root = sim.getObject(END_EFFECTOR_ROOT_PATH)
    except Exception:
        print(f"[WARN] Cannot find END_EFFECTOR_ROOT_PATH: {END_EFFECTOR_ROOT_PATH}")
        return []

    try:
        shapes = sim.getObjectsInTree(root, sim.object_shape_type, 0)
    except Exception:
        print("[WARN] Cannot get shape objects in end-effector tree.")
        return []

    return shapes


def set_whole_end_effector_material(sim, shape_handles, color, mode):
    """给整个夹爪树施加统一材质颜色。"""
    records = []

    if len(shape_handles) == 0:
        return records

    color_component = getattr(sim, "colorcomponent_ambient_diffuse", 0)
    success_count = 0

    for shape in shape_handles:
        success = False
        try:
            sim.setShapeColor(shape, None, color_component, color)
            success = True
        except Exception:
            try:
                sim.setShapeColor(shape, "", color_component, color)
                success = True
            except Exception:
                success = False
        if success:
            success_count += 1

    records.append({
        "mode": mode,
        "rgb01": color,
        "shape_count": int(len(shape_handles)),
        "success_count": int(success_count),
        "root_path": END_EFFECTOR_ROOT_PATH,
    })

    return records


def set_whole_end_effector_black_material(sim, shape_handles):
    if not SET_EE_BLACK_MATERIAL:
        return []

    return set_whole_end_effector_material(
        sim=sim,
        shape_handles=shape_handles,
        color=[0.0, 0.0, 0.0],
        mode="whole_gripper_black",
    )


def set_whole_end_effector_white_material_for_mask(sim, shape_handles):
    if not SET_EE_WHITE_MATERIAL_FOR_MASK_READ:
        return []

    return set_whole_end_effector_material(
        sim=sim,
        shape_handles=shape_handles,
        color=[1.0, 1.0, 1.0],
        mode="whole_gripper_white_for_mask_read",
    )


# ============================================================
# 6. Vision Sensor 读取
# ============================================================

def setup_mask_camera_render_only_end_effector(sim, mask_camera):
    ee_root = sim.getObject(END_EFFECTOR_ROOT_PATH)
    collection = sim.createCollection(0)
    sim.addItemToCollection(collection, sim.handle_tree, ee_root, 0)
    sim.setObjectInt32Param(mask_camera, sim.visionintparam_entity_to_render, collection)
    print(f"Mask camera now renders only tree: {END_EFFECTOR_ROOT_PATH}")
    return collection


def _buffer_to_rgb(sim, img_buf, res):
    width, height = int(res[0]), int(res[1])

    if isinstance(img_buf, (bytes, bytearray)):
        img = np.frombuffer(img_buf, dtype=np.uint8)
    else:
        img = np.array(img_buf, dtype=np.uint8)

    if img.size != width * height * 3:
        img = np.array(sim.unpackUInt8Table(img_buf), dtype=np.uint8)

    img = img.reshape(height, width, 3)

    if FLIP_UP_DOWN:
        img = np.flipud(img)

    if FLIP_LEFT_RIGHT:
        img = np.fliplr(img)

    return img, width, height


def force_read_rgb_static(sim, camera):
    """
    不启动仿真，直接刷新并读取 Vision Sensor 图像。
    """
    last_img = None
    last_w = None
    last_h = None

    for _ in range(10):
        try:
            sim.handleVisionSensor(camera)
        except Exception:
            pass

        time.sleep(0.03)

        img_buf, res = sim.getVisionSensorImg(camera)
        img, width, height = _buffer_to_rgb(sim, img_buf, res)

        last_img = img
        last_w = width
        last_h = height

        if img.mean() > BLACK_MEAN_THRESHOLD and img.max() > 0:
            return img, width, height, True

    return last_img, last_w, last_h, False


def force_read_mask_static(sim, mask_camera):
    last_mask = None
    last_width = None
    last_height = None
    last_area_ratio = 0.0

    for _ in range(max(1, int(MASK_READ_RETRIES))):
        try:
            sim.handleVisionSensor(mask_camera)
        except Exception:
            pass

        time.sleep(0.03)

        img_buf, res = sim.getVisionSensorImg(mask_camera)
        rgb_mask, width, height = _buffer_to_rgb(sim, img_buf, res)

        gray = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2GRAY)
        mask = (gray > MASK_THRESHOLD).astype(np.uint8) * 255

        if MASK_MORPHOLOGY_ENABLED:
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        area_ratio = float((mask > 0).mean())
        last_mask = mask
        last_width = width
        last_height = height
        last_area_ratio = area_ratio

        if area_ratio >= MIN_MASK_AREA_RATIO:
            return mask, width, height, True, area_ratio

    return last_mask, last_width, last_height, False, last_area_ratio


def audit_dataset_triplets(out_dir):
    rgb_dir = out_dir / "rgb"
    mask_dir = out_dir / "mask"
    anno_dir = out_dir / "anno"

    rgb_stems = {p.stem for p in rgb_dir.glob("*.png")}
    mask_stems = {p.stem for p in mask_dir.glob("*.png")}
    anno_stems = {p.stem for p in anno_dir.glob("*.json")}

    missing_mask = sorted(rgb_stems - mask_stems)
    missing_anno = sorted(rgb_stems - anno_stems)
    missing_rgb_for_mask = sorted(mask_stems - rgb_stems)
    missing_rgb_for_anno = sorted(anno_stems - rgb_stems)

    ok = not any([missing_mask, missing_anno, missing_rgb_for_mask, missing_rgb_for_anno])
    return {
        "ok": bool(ok),
        "rgb_count": int(len(rgb_stems)),
        "mask_count": int(len(mask_stems)),
        "anno_count": int(len(anno_stems)),
        "missing_mask_for_rgb": missing_mask[:20],
        "missing_anno_for_rgb": missing_anno[:20],
        "missing_rgb_for_mask": missing_rgb_for_mask[:20],
        "missing_rgb_for_anno": missing_rgb_for_anno[:20],
    }

# ============================================================
# 7. 单张图采集
# ============================================================

def collect_one_image(
    sim,
    camera,
    mask_camera,
    ee,
    base,
    arm_joints,
    q0,
    ee_shape_handles,
    T_BE_base_m,
    T_EC_group_m,
    T_EC_group_mm,
    pose_EC_group,
    group_index,
    image_index_in_group,
    global_index,
    gripper_open_status,
):
    """
    在固定 T_EC_group 下，通过末端小范围运动采集一张图。
    """
    attempts = 0

    while attempts < MAX_ATTEMPTS_PER_IMAGE:
        attempts += 1

        # 1. 采样目标末端位姿
        T_BE_target_m, delta_ee_t_m, delta_ee_rpy_deg = sample_target_ee_pose(T_BE_base_m)

        # 2. IK 求解
        success, q_sol, ik_pos_err_m, ik_rot_err_deg = solve_ik_numeric(
            sim=sim,
            joint_handles=arm_joints,
            ee=ee,
            base=base,
            T_target=T_BE_target_m,
            q_seed=q0
        )

        if not success:
            continue

        # 3. 应用 IK
        set_joint_positions(sim, arm_joints, q_sol)

        # 4. 强制保持该组 T_EC 不变
        set_object_pose(sim, camera, ee, pose_EC_group)
        if mask_camera is not None:
            set_object_pose(sim, mask_camera, ee, pose_EC_group)

        time.sleep(0.03)

        # 5. 渲染前把整个 EG2 夹爪树设成黑色
        ee_color_records = set_whole_end_effector_black_material(
            sim=sim,
            shape_handles=ee_shape_handles
        )

        # 6. 读取图像
        rgb, width, height, ok = force_read_rgb_static(sim, camera)

        if not ok:
            continue

        # 7. 读取 mask。mask camera 与 RGB camera 使用同一个 T_EC 和同一套翻转。
        mask = None
        mask_ok = False
        mask_area_ratio = 0.0
        mask_color_records = []
        if SAVE_MASK_IMAGE:
            if mask_camera is None:
                print("[WARN] mask camera is None, skip this sample.")
                continue

            if SET_EE_BLACK_MATERIAL:
                mask_color_records = set_whole_end_effector_white_material_for_mask(
                    sim=sim,
                    shape_handles=ee_shape_handles
                )

            try:
                mask, mw, mh, mask_ok, mask_area_ratio = force_read_mask_static(sim, mask_camera)
            finally:
                if SET_EE_BLACK_MATERIAL:
                    set_whole_end_effector_black_material(
                        sim=sim,
                        shape_handles=ee_shape_handles
                    )

            if not mask_ok or mask is None:
                print(
                    f"[WARN] mask read failed, group={group_index}, "
                    f"sample={image_index_in_group}, area_ratio={mask_area_ratio:.6f}"
                )
                continue

            if mask.shape[:2] != rgb.shape[:2]:
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # 8. 读取实际 T_EC
        pose_EC_actual = sim.getObjectPose(camera, ee)
        T_EC_actual_m = pose_to_T(pose_EC_actual)
        T_EC_actual_mm = T_m_to_T_mm(T_EC_actual_m)

        max_abs_TEC_diff_mm = float(np.abs(T_EC_actual_mm - T_EC_group_mm).max())

        # 9. 记录实际末端位姿
        T_BE_actual_m = get_ee_T_base(sim, ee, base)
        T_BC_actual_m = T_BE_actual_m @ T_EC_actual_m

        ee_pose_error = transform_error(T_BE_target_m, T_BE_actual_m)
        ee_pos_error_mm = float(np.linalg.norm(ee_pose_error[:3]) * 1000.0)
        ee_rot_error_deg = float(np.rad2deg(np.linalg.norm(ee_pose_error[3:])))

        # 10. 保存
        name = f"g{group_index:03d}_s{image_index_in_group:02d}"
        image_rel = f"rgb/{name}.png"
        mask_rel = f"mask/{name}.png" if SAVE_MASK_IMAGE else None
        anno_rel = f"anno/{name}.json"

        image_path = OUT_DIR / image_rel
        mask_path = (OUT_DIR / mask_rel) if mask_rel is not None else None
        anno_path = OUT_DIR / anno_rel

        image_written = safe_write_image(image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        mask_written = False
        if SAVE_MASK_IMAGE and mask is not None:
            mask_written = safe_write_image(mask_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, int(PNG_COMPRESSION)])
        elif not SAVE_MASK_IMAGE:
            mask_written = True

        if REQUIRE_COMPLETE_SAMPLE_TRIPLET and (not image_written or not mask_written or mask_rel is None):
            cleanup_partial_sample_files(image_path, mask_path, anno_path)
            print(
                f"[WARN] incomplete sample write skipped: "
                f"image_written={image_written}, mask_written={mask_written}, mask_rel={mask_rel}"
            )
            continue

        if not image_written:
            cleanup_partial_sample_files(image_path, mask_path, anno_path)
            continue

        anno = {
            "image": image_rel,
            "mask": mask_rel,
            "width": int(width),
            "height": int(height),

            "dataset_type": "grouped_fixed_TEC",
            "group_index": int(group_index),
            "extrinsic_group_id": f"g{group_index:03d}",
            "image_index_in_group": int(image_index_in_group),
            "global_index": int(global_index),

            # 该组固定 T_EC，单位 mm
            "T_EC": matrix_to_list(T_EC_group_mm),
            "T_EC_unit": "mm_translation_rotation_unitless",

            # 实际读取到的 T_EC，用于检查
            "T_EC_actual": matrix_to_list(T_EC_actual_mm),
            "T_EC_actual_unit": "mm_translation_rotation_unitless",
            "max_abs_TEC_difference_mm": max_abs_TEC_diff_mm,

            # meter debug
            "T_EC_m_debug": matrix_to_list(T_EC_group_m),
            "T_EC_actual_m_debug": matrix_to_list(T_EC_actual_m),

            # 当前末端位姿信息
            "base_T_BE_m": matrix_to_list(T_BE_base_m),
            "target_T_BE_m": matrix_to_list(T_BE_target_m),
            "actual_T_BE_m": matrix_to_list(T_BE_actual_m),
            "actual_T_BC_m": matrix_to_list(T_BC_actual_m),

            "delta_ee_translation_mm": [float(x * 1000.0) for x in delta_ee_t_m],
            "delta_ee_rpy_deg": [float(x) for x in delta_ee_rpy_deg],
            "ee_translation_half_range_mm": float(EE_TRANS_HALF_RANGE_M * 1000.0),
            "ee_rotation_half_range_deg": float(EE_ROT_HALF_RANGE_DEG),
            "ee_pos_error_mm": ee_pos_error_mm,
            "ee_rot_error_deg": ee_rot_error_deg,

            # 当前关节角
            "joint_positions_rad": [float(x) for x in q_sol],
            "initial_joint_positions_rad": [float(x) for x in q0],

            # 图像统计
            "image_min": int(rgb.min()),
            "image_max": int(rgb.max()),
            "image_mean": float(rgb.mean()),
            "mask_ok": bool(mask_ok),
            "mask_mean": float(mask.mean()) if mask is not None else None,
            "mask_area_ratio": float(mask_area_ratio),

            "camera_path": CAMERA_PATH,
            "mask_camera_path": MASK_CAMERA_PATH,
            "end_effector_path": EE_PATH,
            "end_effector_root_path": END_EFFECTOR_ROOT_PATH,
            "definition": "Each group has one fixed T_EC and multiple robot poses.",

            "image_flip": {
                "flip_up_down": bool(FLIP_UP_DOWN),
                "flip_left_right": bool(FLIP_LEFT_RIGHT),
                "note": "RGB image and mask are flipped after reading from CoppeliaSim Vision Sensor."
            },

            "gripper_status": "open" if OPEN_GRIPPER_FIRST else "static_EG2_4CX_step",
            "gripper_open_status": gripper_open_status,

            "domain_randomization": {
                "ee_black_material_enabled": bool(SET_EE_BLACK_MATERIAL),
                "ee_material_mode": "whole_gripper_black_per_image",
                "ee_material": ee_color_records,
                "mask_read_ee_material": mask_color_records,
            },
        }

        try:
            write_json(anno_path, anno)
        except Exception:
            cleanup_partial_sample_files(image_path, mask_path, anno_path)
            continue

        if not anno_path.exists() or anno_path.stat().st_size <= 0:
            cleanup_partial_sample_files(image_path, mask_path, anno_path)
            continue

        return True

    return False


# ============================================================
# 8. 主程序
# ============================================================

def configure():
    parser = argparse.ArgumentParser(description="Collect grouped RGB/mask/T_EC samples in CoppeliaSim.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--images-per-group", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.groups < 1 or args.images_per_group < 1:
        parser.error("groups and images-per-group must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty to avoid overwriting samples")
    cfg = json.loads(args.scene_config.read_text(encoding="utf-8"))
    allowed = {"CAMERA_PATH", "MASK_CAMERA_PATH", "EE_PATH", "UR5_PATH", "END_EFFECTOR_ROOT_PATH",
               "FLIP_UP_DOWN", "FLIP_LEFT_RIGHT", "TEC_TRANS_HALF_RANGE_M", "TEC_ROT_HALF_RANGE_DEG",
               "EE_TRANS_HALF_RANGE_M", "EE_ROT_HALF_RANGE_DEG"}
    if set(cfg) - allowed:
        parser.error("Unknown scene configuration keys: " + str(sorted(set(cfg) - allowed)))
    globals().update(cfg)
    globals().update(OUT_DIR=args.output, NUM_TEC_GROUPS=args.groups, IMAGES_PER_GROUP=args.images_per_group)
    np.random.seed(args.seed)


def main():
    configure()
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    ensure_dirs()

    client = RemoteAPIClient()
    sim = client.require("sim")

    # 确保仿真停止，避免机械臂动力学乱动
    try:
        sim.stopSimulation()
        time.sleep(1.0)
    except Exception:
        pass

    camera = sim.getObject(CAMERA_PATH)
    mask_camera = sim.getObject(MASK_CAMERA_PATH) if SAVE_MASK_IMAGE else None
    ee = sim.getObject(EE_PATH)
    base = sim.getObject(UR5_PATH)

    if mask_camera is not None:
        print(f"Using mask camera: {MASK_CAMERA_PATH}")

    ee_shape_handles = get_end_effector_shape_handles(sim)
    print(f"Using end-effector root: {END_EFFECTOR_ROOT_PATH}")
    print(f"End-effector shape count: {len(ee_shape_handles)}")

    arm_joints = get_ur5_arm_joints(sim)
    q0 = get_joint_positions(sim, arm_joints)

    print("Initial UR5 joint positions:")
    print(q0)

    # Vision Sensor 显式处理
    try:
        sim.setExplicitHandling(camera, 1)
        print("setExplicitHandling(camera, 1) success")
    except Exception as e:
        print("setExplicitHandling failed:", e)

    if mask_camera is not None:
        try:
            sim.setExplicitHandling(mask_camera, 1)
            print("setExplicitHandling(mask_camera, 1) success")
        except Exception as e:
            print("setExplicitHandling mask camera failed:", e)

        if AUTO_SETUP_MASK_CAMERA_RENDER_ONLY_EE:
            try:
                setup_mask_camera_render_only_end_effector(sim, mask_camera)
            except Exception as e:
                raise RuntimeError("setup_mask_camera_render_only_end_effector failed") from e

    # 可选：打开夹爪
    if OPEN_GRIPPER_FIRST:
        gripper_open_status = open_rg2_safely(sim)
    else:
        gripper_open_status = {}

    # 当前相机相对末端的 T_EC 作为所有组的随机中心
    pose_EC_base = sim.getObjectPose(camera, ee)
    T_EC_base_m = pose_to_T(pose_EC_base)
    if mask_camera is not None:
        set_object_pose(sim, mask_camera, ee, pose_EC_base)

    # 当前末端位姿作为机械臂末端采样中心
    T_BE_base_m = get_ee_T_base(sim, ee, base)

    print("\nBase T_EC, unit=m:")
    print(T_EC_base_m)

    print("\nBase T_EC, translation unit=mm:")
    print(T_m_to_T_mm(T_EC_base_m))

    print("\nBase end-effector pose T_BE, unit=m:")
    print(T_BE_base_m)

    total_expected = NUM_TEC_GROUPS * IMAGES_PER_GROUP

    print("\nStart collecting grouped fixed-T_EC dataset...")
    print(f"Groups: {NUM_TEC_GROUPS}")
    print(f"Images per group: {IMAGES_PER_GROUP}")
    print(f"Total expected: {total_expected}")
    print("Output directory:", OUT_DIR.resolve())

    group_records = []
    global_index = 0

    pbar = tqdm(total=total_expected, desc="all_images")

    for group_idx in range(NUM_TEC_GROUPS):
        # 每组先恢复到初始关节位姿，避免 IK 越走越远
        set_joint_positions(sim, arm_joints, q0)

        # 1. 为当前组随机一个 T_EC
        T_EC_group_m, delta_tec_t_m, delta_tec_rpy_deg = sample_group_TEC(T_EC_base_m)
        T_EC_group_mm = T_m_to_T_mm(T_EC_group_m)
        pose_EC_group = T_to_pose(T_EC_group_m)

        # 2. 设置该组 T_EC
        set_object_pose(sim, camera, ee, pose_EC_group)
        if mask_camera is not None:
            set_object_pose(sim, mask_camera, ee, pose_EC_group)

        # 3. 保存 group 级别信息
        group_info = {
            "group_index": int(group_idx),
            "num_images": int(IMAGES_PER_GROUP),
            "T_EC_group": matrix_to_list(T_EC_group_mm),
            "T_EC_group_unit": "mm_translation_rotation_unitless",
            "T_EC_group_m_debug": matrix_to_list(T_EC_group_m),
            "delta_TEC_translation_mm": [float(x * 1000.0) for x in delta_tec_t_m],
            "delta_TEC_rpy_deg": [float(x) for x in delta_tec_rpy_deg],
            "TEC_translation_half_range_mm": float(TEC_TRANS_HALF_RANGE_M * 1000.0),
            "TEC_rotation_half_range_deg": float(TEC_ROT_HALF_RANGE_DEG),
        }

        write_json(OUT_DIR / "groups" / f"group_{group_idx:03d}.json", group_info)

        group_records.append(group_info)

        # 4. 当前组内采 15 张，T_EC 不变，机械臂末端变化
        saved_in_group = 0

        while saved_in_group < IMAGES_PER_GROUP:
            ok = collect_one_image(
                sim=sim,
                camera=camera,
                mask_camera=mask_camera,
                ee=ee,
                base=base,
                arm_joints=arm_joints,
                q0=q0,
                ee_shape_handles=ee_shape_handles,
                T_BE_base_m=T_BE_base_m,
                T_EC_group_m=T_EC_group_m,
                T_EC_group_mm=T_EC_group_mm,
                pose_EC_group=pose_EC_group,
                group_index=group_idx,
                image_index_in_group=saved_in_group,
                global_index=global_index,
                gripper_open_status=gripper_open_status,
            )

            if ok:
                saved_in_group += 1
                global_index += 1
                pbar.update(1)
            else:
                print(f"[WARN] group {group_idx} failed to collect image {saved_in_group}")
                print("Try reducing EE_TRANS_HALF_RANGE_M or EE_ROT_HALF_RANGE_DEG.")
                break

    pbar.close()

    # 恢复初始状态
    if RESTORE_INITIAL_STATE:
        set_joint_positions(sim, arm_joints, q0)
        set_object_pose(sim, camera, ee, pose_EC_base)
        if mask_camera is not None:
            set_object_pose(sim, mask_camera, ee, pose_EC_base)

    triplet_audit = audit_dataset_triplets(OUT_DIR) if SAVE_MASK_IMAGE else None

    summary = {
        "total_saved": int(global_index),
        "total_expected": int(total_expected),
        "num_TEC_groups": int(NUM_TEC_GROUPS),
        "images_per_group": int(IMAGES_PER_GROUP),
        "output_dir": str(OUT_DIR.resolve()),

        "camera_path": CAMERA_PATH,
        "mask_camera_path": MASK_CAMERA_PATH,
        "end_effector_path": EE_PATH,
        "end_effector_root_path": END_EFFECTOR_ROOT_PATH,

        "image_flip": {
            "flip_up_down": bool(FLIP_UP_DOWN),
            "flip_left_right": bool(FLIP_LEFT_RIGHT),
            "note": "RGB image is flipped after reading from CoppeliaSim Vision Sensor."
        },

        "domain_randomization": {
            "ee_black_material_enabled": bool(SET_EE_BLACK_MATERIAL),
            "ee_material_mode": "whole_gripper_black_per_image",
            "end_effector_shape_count": int(len(ee_shape_handles)),
        },

        "mask": {
            "save_mask_image": bool(SAVE_MASK_IMAGE),
            "require_complete_sample_triplet": bool(REQUIRE_COMPLETE_SAMPLE_TRIPLET),
            "set_ee_white_material_for_mask_read": bool(SET_EE_WHITE_MATERIAL_FOR_MASK_READ),
            "mask_read_retries": int(MASK_READ_RETRIES),
            "mask_threshold": int(MASK_THRESHOLD),
            "mask_morphology_enabled": bool(MASK_MORPHOLOGY_ENABLED),
            "auto_setup_mask_camera_render_only_ee": bool(AUTO_SETUP_MASK_CAMERA_RENDER_ONLY_EE),
            "min_mask_area_ratio": float(MIN_MASK_AREA_RATIO),
            "png_compression": int(PNG_COMPRESSION),
        },

        "T_EC_base_mm": matrix_to_list(T_m_to_T_mm(T_EC_base_m)),
        "T_EC_base_m": matrix_to_list(T_EC_base_m),

        "TEC_translation_half_range_mm": float(TEC_TRANS_HALF_RANGE_M * 1000.0),
        "TEC_rotation_half_range_deg": float(TEC_ROT_HALF_RANGE_DEG),

        "base_T_BE_m": matrix_to_list(T_BE_base_m),
        "ee_translation_half_range_mm": float(EE_TRANS_HALF_RANGE_M * 1000.0),
        "ee_rotation_half_range_deg": float(EE_ROT_HALF_RANGE_DEG),

        "ik_max_iters": int(IK_MAX_ITERS),
        "ik_pos_tol_mm": float(IK_POS_TOL_M * 1000.0),
        "ik_rot_tol_deg": float(np.rad2deg(IK_ROT_TOL_RAD)),

        "groups": group_records,
        "triplet_audit": triplet_audit,
    }

    write_json(OUT_DIR / "dataset_summary.json", summary)

    print("\nDone.")
    print("Saved:", global_index)
    print("Expected:", total_expected)
    print("Saved images:", OUT_DIR / "rgb")
    if SAVE_MASK_IMAGE:
        print("Saved masks :", OUT_DIR / "mask")
    print("Saved jsons :", OUT_DIR / "anno")
    print("Group jsons :", OUT_DIR / "groups")
    print("Summary    :", OUT_DIR / "dataset_summary.json")
    if triplet_audit is not None:
        print("Triplet audit ok:", triplet_audit["ok"])

    if global_index < total_expected:
        print("\n[WARNING] 没有采满所有图片。")
        print("建议先调小：")
        print("EE_TRANS_HALF_RANGE_M = 0.005")
        print("EE_ROT_HALF_RANGE_DEG = 4.0")
        print("或者增大 MAX_ATTEMPTS_PER_IMAGE。")

    if triplet_audit is not None and not triplet_audit["ok"]:
        print("\n[WARNING] RGB/mask/anno 三件套不完整，请先检查 dataset_summary.json。")


if __name__ == "__main__":
    main()
