import os
import cv2
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO

class CameraMotionCompensator:
    """
    基于背景特征光流法的极速相机全局运动补偿器 (Global Motion Compensation - GMC)
    """
    def __init__(self, max_corners=80, quality_level=0.02, min_distance=15, motion_deadband=0.5, calc_w=360):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.motion_deadband = motion_deadband # 零漂死区门槛 (像素)
        self.calc_w = calc_w                   # 光流极速计算尺寸
        self.prev_gray_small = None
        self.last_bg_points_orig = []

    def estimate_camera_motion(self, frame, person_box=None):
        h_orig, w_orig = frame.shape[:2]
        scale = self.calc_w / float(w_orig)
        calc_h = int(h_orig * scale)

        small_frame = cv2.resize(frame, (self.calc_w, calc_h), interpolation=cv2.INTER_NEAREST)
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray_small is None:
            self.prev_gray_small = gray_small
            return 0.0, 0.0

        mask = np.ones_like(gray_small, dtype=np.uint8) * 255
        if person_box is not None:
            x1, y1, x2, y2 = map(int, person_box)
            sx1, sy1 = int(max(0, x1 * scale - 6)), int(max(0, y1 * scale - 6))
            sx2, sy2 = int(min(self.calc_w, x2 * scale + 6)), int(min(calc_h, y2 * scale + 6))
            mask[sy1:sy2, sx1:sx2] = 0

        p0 = cv2.goodFeaturesToTrack(self.prev_gray_small, mask=mask, maxCorners=self.max_corners, 
                                     qualityLevel=self.quality_level, minDistance=self.min_distance)

        dx, dy = 0.0, 0.0
        self.last_bg_points_orig = []

        if p0 is not None and len(p0) >= 6:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray_small, gray_small, p0, None)
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) >= 6:
                disp_small = good_new - good_old
                med_disp = np.median(disp_small, axis=0)
                inliers = np.abs(disp_small - med_disp) < 3.0
                valid = inliers[:, 0] & inliers[:, 1]
                
                if np.sum(valid) >= 4:
                    dx_small, dy_small = np.mean(disp_small[valid], axis=0)
                    dx = dx_small / scale
                    dy = dy_small / scale
                    self.last_bg_points_orig = good_new[valid] / scale

        self.prev_gray_small = gray_small

        if np.sqrt(dx**2 + dy**2) < self.motion_deadband:
            dx, dy = 0.0, 0.0

        return float(dx), float(dy)


class RobustPoseExtractor:
    """
    步态姿态提取器 (终极加速版：自动硬件加速 + 动态帧采样 + 前置分辨率压制)
    """
    COCO_SKELETON = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (1, 2), (0, 1), (0, 2), (1, 3), (2, 4)
    ]

    def __init__(self, model_path='yolov8n-pose.pt', conf_thresh=None, init_conf_thresh=0.55, 
                 track_conf_thresh=0.25, smooth_factor=0.4, max_jump_thresh=0.30, 
                 enable_gmc=True, frame_stride=2, **kwargs):
        print("🚀 初始化【一步到位·极速硬件加速版】姿态提取器...")
        self.model = YOLO(model_path)
        
        # 自动检测 GPU 硬件加速
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            print("⚡ 已检测到 NVIDIA GPU！成功激活 CUDA 硬件加速引擎。")
        else:
            print("💻 当前运行于 CPU 模式，已启用轻量化 CPU 矩阵优化。")

        if conf_thresh is not None:
            self.init_conf_thresh = max(0.50, conf_thresh)
            self.track_conf_thresh = min(0.25, conf_thresh)
        else:
            self.init_conf_thresh = init_conf_thresh
            self.track_conf_thresh = track_conf_thresh
            
        self.smooth_factor = smooth_factor
        self.max_jump_thresh = max_jump_thresh
        self.enable_gmc = enable_gmc
        self.frame_stride = max(1, int(frame_stride)) # 帧采样步长 (默认 2: 30FPS -> 15FPS 采样)
        
        self.prev_centroid = None
        self.prev_box = None
        self.prev_kpts = None
        self.lost_counter = 0

    def _compute_box_iou(self, box1, box2):
        xa = max(box1[0], box2[0])
        ya = max(box1[1], box2[1])
        xb = min(box1[2], box2[2])
        yb = min(box1[3], box2[3])
        
        inter = max(0, xb - xa) * max(0, yb - ya)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter + 1e-6
        return inter / union

    def _validate_walking_gait_geometry(self, kpts, confs):
        xs = kpts[:, 0]
        ys = kpts[:, 1]
        box_w = float(np.max(xs) - np.min(xs))
        box_h = float(np.max(ys) - np.min(ys))

        ls_x, ls_y = kpts[5, 0], kpts[5, 1]
        rs_x, rs_y = kpts[6, 0], kpts[6, 1]
        lh_x, lh_y = kpts[11, 0], kpts[11, 1]
        rh_x, rh_y = kpts[12, 0], kpts[12, 1]
        sh_y = (ls_y + rs_y) / 2.0
        hip_y = (lh_y + rh_y) / 2.0
        torso_h = float(np.sqrt(((ls_x+rs_x)/2.0 - (lh_x+rh_x)/2.0)**2 + (sh_y - hip_y)**2))
        head_y = float(kpts[0, 1])

        face_conf = float(np.mean(confs[0:5]))
        core_conf = float(np.mean([confs[5], confs[6], confs[11], confs[12]]))

        is_fall_collapse = (head_y >= hip_y - 0.02) or (box_w > box_h * 1.25 and torso_h < 0.08)
        if is_fall_collapse:
            return False, torso_h, core_conf, face_conf

        is_static_artifact = (face_conf < 0.25) or (torso_h < 0.075) or (box_h < 0.18)
        if is_static_artifact:
            return False, torso_h, core_conf, face_conf

        return True, torso_h, core_conf, face_conf

    def _filter_and_track_person(self, boxes, kpts_array, kpts_conf_array, img_w, img_h):
        valid_candidates = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            kpts = kpts_array[i]
            confs = kpts_conf_array[i]

            is_valid, torso_h, core_conf, face_conf = self._validate_walking_gait_geometry(kpts, confs)
            if not is_valid:
                continue

            bx1, by1, bx2, by2 = x1/img_w, y1/img_h, x2/img_w, y2/img_h
            norm_box = np.array([bx1, by1, bx2, by2])
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            centroid = np.array([cx, cy])

            valid_candidates.append({
                'idx': i,
                'box': norm_box,
                'pixel_box': [x1, y1, x2, y2],
                'centroid': centroid,
                'core_conf': core_conf,
                'face_conf': face_conf,
                'kpts': kpts,
                'confs': confs
            })

        if not valid_candidates:
            self.lost_counter += 1
            if self.lost_counter >= 3:
                self.prev_centroid = None
                self.prev_box = None
            return None

        if self.prev_centroid is None:
            high_quality = [c for c in valid_candidates if (c['core_conf'] >= self.init_conf_thresh) and (c['face_conf'] >= 0.35)]
            if not high_quality:
                return None
            best_init = max(high_quality, key=lambda x: x['core_conf'])
            self.prev_centroid = best_init['centroid']
            self.prev_box = best_init['box']
            self.lost_counter = 0
            return best_init

        candidates_with_score = []
        for c in valid_candidates:
            dist = np.linalg.norm(c['centroid'] - self.prev_centroid)
            iou = self._compute_box_iou(c['box'], self.prev_box)
            if (iou > 0.05 or dist <= self.max_jump_thresh) and (c['core_conf'] >= self.track_conf_thresh):
                candidates_with_score.append((c, dist, iou))

        if candidates_with_score:
            candidates_with_score.sort(key=lambda x: (-x[2], x[1]))
            selected = candidates_with_score[0][0]
            self.prev_centroid = selected['centroid']
            self.prev_box = selected['box']
            self.lost_counter = 0
            return selected

        high_conf_reid = [c for c in valid_candidates if c['core_conf'] >= 0.60 and c['face_conf'] >= 0.35]
        if high_conf_reid:
            best_reid = max(high_conf_reid, key=lambda x: x['core_conf'])
            self.prev_centroid = best_reid['centroid']
            self.prev_box = best_reid['box']
            self.lost_counter = 0
            return best_reid

        self.lost_counter += 1
        if self.lost_counter >= 3:
            self.prev_centroid = None
            self.prev_box = None
        return None

    def _draw_hud_overlay(self, frame, gmc, cam_dx, cam_dy, accum_dx, accum_dy, kpts_pixel, confs):
        """绘制 HUD 面板"""
        h, w = frame.shape[:2]

        if gmc is not None and len(gmc.last_bg_points_orig) > 0:
            for pt in gmc.last_bg_points_orig:
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

        cx, cy = w // 2, h // 2
        arrow_dx = int(cam_dx * 5.0)
        arrow_dy = int(cam_dy * 5.0)
        if abs(arrow_dx) > 1 or abs(arrow_dy) > 1:
            cv2.arrowedLine(frame, (cx, cy), (cx + arrow_dx, cy + arrow_dy), (0, 0, 255), 3, tipLength=0.3)
            cv2.putText(frame, "CAM MOTION VECTOR", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if kpts_pixel is not None:
            for p1_idx, p2_idx in self.COCO_SKELETON:
                if confs[p1_idx] > 0.25 and confs[p2_idx] > 0.25:
                    pt1 = (int(kpts_pixel[p1_idx, 0]), int(kpts_pixel[p1_idx, 1]))
                    pt2 = (int(kpts_pixel[p2_idx, 0]), int(kpts_pixel[p2_idx, 1]))
                    cv2.line(frame, pt1, pt2, (255, 200, 0), 2)

            for i in range(17):
                if confs[i] > 0.25:
                    pt = (int(kpts_pixel[i, 0]), int(kpts_pixel[i, 1]))
                    cv2.circle(frame, pt, 4, (0, 165, 255), -1)

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (360, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        status_text = "GMC Anti-Shake: ACTIVE" if self.enable_gmc else "GMC: DISABLED"
        color_status = (0, 255, 0) if self.enable_gmc else (0, 0, 255)
        
        cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)
        cv2.putText(frame, f"Cam Shift (Frame): dX={cam_dx:+.1f}px, dY={cam_dy:+.1f}px", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"Accum Shift (Total): X={accum_dx*w:+.1f}px, Y={accum_dy*h:+.1f}px", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 255), 1)

    def extract(self, video_path, output_csv_path, output_video_path=None, show_preview=False):
        out_csv_dir = os.path.dirname(output_csv_path)
        if out_csv_dir:
            os.makedirs(out_csv_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频文件: {video_path}")

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_fps = raw_fps / float(self.frame_stride)
            writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (w, h))

        need_rendering = (writer is not None) or show_preview

        frame_idx = 0
        records = []
        self.prev_centroid = None
        self.prev_box = None
        self.prev_kpts = None
        self.lost_counter = 0

        gmc = CameraMotionCompensator() if self.enable_gmc else None
        accum_cam_dx, accum_cam_dy = 0.0, 0.0

        print(f"🎬 开始一步到位极速姿态提取 [帧采样 Stride={self.frame_stride}]: {video_path}...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if (frame_idx % self.frame_stride) != 0:
                continue

            if not need_rendering and w > 640:
                proc_frame = cv2.resize(frame, (640, int(h * 640.0 / w)), interpolation=cv2.INTER_NEAREST)
            else:
                proc_frame = frame

            # YOLO 姿态推理
            proc_h, proc_w = proc_frame.shape[:2]

            results = self.model(
                proc_frame, 
                verbose=False, 
                imgsz=480 if self.device == 'cpu' else 640, 
                device=self.device, 
                classes=0, 
                conf=0.25
            )[0]

            selected_target = None
            if len(results.boxes) > 0 and results.keypoints is not None:
                boxes = results.boxes
                kpts_xyn = results.keypoints.xyn.cpu().numpy()
                kpts_conf = results.keypoints.conf.cpu().numpy() if results.keypoints.conf is not None else np.ones((len(boxes), 17))
                
                # 👈 注意：这里传入的是 proc_w 和 proc_h，而不是原始视频的 w 和 h！
                selected_target = self._filter_and_track_person(boxes, kpts_xyn, kpts_conf, proc_w, proc_h)

            cam_dx_px, cam_dy_px = 0.0, 0.0
            person_pixel_box = selected_target['pixel_box'] if selected_target is not None else None
            if gmc is not None:
                cam_dx_px, cam_dy_px = gmc.estimate_camera_motion(proc_frame, person_box=person_pixel_box)
                accum_cam_dx += cam_dx_px / proc_w
                accum_cam_dy += cam_dy_px / proc_h

            if selected_target is not None:
                curr_kpts = selected_target['kpts'].copy()
                confs = selected_target['confs']

                if self.prev_kpts is not None:
                    curr_kpts = self.smooth_factor * curr_kpts + (1 - self.smooth_factor) * self.prev_kpts
                self.prev_kpts = curr_kpts.copy()

                stabilized_kpts = curr_kpts.copy()
                if self.enable_gmc:
                    stabilized_kpts[:, 0] -= accum_cam_dx
                    stabilized_kpts[:, 1] -= accum_cam_dy

                frame_data = {'frame': frame_idx, 'detected': 1}
                for idx in range(17):
                    frame_data[f'kpt_{idx}_x'] = stabilized_kpts[idx, 0]
                    frame_data[f'kpt_{idx}_y'] = stabilized_kpts[idx, 1]
                    frame_data[f'kpt_{idx}_vis'] = confs[idx]

                records.append(frame_data)

                if need_rendering:
                    annotated_frame = frame.copy()
                    kpts_pixel = curr_kpts.copy()
                    kpts_pixel[:, 0] *= w
                    kpts_pixel[:, 1] *= h
                    # 修正变量名为 accum_cam_dy
                    self._draw_hud_overlay(annotated_frame, gmc, cam_dx_px, cam_dy_px, accum_cam_dx, accum_cam_dy, kpts_pixel, confs)
            else:
                self.prev_kpts = None
                frame_data = {'frame': frame_idx, 'detected': 0}
                for idx in range(17):
                    frame_data[f'kpt_{idx}_x'] = np.nan
                    frame_data[f'kpt_{idx}_y'] = np.nan
                    frame_data[f'kpt_{idx}_vis'] = np.nan
                records.append(frame_data)

                if need_rendering:
                    annotated_frame = frame.copy()
                    # 修正变量名为 accum_cam_dy
                    self._draw_hud_overlay(annotated_frame, gmc, cam_dx_px, cam_dy_px, accum_cam_dx, accum_cam_dy, None, None)

            if need_rendering and writer:
                writer.write(annotated_frame)

            if show_preview:
                cv2.imshow("GMC Anti-Shake Gait Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        df = pd.DataFrame(records)
        df.to_csv(output_csv_path, index=False)
        print(f"✅ 极速处理完成！姿态 CSV 已导出: {output_csv_path}")
        return df

if __name__ == '__main__':
    extractor = RobustPoseExtractor(conf_thresh=0.35, smooth_factor=0.4, enable_gmc=True, frame_stride=2)
    extractor.extract("data/sample/cerebralPalsy_2.mp4", "data/sample/cerebralPalsy_2_pose.csv")