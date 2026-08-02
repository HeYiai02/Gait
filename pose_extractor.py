# 导入所需的“工具箱”
import os          # os: 用来处理电脑系统里的文件和文件夹路径
import cv2         # cv2 (OpenCV): 视觉处理神器，用来读取视频、处理画面、画图
import pandas as pd # pandas: 数据处理工具，用来把最后的结果保存成像 Excel 一样的表格 (CSV)
import numpy as np  # numpy: 数学计算工具，专门用来快速处理大量数字和矩阵[cite: 2]
import torch        # torch: 人工智能框架，用来调用显卡 (GPU) 加速计算[cite: 2]
from ultralytics import YOLO # YOLO: 我们请来的“AI 视觉专家”，专门用来识别画面里的人和骨架[cite: 2]

class CameraMotionCompensator:
    """
    防抖摄影师：基于背景特征光流法的相机全局运动补偿器 (GMC)
    它的工作是：盯着背景里不动的点，看它们移动了多少，从而算出镜头抖动了多少。
    """
    def __init__(self, max_corners=80, quality_level=0.02, min_distance=15, motion_deadband=0.5, calc_w=360):
        # 设定摄影师的工作参数
        self.max_corners = max_corners           # 最多找 80 个背景参考点[cite: 2]
        self.quality_level = quality_level       # 参考点的质量要求（越低越容易找，但可能不准）[cite: 2]
        self.min_distance = min_distance         # 两个参考点之间至少隔开 15 个像素，别扎堆[cite: 2]
        self.motion_deadband = motion_deadband   # 零漂死区：如果抖动小于 0.5 像素，就当做没抖[cite: 2]
        self.calc_w = calc_w                     # 为了算得快，把画面缩小到 360 像素宽来计算[cite: 2]
        
        self.prev_gray_small = None              # 记住上一帧的黑白缩小画面[cite: 2]
        self.last_bg_points_orig = []            # 记住上一帧找到的背景点[cite: 2]

    def estimate_camera_motion(self, frame, person_box=None):
        # 拿到当前画面的原始高度和宽度
        h_orig, w_orig = frame.shape[:2]
        # 计算缩小比例（比如 1920 宽缩小到 360，比例就是 360/1920）
        scale = self.calc_w / float(w_orig)
        calc_h = int(h_orig * scale) # 算出缩小后的高度

        # 把画面缩小，并变成黑白（灰度图），因为电脑算黑白小图最快！[cite: 2]
        small_frame = cv2.resize(frame, (self.calc_w, calc_h), interpolation=cv2.INTER_NEAREST)
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # 如果这是视频的第一帧，没有“上一帧”可以对比，那就直接返回抖动为 0[cite: 2]
        if self.prev_gray_small is None:
            self.prev_gray_small = gray_small
            return 0.0, 0.0

        # 准备一张白纸（mask），白色的地方代表“可以找参考点”[cite: 2]
        mask = np.ones_like(gray_small, dtype=np.uint8) * 255

        # 如果画面里有人 (person_box 存在)
        if person_box is not None:
            # 把人的边框坐标拿出来[cite: 2]
            x1, y1, x2, y2 = map(int, person_box)
            # 把坐标也按比例缩小，并稍微往外扩一点点（+-6），确保把人完全包住[cite: 2]
            sx1, sy1 = int(max(0, x1 * scale - 6)), int(max(0, y1 * scale - 6))
            sx2, sy2 = int(min(self.calc_w, x2 * scale + 6)), int(min(calc_h, y2 * scale + 6))
            # 把人所在的位置在“白纸”上涂黑（0代表黑）。意思是：别在人身上找背景点，人是会动的！[cite: 2]
            mask[sy1:sy2, sx1:sx2] = 0

        # 在上一帧的画面里，只在白纸（mask）允许的范围内，找适合追踪的角点（特征点）[cite: 2]
        p0 = cv2.goodFeaturesToTrack(self.prev_gray_small, mask=mask, maxCorners=self.max_corners, 
                                     qualityLevel=self.quality_level, minDistance=self.min_distance)

        dx, dy = 0.0, 0.0
        self.last_bg_points_orig = []

        # 如果找到了至少 6 个点，我们就可以开始对比了[cite: 2]
        if p0 is not None and len(p0) >= 6:
            # 光流法核心：看这批点 (p0) 在当前帧 (gray_small) 跑到哪里去了 (p1)[cite: 2]
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray_small, gray_small, p0, None)
            
            # 只保留那些成功追踪到的点 (st == 1 代表追踪成功)[cite: 2]
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # 再次确认成功追踪的点不少于 6 个
            if len(good_new) >= 6:
                # 计算每个点横向和纵向移动了多少距离[cite: 2]
                disp_small = good_new - good_old
                # 找这些点移动距离的中位数（排除那些乱跑的错误点）[cite: 2]
                med_disp = np.median(disp_small, axis=0)
                
                # 过滤：如果某个点移动的距离跟大部队（中位数）差得太远（>3），就认为它是坏点[cite: 2]
                inliers = np.abs(disp_small - med_disp) < 3.0
                valid = inliers[:, 0] & inliers[:, 1]
                
                # 如果剩下的好点还有 4 个以上[cite: 2]
                if np.sum(valid) >= 4:
                    # 算出小图上的平均位移[cite: 2]
                    dx_small, dy_small = np.mean(disp_small[valid], axis=0)
                    # 因为我们是在缩小图上算的，现在要把位移按比例放大回原图的尺寸[cite: 2]
                    dx = dx_small / scale
                    dy = dy_small / scale
                    # 记录下这些有效的背景点，一会儿画图用[cite: 2]
                    self.last_bg_points_orig = good_new[valid] / scale

        # 把当前帧存下来，给下一帧当“上一帧”用[cite: 2]
        self.prev_gray_small = gray_small

        # 如果总的移动距离太小（没超过死区），干脆就当没动，防止数据一直微小波动[cite: 2]
        if np.sqrt(dx**2 + dy**2) < self.motion_deadband:
            dx, dy = 0.0, 0.0

        return float(dx), float(dy)


class RobustPoseExtractor:
    """
    AI 追踪员：步态姿态提取器
    它的工作是：用 AI 模型找出人，过滤掉假人/错误识别，并输出关节数据。
    """
    # 定义人体骨架连接的线段（比如 15连13，就是把左眼和左耳连起来）[cite: 2]
    COCO_SKELETON = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (1, 2), (0, 1), (0, 2), (1, 3), (2, 4)
    ]

    def __init__(self, model_path='yolov8n-pose.pt', conf_thresh=None, init_conf_thresh=0.55, 
                 track_conf_thresh=0.25, smooth_factor=0.4, max_jump_thresh=0.30, 
                 enable_gmc=True, frame_stride=2, **kwargs):
        print("🚀 初始化【一步到位·极速硬件加速版】姿态提取器...")
        # 把 YOLO 专家请出来（加载 AI 模型）[cite: 2]
        self.model = YOLO(model_path)
        
        # 看看电脑有没有独立的 N 卡 GPU (显卡加速)，有就用 0 号显卡，没有就用 cpu[cite: 2]
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            print("⚡ 已检测到 NVIDIA GPU！成功激活 CUDA 硬件加速引擎。")
        else:
            print("💻 当前运行于 CPU 模式，已启用轻量化 CPU 矩阵优化。")

        # 设置一些门槛分数：第一次认准一个人需要 0.55 的把握，后面跟踪只要 0.25 把握就行[cite: 2]
        if conf_thresh is not None:
            self.init_conf_thresh = max(0.50, conf_thresh)
            self.track_conf_thresh = min(0.25, conf_thresh)
        else:
            self.init_conf_thresh = init_conf_thresh
            self.track_conf_thresh = track_conf_thresh
            
        self.smooth_factor = smooth_factor       # 平滑系数，防止骨架在两帧之间跳来跳去[cite: 2]
        self.max_jump_thresh = max_jump_thresh   # 一个人一帧里最多能移动多远，超过这个距离说明认错人了[cite: 2]
        self.enable_gmc = enable_gmc             # 是否开启刚才那个“防抖摄影师”[cite: 2]
        self.frame_stride = max(1, int(frame_stride)) # 每隔几帧看一次？=2 就是隔一帧看一次，能快一倍[cite: 2]
        
        # 记录上一帧追踪的人的特征，用来保证我们一直跟踪同一个人[cite: 2]
        self.prev_centroid = None
        self.prev_box = None
        self.prev_kpts = None
        self.lost_counter = 0 # 记录跟丢了几帧[cite: 2]

    def _compute_box_iou(self, box1, box2):
        # 算一下两个框重合度高不高。如果上一帧和这一帧的人框重合度很高，大概率是同一个人[cite: 2]
        xa, ya = max(box1[0], box2[0]), max(box1[1], box2[1])
        xb, yb = min(box1[2], box2[2]), min(box1[3], box2[3])
        
        inter = max(0, xb - xa) * max(0, yb - ya)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter + 1e-6
        return inter / union

    def _validate_walking_gait_geometry(self, kpts, confs):
        # 检查这个人是不是“正常站立/行走”的状态。过滤掉摔倒的人或者奇奇怪怪的误报[cite: 2]
        xs, ys = kpts[:, 0], kpts[:, 1]
        box_w, box_h = float(np.max(xs) - np.min(xs)), float(np.max(ys) - np.min(ys))

        # 取出肩膀(5,6)、臀部(11,12)、头部(0)的坐标[cite: 2]
        ls_y, rs_y = kpts[5, 1], kpts[6, 1]
        lh_y, rh_y = kpts[11, 1], kpts[12, 1]
        sh_y = (ls_y + rs_y) / 2.0
        hip_y = (lh_y + rh_y) / 2.0
        
        # 算出躯干的高度[cite: 2]
        torso_h = float(np.sqrt(((kpts[5, 0]+kpts[6, 0])/2.0 - (kpts[11, 0]+kpts[12, 0])/2.0)**2 + (sh_y - hip_y)**2))
        head_y = float(kpts[0, 1])

        # 算出脸部和核心躯干的“AI 确信度”[cite: 2]
        face_conf = float(np.mean(confs[0:5]))
        core_conf = float(np.mean([confs[5], confs[6], confs[11], confs[12]]))

        # 如果头比屁股还低，或者人趴在地上(宽大于高)，说明摔倒了，不要！返回 False[cite: 2]
        is_fall_collapse = (head_y >= hip_y - 0.02) or (box_w > box_h * 1.25 and torso_h < 0.08)
        if is_fall_collapse:
            return False, torso_h, core_conf, face_conf

        # 如果没有脸，或者太矮，说明可能把旁边的电线杆认成人类了，不要！[cite: 2]
        is_static_artifact = (face_conf < 0.25) or (torso_h < 0.075) or (box_h < 0.18)
        if is_static_artifact:
            return False, torso_h, core_conf, face_conf

        return True, torso_h, core_conf, face_conf

    def _filter_and_track_person(self, boxes, kpts_array, kpts_conf_array, img_w, img_h):
        # 这个函数负责在画面中所有的“人”里，挑出我们要跟踪的那个“主角”[cite: 2]
        valid_candidates = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            kpts, confs = kpts_array[i], kpts_conf_array[i]

            # 叫上面的函数来帮忙检查，姿势对不对[cite: 2]
            is_valid, torso_h, core_conf, face_conf = self._validate_walking_gait_geometry(kpts, confs)
            if not is_valid:
                continue

            # 把坐标转换成比例（0到1之间），方便计算[cite: 2]
            bx1, by1, bx2, by2 = x1/img_w, y1/img_h, x2/img_w, y2/img_h
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            
            valid_candidates.append({
                'idx': i, 'box': np.array([bx1, by1, bx2, by2]),
                'pixel_box': [x1, y1, x2, y2], 'centroid': np.array([cx, cy]),
                'core_conf': core_conf, 'face_conf': face_conf,
                'kpts': kpts, 'confs': confs
            })

        # 如果画面里连一个合格的候选人都没有[cite: 2]
        if not valid_candidates:
            self.lost_counter += 1
            if self.lost_counter >= 3: # 连续 3 帧都没看到，彻底跟丢了，清除记忆[cite: 2]
                self.prev_centroid, self.prev_box = None, None
            return None

        # 如果这是刚刚开始跟踪（还没有记忆）[cite: 2]
        if self.prev_centroid is None:
            # 找一个躯干和脸部都特别清晰的人作为主角[cite: 2]
            high_quality = [c for c in valid_candidates if (c['core_conf'] >= self.init_conf_thresh) and (c['face_conf'] >= 0.35)]
            if not high_quality: return None
            best_init = max(high_quality, key=lambda x: x['core_conf']) # 选最清晰的那个[cite: 2]
            self.prev_centroid, self.prev_box = best_init['centroid'], best_init['box']
            self.lost_counter = 0
            return best_init

        # 如果已经在跟踪了，就在候选人里找离上一帧位置最近的那个[cite: 2]
        candidates_with_score = []
        for c in valid_candidates:
            dist = np.linalg.norm(c['centroid'] - self.prev_centroid) # 算中心点距离[cite: 2]
            iou = self._compute_box_iou(c['box'], self.prev_box)      # 算框的重合度[cite: 2]
            if (iou > 0.05 or dist <= self.max_jump_thresh) and (c['core_conf'] >= self.track_conf_thresh):
                candidates_with_score.append((c, dist, iou))

        if candidates_with_score:
            # 排序：优先选重合度高的，再选距离近的[cite: 2]
            candidates_with_score.sort(key=lambda x: (-x[2], x[1]))
            selected = candidates_with_score[0][0]
            self.prev_centroid, self.prev_box = selected['centroid'], selected['box']
            self.lost_counter = 0
            return selected

        # 没找到匹配的，增加丢失计数[cite: 2]
        self.lost_counter += 1
        if self.lost_counter >= 3:
            self.prev_centroid, self.prev_box = None, None
        return None

        #给画画功能增加了 scale_w 和 scale_h 两个参数，用来把后台计算的小坐标放大回原图
    def _draw_hud_overlay(self, frame, gmc, cam_dx, cam_dy, accum_dx, accum_dy, kpts_pixel, confs, scale_w=1.0, scale_h=1.0):
        # 🎨 画画部分：在画面上加上科技感的数据显示面板[cite: 2]
        h, w = frame.shape[:2]

        # 1. 画出绿色的背景点[cite: 2]
        if gmc is not None and len(gmc.last_bg_points_orig) > 0:
            for pt in gmc.last_bg_points_orig:
                # 还原绿色的背景点到高清大图上
                px = int(pt[0] * scale_w)
                py = int(pt[1] * scale_h)
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

        # 2. 画面中间画个红色箭头，指示当前镜头在往哪晃[cite: 2]
        cx, cy = w // 2, h // 2
        arrow_dx, arrow_dy = int(cam_dx * 5.0), int(cam_dy * 5.0) # 放大 5 倍看得清楚[cite: 2]
        if abs(arrow_dx) > 1 or abs(arrow_dy) > 1:
            cv2.arrowedLine(frame, (cx, cy), (cx + arrow_dx, cy + arrow_dy), (0, 0, 255), 3, tipLength=0.3)

        # 3. 画出人体的蓝色线段和黄色关节[cite: 2]
        if kpts_pixel is not None:
            for p1_idx, p2_idx in self.COCO_SKELETON:
                if confs[p1_idx] > 0.25 and confs[p2_idx] > 0.25:
                    pt1 = (int(kpts_pixel[p1_idx, 0]), int(kpts_pixel[p1_idx, 1]))
                    pt2 = (int(kpts_pixel[p2_idx, 0]), int(kpts_pixel[p2_idx, 1]))
                    cv2.line(frame, pt1, pt2, (255, 200, 0), 2)
            for i in range(17):
                if confs[i] > 0.25:
                    cv2.circle(frame, (int(kpts_pixel[i, 0]), int(kpts_pixel[i, 1])), 4, (0, 165, 255), -1)

        # 4. 左上角画个半透明的黑框，写上当前的抖动数据[cite: 2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (360, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame) # 混合变成半透明效果[cite: 2]

        status_text = "GMC Anti-Shake: ACTIVE" if self.enable_gmc else "GMC: DISABLED"
        color_status = (0, 255, 0) if self.enable_gmc else (0, 0, 255)
        cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)
        cv2.putText(frame, f"Cam Shift (Frame): dX={cam_dx:+.1f}px, dY={cam_dy:+.1f}px", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"Accum Shift (Total): X={accum_dx*w:+.1f}px, Y={accum_dy*h:+.1f}px", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 255), 1)

    def extract(self, video_path, output_csv_path, output_video_path=None, show_preview=False):
        # 🎬 总导演：把上面所有的人组合起来开始干活！[cite: 2]
        out_csv_dir = os.path.dirname(output_csv_path)
        if out_csv_dir: os.makedirs(out_csv_dir, exist_ok=True) # 如果要存文件的文件夹不存在，就建一个[cite: 2]

        cap = cv2.VideoCapture(video_path) # 把光盘放进播放器[cite: 2]
        if not cap.isOpened(): raise FileNotFoundError(f"无法打开视频文件: {video_path}")

        # 拿视频的基本信息：帧率、宽、高[cite: 2]
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 准备输出视频（如果需要的话）[cite: 2]
        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, raw_fps / self.frame_stride, (w, h))

        need_rendering = (writer is not None) or show_preview
        frame_idx = 0
        records = []
        
        # 重置记忆，雇佣防抖摄影师[cite: 2]
        self.prev_centroid, self.prev_box, self.prev_kpts = None, None, None
        self.lost_counter = 0
        gmc = CameraMotionCompensator() if self.enable_gmc else None
        accum_cam_dx, accum_cam_dy = 0.0, 0.0 # 记录视频从头到尾总共偏移了多少[cite: 2]

        print(f"🎬 开始一步到位极速姿态提取 [帧采样 Stride={self.frame_stride}]: {video_path}...")

        while cap.isOpened():
            ret, frame = cap.read() # 读一帧画面[cite: 2]
            if not ret: break # 读完了就结束[cite: 2]

            frame_idx += 1
            if (frame_idx % self.frame_stride) != 0: continue # 如果设了跳帧，这里直接忽略[cite: 2]

            # 为了让 AI 算得快，如果不需要渲染原画，且画面大于640，就缩小画面丢给 AI[cite: 2]
            # proc_frame = cv2.resize(frame, (640, int(h * 640.0 / w)), interpolation=cv2.INTER_NEAREST) if (not need_rendering and w > 640) else frame
            # proc_h, proc_w = proc_frame.shape[:2]

            #无论出不出视频，只要原图宽大于 640，就强制压缩到 640 给后台模型去算
            if w > 640:
                proc_frame = cv2.resize(frame, (640,int(h * 640.0 / w)), interpolation=cv2.INTER_NEAREST)
            else:
                proc_frame = frame 
            proc_h, proc_w = proc_frame.shape[:2]    

            # 让 AI 专家 (YOLO) 辨认画面里的人[cite: 2]
            results = self.model(proc_frame, verbose=False, imgsz=480 if self.device == 'cpu' else 640, 
                                 device=self.device, classes=0, conf=0.25)[0]

            selected_target = None
            if len(results.boxes) > 0 and results.keypoints is not None:
                boxes = results.boxes
                kpts_xyn = results.keypoints.xyn.cpu().numpy() # 拿到相对坐标(0-1)[cite: 2]
                kpts_conf = results.keypoints.conf.cpu().numpy() if results.keypoints.conf is not None else np.ones((len(boxes), 17))
                # 挑选主角[cite: 2]
                selected_target = self._filter_and_track_person(boxes, kpts_xyn, kpts_conf, proc_w, proc_h)

            cam_dx_px, cam_dy_px = 0.0, 0.0
            person_pixel_box = selected_target['pixel_box'] if selected_target is not None else None
            
            # 让防抖摄影师算一下偏移量[cite: 2]
            if gmc is not None:
                cam_dx_px, cam_dy_px = gmc.estimate_camera_motion(proc_frame, person_box=person_pixel_box)
                accum_cam_dx += cam_dx_px / proc_w
                accum_cam_dy += cam_dy_px / proc_h
            
            # 为了让画图时能完美还原到高清大图上，先计算一下原图和小图的长宽比例
            scale_w = w / float(proc_w)
            scale_h = h / float(proc_h)

            # 如果找到了主角
            if selected_target is not None:
                curr_kpts = selected_target['kpts'].copy()
                confs = selected_target['confs']

                # 平滑处理：把上一帧的动作和这一帧融合一下，这样骨架看起来不会乱跳[cite: 2]
                if self.prev_kpts is not None:
                    curr_kpts = self.smooth_factor * curr_kpts + (1 - self.smooth_factor) * self.prev_kpts
                self.prev_kpts = curr_kpts.copy()

                stabilized_kpts = curr_kpts.copy()
                # 核心防抖逻辑：在人身上的骨架坐标里，硬生生地减去相机偏移的距离，骨架就“定住了”！[cite: 2]
                if self.enable_gmc:
                    stabilized_kpts[:, 0] -= accum_cam_dx
                    stabilized_kpts[:, 1] -= accum_cam_dy

                # 把这 17 个点的数据存在一个小本本 (字典) 里[cite: 2]
                frame_data = {'frame': frame_idx, 'detected': 1}
                for idx in range(17):
                    frame_data[f'kpt_{idx}_x'] = stabilized_kpts[idx, 0]
                    frame_data[f'kpt_{idx}_y'] = stabilized_kpts[idx, 1]
                    frame_data[f'kpt_{idx}_vis'] = confs[idx]
                records.append(frame_data)

                # 画图[cite: 2]
                if need_rendering:
                    annotated_frame = frame.copy()
                    kpts_pixel = curr_kpts.copy()
                    kpts_pixel[:, 0] *= w; kpts_pixel[:, 1] *= h
                    # 还原位移数据到原图尺寸，方便在屏幕左上角打印出正确的高清像素位移值
                    true_cam_dx = cam_dx_px * scale_w
                    true_cam_dy = cam_dy_px * scale_h

                    self._draw_hud_overlay(annotated_frame, gmc, true_cam_dx, true_cam_dy, accum_cam_dx, accum_cam_dy, kpts_pixel, confs,scale_w,scale_h)
            else:
                # 没找到人，记录空数据[cite: 2]
                self.prev_kpts = None
                frame_data = {'frame': frame_idx, 'detected': 0}
                for idx in range(17):
                    frame_data[f'kpt_{idx}_x'] = np.nan
                    frame_data[f'kpt_{idx}_y'] = np.nan
                    frame_data[f'kpt_{idx}_vis'] = np.nan
                records.append(frame_data)
                
                if need_rendering:
                    annotated_frame = frame.copy()
                    true_cam_dx = cam_dx_px * scale_w
                    true_cam_dy = cam_dy_px * scale_h
                    self._draw_hud_overlay(annotated_frame, gmc, true_cam_dx, true_cam_dy, accum_cam_dx, accum_cam_dy, None, None,scale_w,scale_h)

            # 保存和显示视频[cite: 2]
            if need_rendering and writer: writer.write(annotated_frame)
            if show_preview:
                cv2.imshow("GMC Anti-Shake Gait Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break # 按 q 键退出[cite: 2]

        # 所有的工作都干完了，关机收工[cite: 2]
        cap.release()
        if writer: writer.release()
        if show_preview: cv2.destroyAllWindows()

        # 把记录的所有数据，转成 pandas 的 DataFrame 格式，再保存成 CSV 表格文件[cite: 2]
        df = pd.DataFrame(records)
        df.to_csv(output_csv_path, index=False)
        print(f"✅ 极速处理完成！姿态 CSV 已导出: {output_csv_path}")
        return df

# 如果是你直接运行这个脚本，就执行以下动作[cite: 2]
if __name__ == '__main__':
    # 请出追踪员，调整好参数[cite: 2]
    extractor = RobustPoseExtractor(conf_thresh=0.35, smooth_factor=0.4, enable_gmc=True, frame_stride=2)
    # 给定视频和保存地址，开始干活！[cite: 2]
    extractor.extract("data/sample/stroke_1.mp4", "data/sample/stroke_1.csv")