import os
import logging
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from typing import List, Tuple, Dict, Optional

# 配置标准日志输出，取代 print，提升后台批处理时的稳定性与日志追踪能力
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UltimateSegmentedGaitAnalyzer:
    """
    终极临床级视角自适应步态分析器 (V9 - 物理尺度归一化 & 局部抗噪重构版):
    本类致力于解决野外单目 2D 摄像头提取步态时的“透视畸变”、“像素抖动”与“相位错位”问题。
    """
    def __init__(self, raw_video_fps: float = 30.0, min_segment_frames: int = 15):
        self.raw_video_fps = raw_video_fps
        self.min_segment_frames = min_segment_frames

    def _detect_viewpoint(self, sub_df: pd.DataFrame, inst_torso_h: np.ndarray) -> Tuple[str, float]:
        """
        【视角感知模块】
        做了什么：自动判断受试者是处于正面 (FRONTAL)、侧面 (SIDE) 还是斜向 (OBLIQUE)。
        怎样做的：计算“双肩宽度”与“躯干高度”的比例。
        抗噪处理：由于单目摄像头走近走远会产生透视放大/缩小，为防止视角在短时间内来回跳变，
                 采用对整个片段的肩宽比取中位数 (np.median) 进行一锤定音的磁滞判定。
        """
        ls_x, rs_x = sub_df['kpt_5_x'].values, sub_df['kpt_6_x'].values
        shoulder_w = np.abs(ls_x - rs_x)
        shoulder_ratio = float(np.median(shoulder_w / (inst_torso_h + 1e-5)))
        
        if shoulder_ratio < 0.15:
            return "SIDE", round(shoulder_ratio, 3)
        elif shoulder_ratio > 0.22:
            return "FRONTAL", round(shoulder_ratio, 3)
        return "OBLIQUE", round(shoulder_ratio, 3)

    def _calc_knee_rom(self, sub_df: pd.DataFrame) -> float:
        """
        【膝关节活动度 (ROM) 计算模块】
        做了什么：计算受试者行走时膝盖的最大屈伸角度 (Range of Motion)。
        怎样做的：利用髋、膝、踝三个关键点，构造“大腿向量”与“小腿向量”，
                 通过向量点乘反余弦 (arccos) 求出夹角。
        抗噪处理：过滤掉因人体遮挡导致的 NaN 噪点，分别求左右腿的角度峰峰值 (Peak-to-Peak)，
                 取最大的一侧代表当前受试者的下肢最大舒展能力。
        """
        lh_x, lh_y = sub_df['kpt_11_x'].values, sub_df['kpt_11_y'].values
        lk_x, lk_y = sub_df['kpt_13_x'].values, sub_df['kpt_13_y'].values
        la_x, la_y = sub_df['kpt_15_x'].values, sub_df['kpt_15_y'].values

        rh_x, rh_y = sub_df['kpt_12_x'].values, sub_df['kpt_12_y'].values
        rk_x, rk_y = sub_df['kpt_14_x'].values, sub_df['kpt_14_y'].values
        ra_x, ra_y = sub_df['kpt_16_x'].values, sub_df['kpt_16_y'].values

        def get_leg_angles(h_x, h_y, k_x, k_y, a_x, a_y):
            v_thigh = np.stack([h_x - k_x, h_y - k_y], axis=1)
            v_shank = np.stack([a_x - k_x, a_y - k_y], axis=1)

            norm_thigh = np.linalg.norm(v_thigh, axis=1) + 1e-5
            norm_shank = np.linalg.norm(v_shank, axis=1) + 1e-5

            dot = np.sum(v_thigh * v_shank, axis=1) / (norm_thigh * norm_shank)
            dot = np.clip(dot, -1.0, 1.0)
            return np.degrees(np.arccos(dot))

        left_angles = get_leg_angles(lh_x, lh_y, lk_x, lk_y, la_x, la_y)
        right_angles = get_leg_angles(rh_x, rh_y, rk_x, rk_y, ra_x, ra_y)

        left_valid = left_angles[~np.isnan(left_angles)]
        right_valid = right_angles[~np.isnan(right_angles)]

        left_rom = float(np.ptp(left_valid)) if len(left_valid) > 5 else 0.0
        right_rom = float(np.ptp(right_valid)) if len(right_valid) > 5 else 0.0

        return round(max(left_rom, right_rom), 1)

    def _segment_straight_walks(self, valid_df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
        """
        【智能来回折返切割模块】
        做了什么：将一段包含“来回走动、转身”的长视频，自动切分为多个“单向直线行走”的纯净片段。
        怎样做的：提取骨盆中点轨迹，进行低通滤波平滑后，通过寻找运动轨迹的“极值点”来定位转身位置。
                 随后利用速度阈值剔除转身时产生的减速缓冲期。
        抗噪处理：防止转身横移的剧烈位移被系统误认为“踉跄失稳”，确保步态评估都在稳态直行下进行。
        """
        n_frames = len(valid_df)
        if n_frames < self.min_segment_frames * 2:
            return [("全视频片段", valid_df)]

        # 利用骨盆中点建立全局运动信号
        ls_x, ls_y = valid_df['kpt_5_x'].values, valid_df['kpt_5_y'].values
        rs_x, rs_y = valid_df['kpt_6_x'].values, valid_df['kpt_6_y'].values
        lh_x, lh_y = valid_df['kpt_11_x'].values, valid_df['kpt_11_y'].values
        rh_x, rh_y = valid_df['kpt_12_x'].values, valid_df['kpt_12_y'].values

        sh_x, sh_y = (ls_x + rs_x) / 2.0, (ls_y + rs_y) / 2.0
        hip_x, hip_y = (lh_x + rh_x) / 2.0, (lh_y + rh_y) / 2.0
        inst_torso_h = np.maximum(np.sqrt((sh_x - hip_x)**2 + (sh_y - hip_y)**2), 1e-4)

        win_len = min(31, n_frames if n_frames % 2 != 0 else n_frames - 1)
        smooth_x = savgol_filter(hip_x, win_len, 2)
        smooth_h = savgol_filter(inst_torso_h, win_len, 2)

        x_ptp = np.ptp(smooth_x)
        h_ptp = np.ptp(smooth_h)

        # 区分横穿走 (主轴 X) 还是迎面走近走远 (主轴为身高 H)
        axis_signal, signal_ptp = (smooth_x, x_ptp) if (x_ptp > h_ptp * 1.3 and x_ptp > 0.15) else (smooth_h, h_ptp)

        # 若位移过小视作原地踏步，不切片
        if signal_ptp < 0.12:
            return [("全视频片段", valid_df)]

        min_dist_frames = max(15, int(self.raw_video_fps * 0.8))
        prom = max(0.04, 0.12 * signal_ptp)
        
        # 提取折返波峰
        peaks_pos, _ = find_peaks(axis_signal, distance=min_dist_frames, prominence=prom)
        peaks_neg, _ = find_peaks(-axis_signal, distance=min_dist_frames, prominence=prom)

        all_turns = np.sort(np.concatenate([peaks_pos, peaks_neg]))
        valid_turns = [t for t in all_turns if self.min_segment_frames <= t <= (n_frames - self.min_segment_frames)]

        if not valid_turns:
            return [("单向直行片段", valid_df)]

        # 剔除转身前后的减速缓冲期
        velocity = np.diff(axis_signal, prepend=axis_signal[0])
        speed = np.abs(velocity)
        speed_thresh = np.percentile(speed, 75) * 0.35

        turn_buffers = []
        for turn_center in valid_turns:
            search_start = max(0, turn_center - 15)
            search_end = min(n_frames, turn_center + 15)
            exact_center = search_start + np.argmin(speed[search_start:search_end])

            t_start, t_end = exact_center, exact_center
            while t_start > 0 and speed[t_start] < speed_thresh and (exact_center - t_start) < 20:
                t_start -= 1
            while t_end < n_frames - 1 and speed[t_end] < speed_thresh and (t_end - exact_center) < 20:
                t_end += 1

            turn_buffers.append((t_start, t_end))

        segments = []
        curr_start = 0

        # 切片
        for i, (t_start, t_end) in enumerate(turn_buffers):
            if t_start - curr_start >= self.min_segment_frames:
                segments.append((f"直行段_{len(segments)+1}", valid_df.iloc[curr_start:t_start].copy()))
            curr_start = t_end

        if n_frames - curr_start >= self.min_segment_frames:
            segments.append((f"直行段_{len(segments)+1}", valid_df.iloc[curr_start:n_frames].copy()))

        return segments if segments else [("全视频片段", valid_df)]

    def _analyze_sub_segment(self, sub_df: pd.DataFrame) -> Optional[Dict]:
        """
        【核心步态生物力学分析与特征提取模块】
        计算步幅、步频、节律变异性 (CV)、双支撑相及躯干摇晃等所有医学子指标并打分。
        """
        if len(sub_df) < 15:
            return None

        frames = sub_df['frame'].values
        frame_diffs = np.diff(frames)
        stride = np.median(frame_diffs) if len(frame_diffs) > 0 else 1.0
        effective_fps = self.raw_video_fps / max(1.0, stride)

        # 提取关键点
        ls_x, ls_y = sub_df['kpt_5_x'].values, sub_df['kpt_5_y'].values
        rs_x, rs_y = sub_df['kpt_6_x'].values, sub_df['kpt_6_y'].values
        lh_x, lh_y = sub_df['kpt_11_x'].values, sub_df['kpt_11_y'].values
        rh_x, rh_y = sub_df['kpt_12_x'].values, sub_df['kpt_12_y'].values
        la_x, la_y = sub_df['kpt_15_x'].values, sub_df['kpt_15_y'].values
        ra_x, ra_y = sub_df['kpt_16_x'].values, sub_df['kpt_16_y'].values

        sh_x, sh_y = (ls_x + rs_x) / 2.0, (ls_y + rs_y) / 2.0
        hip_x, hip_y = (lh_x + rh_x) / 2.0, (lh_y + rh_y) / 2.0
        
        # 物理标尺基准 (Torso Height): 用于消除因人物离镜头远近带来的绝对像素缩放影响
        inst_torso_h = np.maximum(np.sqrt((sh_x - hip_x)**2 + (sh_y - hip_y)**2), 1e-4)

        view_type, _ = self._detect_viewpoint(sub_df, inst_torso_h)
        knee_rom = self._calc_knee_rom(sub_df)

        # ----------------------------------------------------
        # 1. 迈步波峰提取：透视去趋势 (Detrending) + 正面双相交替
        # ----------------------------------------------------
        win_torso = min(15, len(inst_torso_h) if len(inst_torso_h) % 2 != 0 else len(inst_torso_h) - 1)
        smooth_scale = savgol_filter(inst_torso_h, win_torso, 1)

        raw_ankle_dist = np.sqrt((la_x - ra_x)**2 + (la_y - ra_y)**2)
        ankle_dist_norm = raw_ankle_dist / smooth_scale
        win_len = min(11, len(ankle_dist_norm) if len(ankle_dist_norm) % 2 != 0 else len(ankle_dist_norm) - 1)
        smoothed_dist = savgol_filter(ankle_dist_norm, win_len, 2)

        # 宏观去趋势 (High-pass Filter): 利用 31 帧长滑窗抽出走近时的斜率基线，相减得到纯净步态波峰
        win_macro = min(31, len(smoothed_dist) if len(smoothed_dist) % 2 != 0 else len(smoothed_dist) - 1)
        detrended_signal = smoothed_dist - savgol_filter(smoothed_dist, win_macro, 1)

        min_peak_dist = max(4, int(effective_fps * 0.35))
        
        if view_type in ["FRONTAL", "OBLIQUE"]:
            # 【核心修复：正面双相提取法则】
            # 迎面走向镜头时，左右脚交叉会导致单纯欧氏距离在 2D 重叠而漏抓。
            # 这里利用纵坐标相对高度差 dy，区分左脚在前(正峰)和右脚在前(负峰)，精确抗重叠。
            dy_norm = (la_y - ra_y) / smooth_scale
            smooth_dy = savgol_filter(dy_norm, win_len, 2)
            prom_dy = max(0.015, 0.25 * np.std(smooth_dy))
            
            peaks_pos, _ = find_peaks(smooth_dy, distance=min_peak_dist, prominence=prom_dy)
            peaks_neg, _ = find_peaks(-smooth_dy, distance=min_peak_dist, prominence=prom_dy)
            peaks = np.sort(np.concatenate([peaks_pos, peaks_neg]))
        else:
            # 侧面纯切变位移清晰，直接使用去趋势欧氏距离信号找波峰即可
            prom = max(0.015, 0.30 * np.std(detrended_signal))
            peaks, _ = find_peaks(detrended_signal, distance=min_peak_dist, prominence=prom)

        # 最终安全网: 兜底使用原始平滑信号
        if len(peaks) < 3:
            peaks, _ = find_peaks(smoothed_dist, distance=min_peak_dist, prominence=0.03)

        # 强制过滤迈步周期少于 2 次 (波峰<3) 的碎片段
        if len(peaks) < 3:
            return None

        # ----------------------------------------------------
        # 2. 绝对物理时间与变异性计算 (修复 GAI 相位错位 Bug)
        # ----------------------------------------------------
        peak_frames = frames[peaks]
        step_intervals_sec = np.diff(peak_frames) / self.raw_video_fps

        step_lengths = smoothed_dist[peaks]
        step_mean = float(np.mean(step_lengths)) if len(step_lengths) > 0 else 0.0
        step_std = float(np.std(step_lengths)) if len(step_lengths) > 0 else 0.0
        step_cv = float(step_std / (step_mean + 1e-5)) if step_mean > 0 else 0.0

        # 时间变异 (抗 2D 透视干扰，最可靠的节律指标)
        stride_time_cv = float(np.std(step_intervals_sec) / (np.mean(step_intervals_sec) + 1e-5)) if len(step_intervals_sec) > 1 else 0.0

        # 【重点修复】：局部相邻步态不对称指数 (Local Gait Asymmetry Index)
        # 弃用全局奇偶数步比对，防止因 YOLO 单次漏检导致左/右脚相位全局错位。
        if len(step_intervals_sec) > 1:
            # 直接计算相邻两步时间间隔的差值绝对值，局部错位不再污染全局
            step_diffs = np.abs(np.diff(step_intervals_sec))
            gai = float(np.mean(step_diffs) / (np.mean(step_intervals_sec) + 1e-5))
        else:
            gai = 0.0

        # ----------------------------------------------------
        # 3. 躯干摇晃 (Sway) 与 Knee ROM 打分
        # ----------------------------------------------------
        dx = sh_x - hip_x
        dy = sh_y - hip_y
        trunk_angle = np.degrees(np.arctan2(dx, -dy))
        win_len_t = min(21, len(trunk_angle) if len(trunk_angle) % 2 != 0 else len(trunk_angle) - 1)
        com_sway_std = float(np.std(trunk_angle - savgol_filter(trunk_angle, win_len_t, 1)))

        # 躯干摇晃打分映射
        score_sway = 100.0 - (com_sway_std / 2.5) * 15.0 if com_sway_std <= 2.5 else \
                     85.0 - (com_sway_std - 2.5) / (6.0 - 2.5) * 35.0 if com_sway_std <= 6.0 else \
                     max(0.0, 50.0 - (com_sway_std - 6.0) * 10.0)

        # 膝关节活动度单目校准：2D 下 28° 即达到健康步态阈值
        score_knee = 100.0 if knee_rom >= 28.0 else \
                     60.0 + (knee_rom - 18.0) / (28.0 - 18.0) * 40.0 if knee_rom >= 18.0 else \
                     max(0.0, (knee_rom / 18.0) * 60.0)

        # 正面迎面投影下，膝盖弯曲在 2D 是不可见的，故彻底禁用 Knee ROM，仅用 Sway
        score_balance = score_knee if view_type == "SIDE" else score_sway

        # ----------------------------------------------------
        # 4. 双支撑相优化：自适应滑窗绝对速度阈值
        # ----------------------------------------------------
        # 使用相对于地面的绝对物理速度，并按当前躯干尺寸归一化
        la_v_raw = np.sqrt(np.diff(la_x, prepend=la_x[0])**2 + np.diff(la_y, prepend=la_y[0])**2) * effective_fps / smooth_scale
        ra_v_raw = np.sqrt(np.diff(ra_x, prepend=ra_x[0])**2 + np.diff(ra_y, prepend=ra_y[0])**2) * effective_fps / smooth_scale
        
        # 滑窗平滑，消除关键点在单帧间的 Jitter 像素抖动
        win_v = min(7, len(la_v_raw) if len(la_v_raw) % 2 != 0 else len(la_v_raw) - 1)
        if win_v >= 3:
            la_v = savgol_filter(la_v_raw, win_v, 1)
            ra_v = savgol_filter(ra_v_raw, win_v, 1)
        else:
            la_v, ra_v = la_v_raw, ra_v_raw

        # 采用自适应门槛：取每一只脚在整个行走过程中自身速度分布的"下 40% 分位数"作为其着地静止阈值 (Planted Phase)
        la_thresh = np.percentile(la_v, 40)
        ra_thresh = np.percentile(ra_v, 40)
        
        is_double_support = (la_v <= la_thresh) & (ra_v <= ra_thresh)
        double_support_ratio = float(np.clip(np.mean(is_double_support), 0.15, 0.70))

        if double_support_ratio <= 0.45:
            score_support = 100.0 - (double_support_ratio / 0.45) * 15.0
        elif double_support_ratio <= 0.60:
            score_support = 85.0 - (double_support_ratio - 0.45) / (0.60 - 0.45) * 35.0
        else:
            score_support = max(0.0, 50.0 - (double_support_ratio - 0.60) * 150.0)

        # ----------------------------------------------------
        # 5. 节律得分与综合 GSI (透视空间降权)
        # ----------------------------------------------------
        # 【重点修复】：大幅降权 step_cv (乘 0.4)
        # 降低受单目 2D 投影角度污染极大的像素空间指标，
        # 让具有绝对物理时间意义的 stride_time_cv 与 gai 主导节律评估。
        combined_cv = max(step_cv * 0.4, stride_time_cv, gai)

        # 容忍阈值平滑：单目无约束下容忍 0.22 的微变异
        if combined_cv <= 0.22:
            score_rhythm = 100.0 - (combined_cv / 0.22) * 15.0
        elif combined_cv <= 0.45:
            score_rhythm = 85.0 - (combined_cv - 0.22) / (0.45 - 0.22) * 35.0
        else:
            score_rhythm = max(0.0, 50.0 - (combined_cv - 0.45) * 100.0)

        # 加权输出终极 GSI 分数
        final_gsi = float(np.round(np.clip(0.40 * score_rhythm + 0.35 * score_support + 0.25 * score_balance, 0.0, 100.0), 1))

        return {
            'view_type': view_type,
            'knee_rom': knee_rom,
            'step_mean': round(step_mean, 4),
            'step_cv': round(step_cv, 4),
            'stride_time_cv': round(stride_time_cv, 4),
            'gai_asymmetry': round(gai, 4),
            'double_support_ratio': round(double_support_ratio, 4),
            'com_sway_std': round(com_sway_std, 2),
            'gsi_score': final_gsi,
            'segment_len': len(sub_df) # 此处将片段帧数留作加权合并的权重
        }

    def process_single_csv(self, csv_path: str) -> Optional[Dict]:
        """
        【模块 5：全局加权聚合】
        将一段视频里的所有有效独立步行段落，通过 "每段帧数" 作为权重进行加权平均。
        彻底淘汰“最低分片段一票否决制”，消除起步过渡时的边缘噪点拉跨总分的现象。
        """
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.error(f"读取 CSV 失败 ({csv_path}): {e}")
            return None

        if 'detected' not in df.columns:
            return None

        valid_df = df[df['detected'] == 1].copy()
        if len(valid_df) < 15:
            return None

        segments = self._segment_straight_walks(valid_df)
        
        # 逐段计算，自动滤除不符合迈步条件的过渡碎片段
        seg_results = [res for _, sub_df in segments if (res := self._analyze_sub_segment(sub_df))]

        if not seg_results:
            return None

        # 全局加权求均值
        total_frames = sum([res['segment_len'] for res in seg_results])
        
        weighted_metrics = {}
        for key in ['knee_rom', 'step_mean', 'step_cv', 'stride_time_cv', 'gai_asymmetry', 'double_support_ratio', 'com_sway_std', 'gsi_score']:
            weighted_metrics[key] = round(sum(res[key] * (res['segment_len'] / total_frames) for res in seg_results), 4)

        final_gsi = round(weighted_metrics['gsi_score'], 1)
        
        # 根据最终 GSI 判定风险等级
        risk_level = "Level-1 (低风险/健康)" if final_gsi >= 75.0 else \
                     "Level-2 (中风险/轻度退化)" if final_gsi >= 50.0 else \
                     "Level-3 (高风险/重度失稳)"

        return {
            'video_name': os.path.basename(csv_path),
            'view_type': seg_results[0]['view_type'], # 直接沿用时长最大的主干段落视角
            **weighted_metrics,
            'risk_level': risk_level
        }

    def process_directory(self, input_dir: str = ".", output_csv: str = "data/output/gait_medical_features.csv") -> Optional[pd.DataFrame]:
        """批量处理与 CSV 特征矩阵导出"""
        if not os.path.exists(input_dir):
            logging.error(f"输入目录不存在: {input_dir}")
            return None

        csv_files = [os.path.join(r, f) for r, d, fs in os.walk(input_dir) for f in fs if f.lower().endswith('.csv') and 'pose' in f.lower()]
        records = [res for f in csv_files if (res := self.process_single_csv(f))]

        if not records:
            logging.warning("未能从目录中提取到有效步态特征。")
            return None

        df_res = pd.DataFrame(records)
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        df_res.to_csv(output_csv, index=False)
        
        logging.info("=" * 80)
        logging.info(f"✅ 特征矩阵导出完成: {output_csv}")
        
        logging.info("=" * 80)
        print(df_res[['video_name', 'view_type', 'knee_rom', 'step_mean', 'step_cv', 'stride_time_cv', 'gai_asymmetry', 'double_support_ratio', 'com_sway_std', 'gsi_score', 'risk_level']].to_string(index=False))
        return df_res

if __name__ == '__main__':
    analyzer = UltimateSegmentedGaitAnalyzer(raw_video_fps=30.0)
    analyzer.process_directory(input_dir=".", output_csv="data/output/gait_medical_features.csv")