import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter

class UltimateSegmentedGaitAnalyzer:
    """
    终极临床级视角自适应步态分析器 (Viewpoint-Aware & Knee ROM Adaptive Analyzer):
    1. 视角自动感知 (Viewpoint Auto-Detection): 自动识别 FRONTAL (正面), SIDE (侧面), OBLIQUE (斜向)
    2. 侧面视角容错 (Side-View Adaptation): 侧面视角自动激活膝关节活动度 (Knee ROM 伸屈角) 评估肢体控制力与僵硬度
    3. 全方向 & 多折返智能直行分段器: 支持单向直行及 1 至 N 次来回走动折返
    4. 彻底移除 stumble_count 突变误扣分，纯依靠生物力学稳态与关节活动度指标
    """
    def __init__(self, raw_video_fps=30.0, min_segment_frames=15):
        self.raw_video_fps = raw_video_fps
        self.min_segment_frames = min_segment_frames

    def _detect_viewpoint(self, sub_df, inst_torso_h):
        """基于归一化肩膀比例自动判定视角"""
        ls_x, rs_x = sub_df['kpt_5_x'].values, sub_df['kpt_6_x'].values
        shoulder_w = np.abs(ls_x - rs_x)
        shoulder_ratio = float(np.median(shoulder_w / (inst_torso_h + 1e-5)))
        
        # 校准后的归一化门槛：
        if shoulder_ratio < 0.15:
            return "SIDE", round(shoulder_ratio, 3)     # 侧面视角 (如 0.060)
        elif shoulder_ratio > 0.22:
            return "FRONTAL", round(shoulder_ratio, 3)  # 正面视角 (如 0.286)
        return "OBLIQUE", round(shoulder_ratio, 3)      # 斜向视角

    def _calc_knee_rom(self, sub_df):
        """计算侧面视角下膝关节伸屈活动度 (Flexion/Extension ROM)"""
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

        knee_rom = max(left_rom, right_rom)
        return round(knee_rom, 1)

    def _segment_straight_walks(self, valid_df):
        """智能全向 & 多折返直行分段器"""
        n_frames = len(valid_df)
        if n_frames < self.min_segment_frames * 2:
            return [("全视频片段", valid_df)]

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

        if x_ptp > h_ptp * 1.3 and x_ptp > 0.15:
            axis_signal = smooth_x
        else:
            axis_signal = smooth_h

        min_dist_frames = max(15, int(self.raw_video_fps * 0.8))
        signal_ptp = np.ptp(axis_signal)
        
        peaks_pos, _ = find_peaks(axis_signal, distance=min_dist_frames, prominence=max(0.01, 0.08 * signal_ptp))
        peaks_neg, _ = find_peaks(-axis_signal, distance=min_dist_frames, prominence=max(0.01, 0.08 * signal_ptp))

        all_turns = np.sort(np.concatenate([peaks_pos, peaks_neg]))
        valid_turns = [t for t in all_turns if self.min_segment_frames <= t <= (n_frames - self.min_segment_frames)]

        if not valid_turns:
            return [("单向直行片段", valid_df)]

        velocity = np.diff(axis_signal, prepend=axis_signal[0])
        speed = np.abs(velocity)
        max_speed = np.percentile(speed, 75)
        speed_thresh = max_speed * 0.35

        turn_buffers = []
        for turn_center in valid_turns:
            search_start = max(0, turn_center - 15)
            search_end = min(n_frames, turn_center + 15)
            exact_center = search_start + np.argmin(speed[search_start:search_end])

            t_start = exact_center
            while t_start > 0 and speed[t_start] < speed_thresh and (exact_center - t_start) < 20:
                t_start -= 1

            t_end = exact_center
            while t_end < n_frames - 1 and speed[t_end] < speed_thresh and (t_end - exact_center) < 20:
                t_end += 1

            turn_buffers.append((t_start, t_end))

        segments = []
        curr_start = 0

        for i, (t_start, t_end) in enumerate(turn_buffers):
            if t_start - curr_start >= self.min_segment_frames:
                seg_df = valid_df.iloc[curr_start : t_start].copy()
                segments.append((f"直行段_{len(segments)+1}", seg_df))
            curr_start = t_end

        if n_frames - curr_start >= self.min_segment_frames:
            seg_df = valid_df.iloc[curr_start : n_frames].copy()
            segments.append((f"直行段_{len(segments)+1}", seg_df))

        if not segments:
            segments.append(("全视频片段", valid_df))

        return segments

    def _analyze_sub_segment(self, sub_df):
        if len(sub_df) < 15:
            return None

        frames = sub_df['frame'].values
        frame_diffs = np.diff(frames)
        stride = np.median(frame_diffs) if len(frame_diffs) > 0 else 1.0
        effective_fps = self.raw_video_fps / max(1.0, stride)

        ls_x, ls_y = sub_df['kpt_5_x'].values, sub_df['kpt_5_y'].values
        rs_x, rs_y = sub_df['kpt_6_x'].values, sub_df['kpt_6_y'].values
        lh_x, lh_y = sub_df['kpt_11_x'].values, sub_df['kpt_11_y'].values
        rh_x, rh_y = sub_df['kpt_12_x'].values, sub_df['kpt_12_y'].values
        la_x, la_y = sub_df['kpt_15_x'].values, sub_df['kpt_15_y'].values
        ra_x, ra_y = sub_df['kpt_16_x'].values, sub_df['kpt_16_y'].values

        sh_x, sh_y = (ls_x + rs_x) / 2.0, (ls_y + rs_y) / 2.0
        hip_x, hip_y = (lh_x + rh_x) / 2.0, (lh_y + rh_y) / 2.0
        inst_torso_h = np.maximum(np.sqrt((sh_x - hip_x)**2 + (sh_y - hip_y)**2), 1e-4)

        # 1. 视角感知与膝关节活动度计算
        view_type, shoulder_ratio = self._detect_viewpoint(sub_df, inst_torso_h)
        knee_rom = self._calc_knee_rom(sub_df)

        # 2. 动态低通透视缩放平滑
        raw_ankle_dist = np.sqrt((la_x - ra_x)**2 + (la_y - ra_y)**2)
        win_torso = min(15, len(inst_torso_h) if len(inst_torso_h) % 2 != 0 else len(inst_torso_h) - 1)
        smooth_scale = savgol_filter(inst_torso_h, win_torso, 1)

        ankle_dist_norm = raw_ankle_dist / smooth_scale
        win_len = min(11, len(ankle_dist_norm) if len(ankle_dist_norm) % 2 != 0 else len(ankle_dist_norm) - 1)
        smoothed_dist = savgol_filter(ankle_dist_norm, win_len, 2)

        # 3. 寻找波峰
        peaks, _ = find_peaks(smoothed_dist, distance=max(3, int(effective_fps * 0.20)), prominence=0.02)
        if len(peaks) < 2:
            return None

        # 4. 绝对物理时间计算
        peak_frames = frames[peaks]
        step_intervals_sec = np.diff(peak_frames) / self.raw_video_fps

        step_lengths = smoothed_dist[peaks]
        step_mean = float(np.mean(step_lengths))
        step_std = float(np.std(step_lengths))
        step_cv = float(step_std / (step_mean + 1e-5))

        stride_time_cv = float(np.std(step_intervals_sec) / (np.mean(step_intervals_sec) + 1e-5))

        even_steps = step_intervals_sec[0::2]
        odd_steps = step_intervals_sec[1::2]
        gai = float(abs(np.mean(even_steps) - np.mean(odd_steps)) / (np.mean(step_intervals_sec) + 1e-5)) if len(even_steps)>0 and len(odd_steps)>0 else 0.0

        # 5. 视角自适应躯干摇晃与肢体控制打分
        dx = sh_x - hip_x
        dy = sh_y - hip_y
        trunk_angle = np.degrees(np.arctan2(dx, -dy))
        win_len_t = min(21, len(trunk_angle) if len(trunk_angle) % 2 != 0 else len(trunk_angle) - 1)
        com_sway_std = float(np.std(trunk_angle - savgol_filter(trunk_angle, win_len_t, 1)))

        if view_type == "SIDE":
            # 侧面视角：激活 Knee ROM (膝关节活动度) 评估 (正常 42°~65°，低于 25° 属僵硬退化)
            if knee_rom >= 42.0:
                score_balance = 100.0
            elif knee_rom >= 25.0:
                score_balance = 60.0 + (knee_rom - 25.0) / (42.0 - 25.0) * 35.0
            else:
                score_balance = max(0.0, (knee_rom / 25.0) * 60.0)
        else:
            # 正面/斜向视角：使用 com_sway_std (左右摇晃) 评估
            if com_sway_std <= 2.5:
                score_balance = 100.0 - (com_sway_std / 2.5) * 15.0
            elif com_sway_std <= 6.0:
                score_balance = 85.0 - (com_sway_std - 2.5) / (6.0 - 2.5) * 35.0
            else:
                score_balance = max(0.0, 50.0 - (com_sway_std - 6.0) * 10.0)

        # 6. 双支撑相占比
        rel_la_x = (la_x - hip_x) / smooth_scale
        rel_la_y = (la_y - hip_y) / smooth_scale
        rel_ra_x = (ra_x - hip_x) / smooth_scale
        rel_ra_y = (ra_y - hip_y) / smooth_scale

        la_v = np.sqrt(np.diff(rel_la_x, prepend=rel_la_x[0])**2 + np.diff(rel_la_y, prepend=rel_la_y[0])**2) * effective_fps
        ra_v = np.sqrt(np.diff(rel_ra_x, prepend=rel_ra_x[0])**2 + np.diff(rel_ra_y, prepend=rel_ra_y[0])**2) * effective_fps

        la_v_norm = la_v / smooth_scale
        ra_v_norm = ra_v / smooth_scale

        ABS_STANCE_THRESH = 0.35 
        is_double_support = (la_v_norm <= ABS_STANCE_THRESH) & (ra_v_norm <= ABS_STANCE_THRESH)
        double_support_ratio = float(np.clip(np.mean(is_double_support), 0.15, 0.70))

        # 7. 节律打分与视角加权 GSI
        combined_cv = max(step_cv, stride_time_cv, gai * 1.2)

        if combined_cv <= 0.15:
            score_rhythm = 100.0 - (combined_cv / 0.15) * 15.0
        elif combined_cv <= 0.35:
            score_rhythm = 85.0 - (combined_cv - 0.15) / (0.35 - 0.15) * 35.0
        else:
            score_rhythm = max(0.0, 50.0 - (combined_cv - 0.35) * 120.0)

        if double_support_ratio <= 0.45:
            score_support = 100.0 - (double_support_ratio / 0.45) * 15.0
        elif double_support_ratio <= 0.60:
            score_support = 85.0 - (double_support_ratio - 0.45) / (0.60 - 0.45) * 35.0
        else:
            score_support = max(0.0, 50.0 - (double_support_ratio - 0.60) * 150.0)

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
            'gsi_score': final_gsi
        }

    def process_single_csv(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"⚠️ 读取 CSV 失败 ({csv_path}): {e}")
            return None

        if 'detected' not in df.columns:
            return None

        valid_df = df[df['detected'] == 1].copy()
        if len(valid_df) < 15:
            return None

        segments = self._segment_straight_walks(valid_df)
        
        seg_results = []
        for seg_name, sub_df in segments:
            res = self._analyze_sub_segment(sub_df)
            if res:
                seg_results.append(res)

        if not seg_results:
            return None

        gsi_scores = [x['gsi_score'] for x in seg_results]
        if min(gsi_scores) < 70.0:
            best_res = min(seg_results, key=lambda x: x['gsi_score'])
        else:
            avg_score = float(np.mean(gsi_scores))
            best_res = min(seg_results, key=lambda x: abs(x['gsi_score'] - avg_score))
            best_res['gsi_score'] = round(avg_score, 1)

        final_gsi = best_res['gsi_score']

        if final_gsi >= 75.0:
            risk_level = "Level-1 (低风险/健康)"
        elif final_gsi >= 50.0:
            risk_level = "Level-2 (中风险/轻度退化)"
        else:
            risk_level = "Level-3 (高风险/重度失稳)"

        video_name = os.path.basename(csv_path)

        return {
            'video_name': video_name,
            'view_type': best_res['view_type'],
            'knee_rom': best_res['knee_rom'],
            'step_mean': best_res['step_mean'],
            'step_cv': best_res['step_cv'],
            'stride_time_cv': best_res['stride_time_cv'],
            'gai_asymmetry': best_res['gai_asymmetry'],
            'double_support_ratio': best_res['double_support_ratio'],
            'com_sway_std': best_res['com_sway_std'],
            'gsi_score': final_gsi,
            'risk_level': risk_level
        }

    def process_directory(self, input_dir=".", output_csv="data/output/gait_medical_features.csv"):
        if not os.path.exists(input_dir):
            return None

        csv_files = [os.path.join(r, f) for r, d, fs in os.walk(input_dir) for f in fs if f.lower().endswith('.csv') and 'pose' in f.lower()]
        
        records = []
        for f in csv_files:
            res = self.process_single_csv(f)
            if res is not None:
                records.append(res)

        if not records:
            return None

        df_res = pd.DataFrame(records)
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        df_res.to_csv(output_csv, index=False)
        print("=" * 80)
        print(f"✅ 特征矩阵导出完成: {output_csv}")
        print("=" * 80)
        print(df_res[['video_name', 'view_type', 'knee_rom', 'step_mean', 'step_cv', 'stride_time_cv', 'gai_asymmetry', 'double_support_ratio', 'com_sway_std', 'gsi_score', 'risk_level']].to_string(index=False))
        return df_res

if __name__ == '__main__':
    analyzer = UltimateSegmentedGaitAnalyzer(raw_video_fps=30.0)
    analyzer.process_directory(input_dir=".", output_csv="data/output/gait_medical_features.csv")