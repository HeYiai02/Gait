import os
import joblib
import numpy as np
import pandas as pd
from medical_metrics import UltimateSegmentedGaitAnalyzer

def predict_gait_risk(new_pose_csv, model_path="models/gsi_risk_model.joblib"):
    print(f"\n🎬 正在准备评估姿态文件: {new_pose_csv}")

    # 1. 实例化最新版分析器 (自动分段 + 采样率自适应)
    analyzer = UltimateSegmentedGaitAnalyzer(raw_video_fps=30.0)
    metrics = analyzer.process_single_csv(new_pose_csv)
    
    if metrics is None:
        print("🛑 [拦截]: 该视频有效直行步态帧数不足或无法提取波峰，已安全忽略。")
        return None

    # 2. 加载模型或输出临床得分
    if not os.path.exists(model_path):
        print(f"⚠️ 未找到已训练的模型权重: {model_path}，将直接使用临床规则 GSI 得分。")
        predicted_gsi = metrics['gsi_score']
    else:
        model_data = joblib.load(model_path)
        pipeline = model_data['pipeline']
        feature_names = model_data['feature_names']

        # 组织特征输入 (纯 2D numpy 数组输入，避免特征名警告)
        input_data = np.array([[metrics[col] for col in feature_names]])
        predicted_gsi = round(float(pipeline.predict(input_data)[0]), 1)

    # 3. 结果汇总
    fall_risk_idx = round(100.0 - predicted_gsi, 1)
    risk_level = metrics['risk_level']

    print("\n" + "="*50)
    print(f"📊 分析视频: {metrics['video_name']}")
    print(f"📈 步长变异系数 (Step CV): {metrics['step_cv']}")
    print(f"⏱️ 步频变异系数 (Stride Time CV): {metrics['stride_time_cv']}")
    print(f"⚖️ 身体摇晃标准差 (Sway Std): {metrics['com_sway_std']}°")
    print(f"🏃 终极 GSI 步态得分: {predicted_gsi} 分")
    print(f"⚠️ 跌倒风险指数: {fall_risk_idx}%")
    print(f"🚨 诊断结论: {risk_level}")
    print("="*50)

    return predicted_gsi

if __name__ == '__main__':
    # 替换为您本地导出的姿态 CSV 测试
    predict_gait_risk("data/output/pepole_2_1_pose.csv")