import os
import joblib
import pandas as pd
import numpy as np
from medical_metrics import UltimateSegmentedGaitAnalyzer

def predict_gait_risk(new_pose_csv, model_path="models/fall_risk_classifier.joblib"):
    print(f"\n🎬 正在准备评估姿态文件: {new_pose_csv}")

    # 1. 提取医学特征 (激活视角感知与侧面 Knee ROM 容错)
    analyzer = UltimateSegmentedGaitAnalyzer(raw_video_fps=30.0, min_segment_frames=15)
    metrics = analyzer.process_single_csv(new_pose_csv)
    
    # 2. 响应质量闸口
    if metrics is None:
        print("🛑 [系统提示]: 该视频未能提取到有效直行步态片段（可能原因：帧数不足或无法识别迈步周期）。")
        return None

    # 3. 加载模型或输出临床规则 GSI 得分
    if not os.path.exists(model_path):
        print(f"⚠️ 未找到已训练的模型权重: {model_path}，将直接输出临床规则 GSI 得分。")
        predicted_gsi = metrics['gsi_score']
    else:
        model_data = joblib.load(model_path)
        pipeline = model_data.get('pipeline')
        feature_names = model_data.get('feature_names', [])

        missing_features = [col for col in feature_names if col not in metrics]
        if missing_features or pipeline is None:
            print(f"⚠️ 模型权重文件不匹配（缺少特征 {missing_features}），已安全回退为 GSI 临床规则打分。")
            predicted_gsi = metrics['gsi_score']
        else:
            input_data = np.array([[metrics[col] for col in feature_names]])
            
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(input_data)[0]
                classifier = pipeline.named_steps['classifier'] if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps else pipeline
                classes = classifier.classes_
                if 1 in classes:
                    idx = list(classes).index(1)
                    risk_prob = probs[idx]
                else:
                    risk_prob = 0.0 if classes[0] == 0 else 1.0
                predicted_gsi = round(100.0 - risk_prob * 100.0, 1)
            else:
                label = int(pipeline.predict(input_data)[0])
                predicted_gsi = 85.0 if label == 0 else 40.0

    # 4. 确定预警等级
    fall_risk_idx = round(100.0 - predicted_gsi, 1)
    
    if predicted_gsi >= 75.0:
        risk_level = "Level-1 (低风险/健康)"
    elif predicted_gsi >= 50.0:
        risk_level = "Level-2 (中风险/轻度退化)"
    else:
        risk_level = "Level-3 (高风险/重度失稳)"

    # 5. 视角自适应日志打印
    view_str = "侧面视角 (SIDE)" if metrics['view_type'] == 'SIDE' else ("正面视角 (FRONTAL)" if metrics['view_type'] == 'FRONTAL' else "斜向视角 (OBLIQUE)")
    
    print("\n" + "="*50)
    print(f"📊 分析视频: {metrics['video_name']}")
    print(f"📹 自动识别拍摄视角: {view_str}")
    if metrics['view_type'] == 'SIDE':
        print(f"🦵 膝关节屈伸活动度 (Knee ROM): {metrics['knee_rom']}° (侧面核心指标)")
    else:
        print(f"⚖️ 躯干左右摇晃标准差 (Sway Std): {metrics['com_sway_std']}° (正面核心指标)")
    print(f"📏 平均归一化步长 (Step Mean): {metrics.get('step_mean', 'N/A')}")
    print(f"📈 步长变异系数 (Step CV): {metrics['step_cv']}")
    print(f"⏱️ 步频时间变异 (Stride Time CV): {metrics['stride_time_cv']}")
    print(f"⚖️ 步态不对称度 (GAI): {metrics['gai_asymmetry']}")
    print(f"🏃 终极 GSI 稳定性得分: {predicted_gsi} 分")
    print(f"⚠️ 跌倒风险指数: {fall_risk_idx}%")
    print(f"🚨 诊断结论: {risk_level}")
    print("="*50)

    return predicted_gsi

if __name__ == '__main__':
    predict_gait_risk("data/sample/stroke_1.csv")