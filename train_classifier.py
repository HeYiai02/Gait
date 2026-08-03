import os
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def clean_gait_data(df, feature_cols, target_col='gait_risk_label'):
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    if target_col not in df.columns:
        if 'gsi_score' in df.columns:
            df[target_col] = df['gsi_score'].apply(lambda score: 0 if score >= 70.0 else 1)
        elif 'risk_level' in df.columns:
            df[target_col] = df['risk_level'].apply(lambda lvl: 0 if 'Level-1' in str(lvl) else 1)
        else:
            raise KeyError("未在 CSV 中找到 gsi_score 或 risk_level")
    return df

def build_gait_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=30,          
            max_depth=3,              
            class_weight='balanced',  
            random_state=42
        ))
    ])

def train_and_evaluate_gait_model(
    features_csv_path="data/output/gait_medical_features.csv",
    output_dir="data/evaluation",
    model_save_path="models/fall_risk_classifier.joblib"
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path) or '.', exist_ok=True)

    if not os.path.exists(features_csv_path):
        raise FileNotFoundError(f"未找到步态特征文件: {features_csv_path}")

    df = pd.read_csv(features_csv_path)

    # 【同步升级】：剔除受 2D 空间畸变污染的 step_cv 与 step_mean，
    # 纯粹采用“时间频率”与“解剖学角度”指标训练模型，彻底抵御视角干扰！
    candidate_features = [
        'stride_time_cv',        # 核心：步频节奏变异（不受透视影响）
        'gai_asymmetry',         # 核心：步态不对称指数
        'double_support_ratio',  # 核心：支撑相/拖步评估
        'com_sway_std',          # 核心：正面平衡评估
        'knee_rom'               # 核心：侧面下肢僵硬评估
    ]
    feature_cols = [col for col in candidate_features if col in df.columns]

    df = clean_gait_data(df, feature_cols)
    print(f"📊 成功载入 {len(df)} 条样本 | 特征列: {feature_cols}")

    X = df[feature_cols].values
    y = df['gait_risk_label'].values

    loo = LeaveOneOut()
    y_true, y_pred, y_scores = [], [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline = build_gait_pipeline()
        pipeline.fit(X_train, y_train)

        pred = pipeline.predict(X_test)
        
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_test)[0]
            classifier = pipeline.named_steps['classifier']
            classes = classifier.classes_
            if 1 in classes:
                prob = probs[list(classes).index(1)]
            else:
                prob = 0.0 if classes[0] == 0 else 1.0
        else:
            prob = float(pred[0])

        y_true.append(y_test[0])
        y_pred.append(pred[0])
        y_scores.append(prob)

    acc = accuracy_score(y_true, y_pred)
    print("="*60)
    print(f"🎉 步态退化风险分类 LOOCV 准确率: {acc * 100:.2f}%")
    print("="*60)

    final_pipeline = build_gait_pipeline()
    final_pipeline.fit(X, y)

    joblib.dump({
        'pipeline': final_pipeline,
        'feature_names': feature_cols,
        'model_type': 'Gait_Risk_Classifier'
    }, model_save_path)
    print(f"✅ 视角适应步态评估模型权重已导出至: {model_save_path}")

if __name__ == '__main__':
    train_and_evaluate_gait_model()