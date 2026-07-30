import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc

# 设置 Matplotlib 学术画图风格
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def run_thesis_experiments(features_csv="data/output/gait_medical_features.csv", output_dir="data/thesis_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(features_csv):
        print(f"⚠️ 未找到特征数据文件: {features_csv}，请先运行 medical_metrics.py 导出特征矩阵！")
        return

    df = pd.read_csv(features_csv)
    print(f"📊 成功载入 {len(df)} 条样本进行论文实验分析...")

    # 1. 自动根据文件名/已知标签划分为 健康对照组 (Healthy) 与 异常患病组 (Abnormal)
    def label_group(row):
        vname = str(row['video_name']).lower()
        # if any(x in vname for x in ['antalgic', 'cerebralPalsy', 'myopathic', 'parkinsons', 'abnormal', 'stroke']):
        if any(x in vname for x in ['abnormal']):
            return 'Abnormal Group'
        return 'Healthy Baseline'

    df['group'] = df.apply(label_group, axis=1)
    
    healthy_df = df[df['group'] == 'Healthy Baseline']

    # healthy_df = healthy_df[healthy_df['gsi_score'] >= 75 ]
    # 标准统计学 IQR 剔除离群点示例 (符合统计学规范)
    # Q1 = healthy_df['gsi_score'].quantile(0.25)
    # Q3 = healthy_df['gsi_score'].quantile(0.75)
    # IQR = Q3 - Q1
    # #仅剔除低于 (Q1 - 1.5 * IQR) 的极端离群噪声
    # healthy_df = healthy_df[healthy_df['gsi_score'] >= (Q1 - 1.5 * IQR)]
    # print(f" 健康对照组: gsi_score>={(Q1 - 1.5 * IQR)}")
    abnormal_df = df[df['group'] == 'Abnormal Group']
    
    print(f"  └─ 健康对照组: {len(healthy_df)} 样本 | 患病异常组: {len(abnormal_df)} 样本")

    # 2. 统计学显著性检验 ($t$-test & Mann-Whitney U test)
    metrics_to_test = ['gsi_score', 'step_cv', 'stride_time_cv', 'gai_asymmetry', 'double_support_ratio', 'knee_rom']
    stats_summary = []

    for metric in metrics_to_test:
        if metric not in df.columns:
            continue
            
        h_vals = healthy_df[metric].dropna()
        a_vals = abnormal_df[metric].dropna()
        
        if len(h_vals) == 0 or len(a_vals) == 0:
            continue

        h_mean, h_std = np.mean(h_vals), np.std(h_vals)
        a_mean, a_std = np.mean(a_vals), np.std(a_vals)
        
        # 统计检验
        if len(h_vals) >= 3 and len(a_vals) >= 3:
            u_stat, p_val = stats.mannwhitneyu(h_vals, a_vals, alternative='two-sided')
        else:
            p_val = np.nan

        stats_summary.append({
            'Metric': metric,
            'Healthy (Mean ± SD)': f"{h_mean:.2f} ± {h_std:.2f}",
            'Abnormal (Mean ± SD)': f"{a_mean:.2f} ± {a_std:.2f}",
            'p-value': f"{p_val:.4f}" if not np.isnan(p_val) else "N/A",
            'Significance': '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'NS'))
        })

    df_stats = pd.DataFrame(stats_summary)
    stats_csv = os.path.join(output_dir, "statistical_significance_table.csv")
    df_stats.to_csv(stats_csv, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*70)
    print("📈 【论文表 4-1】两组生物力学指标差异与显著性检验 ($p$-value Table):")
    print("="*70)
    print(df_stats.to_string(index=False))
    print("="*70)

    # 3. 绘制三合一学术图表：Boxplot 比对 + ROC 曲线
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # 图 A: GSI 稳定性得分箱线图 (已修复 Warning: 显式赋值 hue='group' 并设置 legend=False)
    sns.boxplot(
        x='group', y='gsi_score', data=df, ax=axes[0], 
        hue='group', palette=['#2ecc71', '#e74c3c'], legend=False, width=0.4
    )
    sns.stripplot(x='group', y='gsi_score', data=df, ax=axes[0], color='black', alpha=0.6, jitter=0.2)
    axes[0].set_title('A. GSI Score Comparison', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Gait Stability Index (GSI)', fontsize=10)
    axes[0].set_xlabel('')

    # 图 B: 步长变异系数 (Step CV) 箱线图 (已修复 Warning: 显式赋值 hue='group' 并设置 legend=False)
    sns.boxplot(
        x='group', y='step_cv', data=df, ax=axes[1], 
        hue='group', palette=['#2ecc71', '#e74c3c'], legend=False, width=0.4
    )
    sns.stripplot(x='group', y='step_cv', data=df, ax=axes[1], color='black', alpha=0.6, jitter=0.2)
    axes[1].set_title('B. Step Length Variation (Step CV)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Step CV', fontsize=10)
    axes[1].set_xlabel('')

    # 图 C: GSI 得分的 ROC 曲线与 AUC
    if len(healthy_df) > 0 and len(abnormal_df) > 0:
        y_true = np.array([1 if g == 'Abnormal Group' else 0 for g in df['group']])
        y_scores = -df['gsi_score'].values

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        axes[2].plot(fpr, tpr, color='#e74c3c', lw=2.5, label=f'GSI Indicator (AUC = {roc_auc:.3f})')
        axes[2].plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
        axes[2].set_xlim([0.0, 1.0])
        axes[2].set_ylim([0.0, 1.05])
        axes[2].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
        axes[2].set_ylabel('True Positive Rate (Sensitivity)', fontsize=10)
        axes[2].set_title('C. ROC Curve for Gait Risk Detection', fontsize=12, fontweight='bold')
        axes[2].legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "thesis_experimental_figures.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    print(f"\n🖼️ 论文高质量矢量图表已成功导出至: {chart_path}")

if __name__ == '__main__':
    run_thesis_experiments()