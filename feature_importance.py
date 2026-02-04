# -*- coding: utf-8 -*-
"""
特征重要性分析：SHAP + 互信息分析
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import shap
import xgboost

# ===================== 参数 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 工具函数 =====================
def save_vector_plot(fig, path):
    """保存为矢量图"""
    try:
        fig.savefig(path, format='svg', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"保存图像失败: {path}, 错误: {e}")


def analyze_feature_importance_shap_mi(original_data, features, labels, save_folder=None):
    """计算 SHAP + MI 特征贡献度"""
    X = original_data[features]
    y = labels

    if len(np.unique(y)) < 2:
        print("只有一个聚类簇，无法进行特征贡献度分析。")
        return

    print("\n--- 开始特征重要性分析 ---")
    
    # 互信息分析
    print("计算互信息...")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({'feature': features, 'mutual_information': mi_scores}).sort_values('mutual_information', ascending=False)
    print("\n--- 特征贡献度 (互信息) ---")
    print(mi_df)

    # 互信息可视化
    fig_mi, ax_mi = plt.subplots(figsize=(8, 6))
    sns.barplot(x='mutual_information', y='feature', data=mi_df, palette='plasma', ax=ax_mi)
    ax_mi.set_title('特征贡献度 (Mutual Information)', fontsize=14)
    ax_mi.set_xlabel('Mutual Information Score', fontsize=12)
    ax_mi.set_ylabel('特征', fontsize=12)
    plt.tight_layout()
    
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        save_vector_plot(fig_mi, os.path.join(save_folder, "feature_importance_MI.svg"))
    plt.close(fig_mi)

    # SHAP 分析
    print("计算SHAP值...")
    model = xgboost.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', 
                                  use_label_encoder=False, random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print("SHAP 分析完成")

    # SHAP条形图
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("特征贡献度 (SHAP Bar Plot)", fontsize=14)
    plt.tight_layout()
    if save_folder:
        plt.savefig(os.path.join(save_folder, "feature_importance_SHAP_bar.svg"), 
                   format='svg', dpi=300, bbox_inches='tight')
    plt.close()

    # SHAP蜂群图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("特征贡献度 (SHAP Swarm Plot)", fontsize=14)
    plt.tight_layout()
    if save_folder:
        plt.savefig(os.path.join(save_folder, "feature_importance_SHAP_swarm.svg"), 
                   format='svg', dpi=300, bbox_inches='tight')
    plt.close()

    # 保存特征重要性结果到CSV
    if save_folder:
        # 计算SHAP重要性（平均绝对SHAP值）
        if isinstance(shap_values, list):
            # 多分类情况：取所有类别的平均
            shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': features,
            'mutual_information': mi_scores,
            'shap_importance': shap_importance
        })
        importance_df['normalized_mi'] = importance_df['mutual_information'] / importance_df['mutual_information'].sum()
        importance_df['normalized_shap'] = importance_df['shap_importance'] / importance_df['shap_importance'].sum()
        importance_df['combined_score'] = (importance_df['normalized_mi'] + importance_df['normalized_shap']) / 2
        
        importance_df = importance_df.sort_values('combined_score', ascending=False)
        importance_df.to_csv(os.path.join(save_folder, "feature_importance_scores.csv"), 
                           index=False, encoding='utf-8-sig')
        
        print(f"\n特征重要性得分已保存到: {os.path.join(save_folder, 'feature_importance_scores.csv')}")
    
    print("特征重要性分析完成！")
    
    return mi_df, shap_values


def plot_feature_correlation(features_df, labels, save_folder=None):
    """绘制特征与聚类标签的关系图"""
    unique_labels = np.unique(labels)
    n_features = len(features_df.columns)
    
    # 为每个特征绘制箱线图
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 3 * n_features))
    if n_features == 1:
        axes = [axes]
    
    for i, feature in enumerate(features_df.columns):
        data_by_cluster = [features_df[feature][labels == label] for label in unique_labels]
        axes[i].boxplot(data_by_cluster, labels=[f'Cluster {label}' for label in unique_labels])
        axes[i].set_title(f'{feature} 分布 by Cluster', fontsize=12)
        axes[i].set_ylabel(feature, fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_folder:
        save_vector_plot(fig, os.path.join(save_folder, "feature_distribution_by_cluster.svg"))
    
    plt.show()
    plt.close(fig)


# ===================== 主函数 =====================
def run_feature_importance_analysis(cluster_data_path, save_folder=None):
    """运行特征重要性分析"""
    # 加载聚类结果数据
    if not os.path.exists(cluster_data_path):
        print(f"错误: 文件不存在 '{cluster_data_path}'")
        return None
    
    df = pd.read_csv(cluster_data_path, encoding="utf-8-sig")
    
    # 确定特征列和聚类标签列
    feature_cols = ['Sr87/Sr86', 'Nd143/Nd144', 'Pb206/Pb204', 'Pb207/Pb204', 'Pb208/Pb204']
    
    # 检查是否存在这些特征列，如果不存在，尝试其他可能的列名
    available_features = []
    for col in feature_cols:
        if col in df.columns:
            available_features.append(col)
        else:
            # 尝试查找替代列名
            alt_names = [col.replace('/', '_'), f"smoothed_{col}", col.lower()]
            for alt in alt_names:
                if alt in df.columns:
                    available_features.append(alt)
                    break
    
    if len(available_features) < 2:
        print("错误: 数据中特征列不足")
        return None
    
    # 查找聚类标签列
    cluster_cols = [col for col in df.columns if col.startswith('Cluster_k')]
    if not cluster_cols:
        print("错误: 未找到聚类标签列")
        return None
    
    # 使用第一个聚类标签列
    cluster_col = cluster_cols[0]
    
    # 清理数据
    analysis_data = df.dropna(subset=available_features + [cluster_col]).copy()
    
    if len(analysis_data) < 20:
        print("有效样本过少（<20），无法进行特征重要性分析。")
        return None
    
    print(f"分析数据: {len(analysis_data)} 个样本")
    print(f"特征: {available_features}")
    print(f"聚类标签列: {cluster_col}")
    print(f"聚类类别: {sorted(analysis_data[cluster_col].unique())}")
    
    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(analysis_data[available_features])
    scaled_df = pd.DataFrame(scaled_features, columns=available_features, index=analysis_data.index)
    
    # 运行特征重要性分析
    mi_df, shap_values = analyze_feature_importance_shap_mi(
        scaled_df, 
        available_features, 
        analysis_data[cluster_col].values,
        save_folder
    )
    
    # 绘制特征与聚类关系图
    plot_feature_correlation(scaled_df, analysis_data[cluster_col].values, save_folder)
    
    return mi_df, shap_values


# ===================== 主程序 =====================
if __name__ == "__main__":
    # 配置路径
    base_path = "E:/_00_Master_research/_0_Research_data/"
    
    # 输入文件：聚类结果
    input_file = os.path.join(base_path, "_1_EarthData_IO/OIBID/_fig/weighted_results/final_data_with_labels.csv")
    
    # 输出目录
    output_folder = os.path.join(base_path, "_1_EarthData_IO/OIBID/_fig/feature_importance/")
    
    print("开始特征重要性分析...")
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_folder}")
    
    # 运行分析
    results = run_feature_importance_analysis(input_file, output_folder)
    
    if results:
        print("\n特征重要性分析完成！")
    else:
        print("\n特征重要性分析失败")
