# -*- coding: utf-8 -*-
"""
自适应特征加权UMAP降维与层次聚类分析（主程序）
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import umap

# ===================== 参数 =====================
MAX_CLUSTERS = 10
COLOR_PALETTE = sns.color_palette("deep", MAX_CLUSTERS + 1)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 工具函数 =====================
def build_ratios(data):
    """从原始数据构建同位素比值"""
    data['Pb206_204'] = data.get("Pb206/Pb204", np.nan)
    data['Pb207_204'] = data.get("Pb207/Pb204", np.nan)
    data['Pb208_204'] = data.get("Pb208/Pb204", np.nan)
    data['Sr87_86'] = data.get("Sr87/Sr86", np.nan)
    data['Nd143_144'] = data.get("Nd143/Nd144", np.nan)
    return data


def scale_features(df, cols, method="standard"):
    """标准化/归一化"""
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    scaler = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "minmax": MinMaxScaler(),
    }[method]
    scaled_data = scaler.fit_transform(df[cols])
    df_scaled = pd.DataFrame(scaled_data, columns=cols, index=df.index)
    return df_scaled


def save_vector_plot(fig, path):
    """保存为矢量图"""
    try:
        fig.savefig(path, format='svg', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"保存图像失败: {path}, 错误: {e}")


def generate_feature_weights(df, features, core_feature_weights, method="spearman"):
    """
    根据核心特征与其他特征的相关性，自适应生成权重。
    - 核心特征权重由 core_feature_weights 字典指定
    - 其他特征的权重 = 1 / (1 + 与核心特征的平均相关性)
    """
    corr_matrix = df[features].corr(method=method).abs()
    weights = {}

    for f in features:
        if f in core_feature_weights:
            weights[f] = core_feature_weights[f]
        else:
            corr_with_core = corr_matrix.loc[f, core_feature_weights.keys()].mean()
            weights[f] = 1 / (1 + corr_with_core)
    return weights


def plot_grid_search_results(result_df, save_folder=None):
    """绘制网格搜索结果：热力图（n_neighbors vs n_clusters）和按 n_neighbors 的折线图。"""
    if save_folder is None:
        save_folder = os.getcwd()
    try:
        os.makedirs(save_folder, exist_ok=True)
    except Exception:
        pass

    # 准备透视表
    pivot = result_df.pivot_table(index='n_clusters', columns='n_neighbors', values='composite_score')

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, cmap='viridis', annot=False, ax=ax)
    ax.set_title('Composite Score Heatmap (n_clusters vs n_neighbors)')
    ax.set_xlabel('n_neighbors')
    ax.set_ylabel('n_clusters')
    plt.tight_layout()
    heatmap_path = os.path.join(save_folder, 'grid_search_composite_heatmap.svg')
    save_vector_plot(fig, heatmap_path)
    plt.close(fig)

    # Line plot: composite vs n_neighbors for each k
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in sorted(result_df['n_clusters'].unique()):
        sub = result_df[result_df['n_clusters'] == k].sort_values('n_neighbors')
        ax.plot(sub['n_neighbors'], sub['composite_score'], marker='o', label=f'k={k}')
    ax.set_xlabel('n_neighbors')
    ax.set_ylabel('composite_score')
    ax.set_title('Composite Score vs n_neighbors for different k')
    ax.legend(title='n_clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    line_path = os.path.join(save_folder, 'composite_vs_nneighbors_by_k.svg')
    save_vector_plot(fig, line_path)
    plt.close(fig)

    # Metrics trends: silhouette, calinski_harabasz, davies_bouldin
    metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for idx, metric in enumerate(metrics):
        axm = axes[idx]
        for k in sorted(result_df['n_clusters'].unique()):
            sub = result_df[result_df['n_clusters'] == k].sort_values('n_neighbors')
            axm.plot(sub['n_neighbors'], sub[metric], marker='.', label=f'k={k}')
        axm.set_xlabel('n_neighbors')
        axm.set_title(metric)
        axm.grid(alpha=0.3)
        if idx == 0:
            axm.legend(title='n_clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    metrics_path = os.path.join(save_folder, 'metrics_trends_by_k.svg')
    save_vector_plot(fig, metrics_path)
    plt.close(fig)

    print(f"已生成寻优图并保存到: {save_folder}")


def plot_min_dist_nneighbors_heatmaps(result_df, k_values, save_folder=None):
    """为每个 k 绘制 n_neighbors vs min_dist 的 composite_score 热力图"""
    if save_folder is None:
        save_folder = os.getcwd()
    os.makedirs(save_folder, exist_ok=True)

    ks = list(k_values)
    n = len(ks)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, k in enumerate(ks):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        sub = result_df[result_df['n_clusters'] == k]
        if sub.empty:
            ax.set_visible(False)
            continue
        pivot = sub.pivot_table(index='min_dist', columns='n_neighbors', values='composite_score')
        sns.heatmap(pivot, ax=ax, cmap='viridis')
        ax.set_title(f'k={k}')
        ax.set_xlabel('n_neighbors')
        ax.set_ylabel('min_dist')

    # 隐藏多余子图
    total = rows * cols
    for idx in range(len(ks), total):
        r = idx // cols
        c = idx % cols
        axes[r][c].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_folder, 'min_dist_nneighbors_heatmaps_by_k.svg')
    save_vector_plot(fig, path)
    plt.close(fig)
    print(f"已生成 n_neighbors vs min_dist 热力图并保存到: {path}")


def summarize_grid_search(result_df, save_folder=None, top_n=5):
    """生成网格搜索的参数候选摘要与解释并保存"""
    if save_folder is None:
        save_folder = os.getcwd()
    os.makedirs(save_folder, exist_ok=True)

    df = result_df.copy()
    # 保存 top_n 组合
    top_df = df.sort_values('composite_score', ascending=False).head(top_n).reset_index(drop=True)
    top_csv = os.path.join(save_folder, 'grid_search_top_combinations.csv')
    top_df.to_csv(top_csv, index=False, encoding='utf-8-sig')

    # 按 k 汇总稳定性
    stability = df.groupby('n_clusters')['composite_score'].agg(['mean', 'std', 'max']).rename(columns={'mean':'mean_comp','std':'std_comp','max':'max_comp'})
    stability['cv'] = stability['std_comp'] / (stability['mean_comp'] + 1e-12)
    stability_csv = os.path.join(save_folder, 'grid_search_stability_by_k.csv')
    stability.to_csv(stability_csv, encoding='utf-8-sig')

    # 对 top 组合进行单项指标一致性检查
    warnings = []
    for i, row in top_df.iterrows():
        k = int(row['n_clusters'])
        nn = int(row['n_neighbors'])
        sil = row['silhouette']
        ch = row['calinski_harabasz']
        db = row['davies_bouldin']
        if db > df['davies_bouldin'].median():
            warnings.append(f"Top#{i+1} (k={k}, n_neighbors={nn}): DB 值偏高 ({db:.3f})，可能簇间分离差或簇形状问题。")
        if sil < df['silhouette'].median():
            warnings.append(f"Top#{i+1} (k={k}, n_neighbors={nn}): Silhouette 值较低 ({sil:.3f})，簇内紧致度可能不足。")

    # 为最佳 k 计算 n_neighbors 的稳定区间
    best_row = df.loc[df['composite_score'].idxmax()]
    best_k = int(best_row['n_clusters'])
    k_group = df[df['n_clusters'] == best_k].copy()
    max_comp = k_group['composite_score'].max()
    plateau = k_group[k_group['composite_score'] >= 0.95 * max_comp].sort_values('n_neighbors')
    plateau_range = (int(plateau['n_neighbors'].min()), int(plateau['n_neighbors'].max())) if not plateau.empty else (int(k_group.loc[k_group['composite_score'].idxmax(),'n_neighbors']),)

    # 写入文本摘要
    summary_lines = []
    summary_lines.append('网格搜索参数汇总与建议')
    summary_lines.append('--------------------------------')
    summary_lines.append(f"最佳复合分参数: n_neighbors={int(best_row['n_neighbors'])}, n_clusters={best_k}, composite_score={best_row['composite_score']:.4f}")
    summary_lines.append('')
    summary_lines.append('Top 参数组合:')
    for i, row in top_df.iterrows():
        summary_lines.append(f"  Top#{i+1}: n_neighbors={int(row['n_neighbors'])}, n_clusters={int(row['n_clusters'])}, composite={row['composite_score']:.4f}, sil={row['silhouette']:.4f}, ch={row['calinski_harabasz']:.2f}, db={row['davies_bouldin']:.4f}")
    summary_lines.append('')
    summary_lines.append('稳定性（按 n_clusters 分组）:')
    for k, r in stability.iterrows():
        summary_lines.append(f"  k={k}: mean_comp={r['mean_comp']:.4f}, std={r['std_comp']:.4f}, cv={r['cv']:.3f}, max={r['max_comp']:.4f}")
    summary_lines.append('')
    if isinstance(plateau_range, tuple) and len(plateau_range) == 2:
        summary_lines.append(f"对于最佳 k={best_k}，n_neighbors 在 {plateau_range[0]} 到 {plateau_range[1]} 之间表现稳定（>=95% max composite）。")
    else:
        summary_lines.append(f"对于最佳 k={best_k}，未发现明显稳定平台，建议以 n_neighbors={int(best_row['n_neighbors'])} 为首选。")
    summary_lines.append('')
    if warnings:
        summary_lines.append('警告与注意事项:')
        for w in warnings:
            summary_lines.append('  - ' + w)
    else:
        summary_lines.append('未发现明显单项指标冲突。')

    summary_text = '\n'.join(summary_lines)
    summary_path = os.path.join(save_folder, 'grid_search_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print('已保存网格搜索摘要与候选组合：')
    print(' -', top_csv)
    print(' -', stability_csv)
    print(' -', summary_path)
    return {'top_csv': top_csv, 'stability_csv': stability_csv, 'summary_txt': summary_path}


# ===================== 主函数 =====================
def umap_hierarchical_analysis(df, save_folder=None, k_values=None, core_feature_weights=None, 
                               preferred_k=None, min_dist_list=None, determine_k_method='rf', rf_cv=5):
    """执行完整的自适应流形聚类与解释流程, 去掉 PCA 降噪"""
    data = build_ratios(df.copy())
    features = ['Sr87/Sr86', 'Nd143/Nd144', 'Pb206/Pb204', 'Pb207/Pb204', 'Pb208/Pb204']
    clean_data = data.dropna(subset=features).reset_index(drop=True)

    if len(clean_data) < 20:
        print("有效样本过少（<20），无法继续分析。")
        return None, None

    # === 标准化 ===
    scaled_df = scale_features(clean_data, features, method="standard")

    # === 应用特征权重（新方法） ===
    feature_weights = None
    # 如果用户未指定核心特征权重，则在寻优过程中自动选择核心特征（按方差）并赋初始权重
    if not core_feature_weights:
        num_core = 2
        variances = scaled_df.var().sort_values(ascending=False)
        auto_cores = list(variances.index[:num_core])
        core_feature_weights = {f: 1.0 for f in auto_cores}
        print(f"\n--- 自动选择核心特征并赋初始权重: {core_feature_weights} ---")
    else:
        print("\n--- 基于核心特征的自适应加权 (用户指定) ---")

    # 生成并应用权重
    feature_weights = generate_feature_weights(scaled_df, features, core_feature_weights)
    for feature, weight in feature_weights.items():
        print(f"  - {feature}: {weight:.3f}")
        scaled_df[feature] *= weight
    print("--------------------------------")

    scaled_data = scaled_df.values  # 不再使用 PCA

    # === 参数搜索 ===
    n_neighbors_list = range(10, 41)
    if min_dist_list is None:
        min_dist_list = [0.01, 0.05, 0.1, 0.15, 0.3, 0.5]
    metric = "minkowski"
    n_clusters_list = list(k_values) if k_values else range(2, MAX_CLUSTERS + 1)

    results = []
    all_cluster_labels = pd.DataFrame(index=clean_data.index)

    print(f"\n--- 开始参数搜索 (n_neighbors: {list(n_neighbors_list)}, min_dist: {min_dist_list}, k: {n_clusters_list}) ---")
    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
            embedding = reducer.fit_transform(scaled_data)
            for k in n_clusters_list:
                clusterer = AgglomerativeClustering(n_clusters=k)
                labels = clusterer.fit_predict(embedding)
                # 仅记录一份 embedding 对应的标签
                all_cluster_labels[f"Cluster_k{k}"] = labels
                results.append({
                    'n_neighbors': n_neighbors, 'min_dist': min_dist, 'n_clusters': k,
                    'silhouette': silhouette_score(embedding, labels),
                    'calinski_harabasz': calinski_harabasz_score(embedding, labels),
                    'davies_bouldin': davies_bouldin_score(embedding, labels)
                })

    result_df = pd.DataFrame(results)
    scaler = MinMaxScaler()
    result_df['sil_norm'] = scaler.fit_transform(result_df[['silhouette']])
    result_df['ch_norm']  = scaler.fit_transform(result_df[['calinski_harabasz']])
    result_df['db_norm']  = 1 - scaler.fit_transform(result_df[['davies_bouldin']])
    result_df['composite_score'] = (result_df['sil_norm'] + result_df['ch_norm'] + result_df['db_norm']) / 3
    best_params_row = result_df.loc[result_df['composite_score'].idxmax()]
    best_n_neighbors = int(best_params_row['n_neighbors'])
    best_min_dist = float(best_params_row.get('min_dist', 0.3))
    best_k_auto = int(best_params_row['n_clusters'])
    
    # 如果用户提供了 preferred_k，则用之覆盖自动选择的 k
    if preferred_k is not None:
        best_k = int(preferred_k)
        print(f"\n搜索完成！自动选择参数: n_neighbors={best_n_neighbors}, k={best_k_auto}")
        print(f"使用用户指定的 preferred_k={best_k} 覆盖自动选择。")
    else:
        best_k = best_k_auto
        print(f"\n搜索完成！最佳参数: n_neighbors={best_n_neighbors}, k={best_k}")

    # === 使用指定方法确定最佳 k ===
    selected_k = best_k
    rf_scores = {}
    if determine_k_method == 'rf':
        for k in n_clusters_list:
            # 选取该 k 的最佳 (n_neighbors, min_dist) 组合
            subk = result_df[result_df['n_clusters'] == k]
            if subk.empty:
                continue
            best_row_k = subk.loc[subk['composite_score'].idxmax()]
            nn_k = int(best_row_k['n_neighbors'])
            md_k = float(best_row_k['min_dist'])
            reducer_k = umap.UMAP(n_components=2, n_neighbors=nn_k, min_dist=md_k, metric=metric, random_state=42)
            emb_k = reducer_k.fit_transform(scaled_data)
            clusterer_k = AgglomerativeClustering(n_clusters=k)
            labels_k = clusterer_k.fit_predict(emb_k)
            # 用随机森林判断用原始加权特征预测簇标签的可分性
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            try:
                scores = cross_val_score(rf, scaled_df.values, labels_k, cv=rf_cv, scoring='accuracy')
                rf_scores[k] = scores.mean()
            except Exception:
                rf_scores[k] = 0.0
        if rf_scores:
            selected_k = max(rf_scores, key=rf_scores.get)
            print(f"基于 RandomForest 的 k 选择结果: {rf_scores}; 选择 k={selected_k}")

    # 如果用户提供 preferred_k，则覆盖
    if preferred_k is not None:
        selected_k = int(preferred_k)
        print(f"使用用户指定的 preferred_k={selected_k} 覆盖选择。")

    # 对最终选择的 k，使用其最佳参数 (n_neighbors, min_dist)
    final_sub = result_df[result_df['n_clusters'] == selected_k]
    if not final_sub.empty:
        final_best_row = final_sub.loc[final_sub['composite_score'].idxmax()]
        final_n_neighbors = int(final_best_row['n_neighbors'])
        final_min_dist = float(final_best_row['min_dist'])
    else:
        final_n_neighbors = best_n_neighbors
        final_min_dist = best_min_dist

    final_reducer = umap.UMAP(n_components=2, n_neighbors=final_n_neighbors, 
                             min_dist=final_min_dist, metric=metric, random_state=42)
    final_embedding = final_reducer.fit_transform(scaled_data)

    final_clusterer = AgglomerativeClustering(n_clusters=selected_k)
    final_labels = final_clusterer.fit_predict(final_embedding)
    all_cluster_labels[f"Cluster_k{selected_k}"] = final_labels
    all_cluster_labels["UMAP-1"] = final_embedding[:, 0]
    all_cluster_labels["UMAP-2"] = final_embedding[:, 1]

    # 保存经纬度信息供后续边界划分使用
    if 'LATITUDE' in clean_data.columns and 'LONGITUDE' in clean_data.columns:
        all_cluster_labels["LATITUDE"] = clean_data["LATITUDE"].values
        all_cluster_labels["LONGITUDE"] = clean_data["LONGITUDE"].values
    
    final_data = pd.concat([clean_data, all_cluster_labels], axis=1)

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        save_subfolder = os.path.join(save_folder, "weighted_results" if core_feature_weights else "unweighted_results")
        os.makedirs(save_subfolder, exist_ok=True)
        final_data.to_csv(os.path.join(save_subfolder, "final_data_with_labels.csv"), index=False, encoding="utf-8-sig")
        result_df.to_csv(os.path.join(save_subfolder, "grid_search_results.csv"), index=False, encoding="utf-8-sig")
        
        # 生成并保存参数寻优的可视化图
        try:
            plot_grid_search_results(result_df, save_subfolder)
        except Exception as e:
            print(f"绘制网格搜索图失败: {e}")
        
        # 生成并保存参数寻优的文字摘要与 top 组合 csv
        try:
            summarize_grid_search(result_df, save_subfolder, top_n=8)
        except Exception as e:
            print(f"生成网格搜索摘要失败: {e}")
        
        # 生成 n_neighbors vs min_dist 的热力图（按 k）
        try:
            plot_min_dist_nneighbors_heatmaps(result_df, n_clusters_list, save_subfolder)
        except Exception as e:
            print(f"绘制 n_neighbors vs min_dist 热力图失败: {e}")

    # === 可视化聚类结果 ===
    fig, ax = plt.subplots(figsize=(10, 8))
    labels_col = final_data.loc[:, f"Cluster_k{selected_k}"]
    if isinstance(labels_col, pd.DataFrame):
        labels_to_plot = labels_col.iloc[:, 0].values
    else:
        labels_to_plot = labels_col.values
    unique_labels = sorted(np.unique(labels_to_plot))
    for lab in unique_labels:
        mask = (labels_to_plot == lab)
        color = COLOR_PALETTE[int(lab) % len(COLOR_PALETTE)]
        ax.scatter(final_embedding[mask, 0], final_embedding[mask, 1], 
                   color=color, label=f'Cluster {lab}', edgecolor='k', s=50, alpha=0.85)
    ax.legend(title='聚类簇', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f"最佳UMAP+层次聚类 (k={selected_k}, n_neighbors={final_n_neighbors})", fontsize=14)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_folder:
        save_vector_plot(fig, os.path.join(save_subfolder, "best_clustering_result.svg"))
    plt.show()
    plt.close(fig)

    # 构建返回的最优参数信息
    best_params_selected = best_params_row.copy()
    best_params_selected['n_clusters'] = selected_k
    best_params_selected['preferred_k_used'] = (preferred_k is not None)
    
    return result_df, best_params_selected.to_dict(), final_data, feature_weights


# ===================== 主程序 =====================
if __name__ == "__main__":
    # 请替换为实际文件路径
    base_path = "E:/"
    input_file = os.path.join(base_path, ".csv")
    output_folder = os.path.join(base_path, "/")

    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在于 '{input_file}'")
    else:
        df = pd.read_csv(input_file, encoding="utf-8", low_memory=False)

        k_to_search = [2, 3, 4, 5, 6, 7, 8, 9, 10]

        # 在此处明确指定核心特征及权重（若希望使用自动选择，可改为 None）
        core_feature_weights = {
            "Pb206/Pb204": 1.0,
            "Sr87/Sr86": 1.0
        }

        T1 = time.time()
        result_df, best_params, final_data, feature_weights = umap_hierarchical_analysis(
            df,
            save_folder=output_folder,
            k_values=k_to_search,
            core_feature_weights=core_feature_weights,
            preferred_k=5
        )
        T2 = time.time()

        if best_params:
            print(f"\n总运行时间: {T2 - T1:.2f} 秒")
            print("--- 最终最优参数 ---")
            print(pd.Series(best_params))
            print("\n--- 特征权重 ---")
            for feature, weight in feature_weights.items():
                print(f"{feature}: {weight:.3f}")
            
            # 提示后续步骤
            print("\n--- 下一步骤 ---")
            print("1. 运行 boundary_mapping.py 进行边界划分")
            print("2. 运行 feature_importance.py 进行特征重要性分析")
