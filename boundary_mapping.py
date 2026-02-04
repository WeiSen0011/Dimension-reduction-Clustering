# -*- coding: utf-8 -*-
"""
Cluster boundary generation and PyGMT mapping（仅在 Indian Ocean shp 区域内插值）
插值算法已改为 KDE，bandwidth（sigma）自动选择。
"""

import os
import warnings
warnings.filterwarnings('ignore')

# 可选：根据本机 conda 环境路径设置 GDAL_DATA，按需调整路径
if 'GDAL_DATA' not in os.environ:
    os.environ['GDAL_DATA'] = r"D:\_00_anaconda3\envs\ML\Library\share\gdal"

import pandas as pd
import numpy as np
import xarray as xr
import pygmt
import geopandas as gpd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import scipy.ndimage as ndi
from shapely.geometry import Point, Polygon, box, MultiPolygon
from shapely.prepared import prep
from scipy.special import logsumexp
from shapely.ops import unary_union
import fiona
from fiona.crs import from_epsg

# ==========================================
# 1. 数据读取与预处理
# ==========================================
def load_cluster_data(file_path):
    """从聚类结果文件中加载数据"""
    try:
        df = pd.read_csv(file_path)
        # 检查必要的列是否存在
        required_cols = ['LATITUDE', 'LONGITUDE', 'Cluster_k5']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # 尝试查找聚类标签列
            cluster_cols = [col for col in df.columns if col.startswith('Cluster_k')]
            if cluster_cols:
                df['Cluster_k5'] = df[cluster_cols[0]]
            else:
                raise ValueError(f"缺失必要的列: {missing_cols}")
        
        df_clean = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'Cluster_k5']).copy()
        df_clean['LONGITUDE'] = pd.to_numeric(df_clean['LONGITUDE'], errors='coerce')
        df_clean['LATITUDE'] = pd.to_numeric(df_clean['LATITUDE'], errors='coerce')
        df_clean['Cluster_k5'] = pd.to_numeric(df_clean['Cluster_k5'], errors='coerce')
        df_clean = df_clean.dropna(subset=['LONGITUDE', 'LATITUDE', 'Cluster_k5'])
        df_clean['Cluster_k5'] = df_clean['Cluster_k5'].astype(int)
        return df_clean
    except Exception as e:
        print(f"读取聚类数据失败: {e}")
        return None

# ==========================================
# helper: 读取并准备 Indian Ocean 面掩模
# ==========================================
def load_mask_polygon(indian_shp):
    """加载印度洋区域掩模"""
    if indian_shp is None or not os.path.exists(indian_shp):
        return None, None
    try:
        gdf = gpd.read_file(indian_shp).to_crs("EPSG:4326")
        if hasattr(gdf, "union_all"):
            geom = gdf.union_all()
        else:
            geom = gdf.unary_union
        return prep(geom), geom
    except Exception as e:
        print(f"读取 Indian shp 失败: {e}")
        return None, None

# ==========================================
# KDE bandwidth 自动选择（per-class）
# ==========================================
def estimate_bandwidth_by_rule(coords):
    """基于规则的带宽估计"""
    n, d = coords.shape
    factor = n ** (-1.0 / (d + 4.0))
    std = np.mean(np.std(coords, axis=0))
    bw = factor * std
    return float(max(bw, 1e-3))

def select_bandwidth(coords, bw_candidates=None, min_samples_for_gridsearch=20):
    """自动选择最佳带宽"""
    n = coords.shape[0]
    if n < 3:
        return 1.0
    if bw_candidates is None:
        bw_candidates = np.logspace(np.log10(0.1), np.log10(5.0), 8)
    if n < min_samples_for_gridsearch:
        return estimate_bandwidth_by_rule(coords)
    try:
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bw_candidates}, cv=5)
        grid.fit(coords)
        return float(grid.best_params_['bandwidth'])
    except Exception:
        return estimate_bandwidth_by_rule(coords)

# ==========================================
# 2. KDE-based 分类与网格化
# ==========================================
def generate_cluster_grid(df, region, resolution=1.0, indian_shp=None, min_points_per_class=3):
    """
    使用 KDE 在网格上估计每类密度并赋类（取最大后验），
    仅对 indian_shp 区域进行计算，区域外为 NaN。
    """
    coords_all = df[['LONGITUDE', 'LATITUDE']].values
    classes = np.unique(df['Cluster_k5'].astype(int))
    if classes.size == 0:
        raise ValueError("没有类别数据")

    lon_range = np.arange(region[0], region[1] + resolution, resolution)
    lat_range = np.arange(region[2], region[3] + resolution, resolution)
    xx, yy = np.meshgrid(lon_range, lat_range)
    pts_grid = np.vstack([xx.ravel(), yy.ravel()]).T

    mask_prepared, _ = load_mask_polygon(indian_shp)
    if mask_prepared is not None:
        mask_flat = np.array([mask_prepared.contains(Point(lon, lat)) for lon, lat in pts_grid], dtype=bool)
    else:
        mask_flat = np.ones(len(pts_grid), dtype=bool)

    # 为每个类拟合 KDE
    logdens = np.full((len(classes), len(pts_grid)), -np.inf, dtype=float)
    for i, c in enumerate(classes):
        sub = df[df['Cluster_k5'] == int(c)][['LONGITUDE', 'LATITUDE']].dropna().values
        if sub.shape[0] < min_points_per_class:
            continue
        bw = select_bandwidth(sub)
        try:
            kde = KernelDensity(bandwidth=bw, kernel='gaussian')
            kde.fit(sub)
            if mask_flat.any():
                logd = kde.score_samples(pts_grid[mask_flat])
                logdens[i, mask_flat] = logd
        except Exception:
            bw2 = estimate_bandwidth_by_rule(sub)
            kde = KernelDensity(bandwidth=bw2, kernel='gaussian')
            kde.fit(sub)
            if mask_flat.any():
                logd = kde.score_samples(pts_grid[mask_flat])
                logdens[i, mask_flat] = logd

    # 计算概率并赋类
    with np.errstate(divide='ignore', invalid='ignore'):
        lse = logsumexp(logdens, axis=0)
        probs = np.exp(logdens - lse[np.newaxis, :])
        probs[:, ~np.isfinite(lse)] = 0.0

    idx = np.argmax(probs, axis=0)
    chosen = np.full(len(pts_grid), np.nan, dtype=float)
    valid = (probs.sum(axis=0) > 0) & mask_flat  # 只在 mask 内有效
    chosen[valid] = classes[idx[valid]]

    Z = chosen.reshape(xx.shape)
    da = xr.DataArray(
        data=Z,
        coords={"lat": lat_range, "lon": lon_range},
        dims=("lat", "lon"),
        name="Cluster_KDE",
        attrs={"long_name": "KDE-based Geochemical Cluster ID"}
    )
    return da

# ==========================================
# 3. 提取矢量边界并转换为 shapely polygons
# ==========================================
def extract_cluster_polygons(grid_da, min_area=0.1):
    """
    从网格中提取每个类别的多边形，并过滤小区域
    """
    data = grid_da.values
    lons = grid_da.lon.values
    lats = grid_da.lat.values
    
    # 计算每个网格单元的面积（近似）
    dlon = lons[1] - lons[0] if len(lons) > 1 else 1.0
    dlat = lats[1] - lats[0] if len(lats) > 1 else 1.0
    cell_area = abs(dlon * dlat) * 111 * 111  # 近似 km²
    
    polygons_dict = {}
    unique_clusters = np.unique(data[~np.isnan(data)])
    
    for val in unique_clusters:
        binary_grid = np.where(data == val, 1, 0).astype(np.float32)
        
        # 找到轮廓
        contours = measure.find_contours(binary_grid, 0.5)
        
        polygons = []
        for contour in contours:
            if len(contour) < 3:
                continue
                
            # 转换坐标
            row_idx = contour[:, 0]
            col_idx = contour[:, 1]
            lat_coords = lats[0] + row_idx * (lats[1] - lats[0])
            lon_coords = lons[0] + col_idx * (lons[1] - lons[0])
            
            # 创建多边形
            poly_coords = list(zip(lon_coords, lat_coords))
            if len(poly_coords) >= 3:
                polygon = Polygon(poly_coords)
                if polygon.is_valid and polygon.area > min_area * cell_area:
                    polygons.append(polygon)
        
        # 合并相邻多边形
        if polygons:
            merged = unary_union(polygons)
            if merged.is_empty:
                continue
            if isinstance(merged, Polygon):
                polygons_dict[int(val)] = [merged]
            else:  # MultiPolygon
                polygons_dict[int(val)] = list(merged.geoms)
        else:
            polygons_dict[int(val)] = []
    
    return polygons_dict

# ==========================================
# 保存边界为 shapefile
# ==========================================
def save_boundaries_shapefile(polygons_dict, output_shp):
    """
    将多边形保存为 shapefile 格式
    """
    from shapely.geometry import mapping
    
    schema = {
        'geometry': 'Polygon',
        'properties': {'cluster_id': 'int'}
    }
    
    with fiona.open(output_shp, 'w', driver='ESRI Shapefile', 
                   schema=schema, crs=from_epsg(4326)) as c:
        for cls, polygons in polygons_dict.items():
            for polygon in polygons:
                c.write({
                    'geometry': mapping(polygon),
                    'properties': {'cluster_id': int(cls)}
                })
    
    print(f"边界 shapefile 已保存: {output_shp}")

# ==========================================
# 4. PyGMT 绘图主程序（使用矢量多边形）
# ==========================================
def plot_pygmt_map(df, grid_da, region, indian_shp=None, output_img="Indian_Ocean_Map.svg",
                   smooth_sigma=0.5, majority_size=2):
    """
    使用矢量多边形绘制分类区域，确保颜色与点一致
    """
    # 平滑处理
    if smooth_sigma > 0:
        data = grid_da.values
        mask = ~np.isnan(data)
        classes = np.unique(data[mask]) if mask.any() else np.array([])
        if classes.size > 0:
            one_hot = np.stack([((data == c) & mask).astype(float) for c in classes], axis=0)
            smoothed = ndi.gaussian_filter(one_hot, sigma=(0, smooth_sigma, smooth_sigma), mode='nearest')
            idx = np.argmax(smoothed, axis=0)
            new_data = np.full(data.shape, np.nan, dtype=float)
            new_data[mask] = classes[idx[mask]]
            grid_da = xr.DataArray(new_data, coords=grid_da.coords, dims=grid_da.dims, name=grid_da.name)
    
    # 提取多边形
    polygons_dict = extract_cluster_polygons(grid_da, min_area=0.1)
    
    # 确定所有类别
    grid_classes = list(polygons_dict.keys())
    sample_classes = np.unique(df['Cluster_k5'].dropna().astype(int).values)
    all_classes = sorted(set(list(grid_classes) + list(sample_classes)))
    
    # 创建颜色映射
    ncls = max(1, len(all_classes))
    cmap = plt.get_cmap("tab10", ncls)
    colors_map = {cls: to_hex(cmap(i % 10)) for i, cls in enumerate(all_classes)}
    
    # 创建 PyGMT 图形
    proj = "W70/20c"  # Mollweide 投影
    fig = pygmt.Figure()
    
    # 底图
    fig.basemap(region=region, projection=proj, frame=["afg", "+tIndian Ocean Geochemical Provinces"])
    fig.coast(region=region, projection=proj, shorelines="0.5p,black", area_thresh=10000, land="gray90", water="#a1defd")
    
    # 绘制 Indian Ocean 区域外的灰色区域
    _, indian_geom = load_mask_polygon(indian_shp)
    if indian_geom is not None:
        bbox_poly = box(region[0], region[2], region[1], region[3])
        outside_poly = bbox_poly.difference(indian_geom)
        
        if not outside_poly.is_empty:
            if outside_poly.geom_type == "Polygon":
                xs, ys = outside_poly.exterior.xy
                fig.plot(x=list(xs), y=list(ys), pen="0.1p,black", fill="#d9d9d9")
                for interior in outside_poly.interiors:
                    xi, yi = interior.xy
                    fig.plot(x=list(xi), y=list(yi), pen="0.1p,black", fill="white")
            else:
                for poly in outside_poly.geoms:
                    xs, ys = poly.exterior.xy
                    fig.plot(x=list(xs), y=list(ys), pen="0.1p,black", fill="#d9d9d9")
                    for interior in poly.interiors:
                        xi, yi = interior.xy
                        fig.plot(x=list(xi), y=list(yi), pen="0.1p,black", fill="white")
    
    # 绘制分类区域（矢量多边形）
    for cls, polygons in polygons_dict.items():
        if not polygons:
            continue
        
        color = colors_map.get(cls, "#808080")
        for polygon in polygons:
            if polygon.geom_type == 'Polygon':
                xs, ys = polygon.exterior.xy
                fig.plot(x=list(xs), y=list(ys), pen="0.5p,black", fill=color)
                
                # 绘制内部空洞（如果有）
                for interior in polygon.interiors:
                    xi, yi = interior.xy
                    fig.plot(x=list(xi), y=list(yi), pen="0.5p,black", fill="white")
    
    # 绘制采样点
    _, indian_geom = load_mask_polygon(indian_shp)
    if indian_geom is not None:
        df_plot = df[df.apply(lambda r: indian_geom.contains(Point(r['LONGITUDE'], r['LATITUDE'])), axis=1)]
    else:
        df_plot = df
    
    for cls in all_classes:
        sub = df_plot[df_plot['Cluster_k5'] == int(cls)]
        if sub.empty:
            continue
        color = colors_map.get(cls, "#808080")
        fig.plot(
            x=sub['LONGITUDE'].values,
            y=sub['LATITUDE'].values,
            style="c0.12c",
            pen="0.3p,black",
            fill=color
        )
    
    # 绘制边界
    for cls, polygons in polygons_dict.items():
        color = colors_map.get(cls, "#808080")
        for polygon in polygons:
            if polygon.geom_type == 'Polygon':
                xs, ys = polygon.exterior.xy
                fig.plot(x=list(xs), y=list(ys), pen="1p,black")
                for interior in polygon.interiors:
                    xi, yi = interior.xy
                    fig.plot(x=list(xi), y=list(yi), pen="1p,black")
    
    # 图例
    legend_items = []
    for cls in all_classes:
        color = colors_map.get(cls, "#808080")
        legend_items.append(f"S 0.28c c {color} 0.3p,black {cls}")
    
    try:
        fig.legend(items=legend_items, position="JTR+o0.2c", box=True)
    except Exception:
        y0 = region[2] + (region[3]-region[2])*0.05
        x0 = region[0] + (region[1]-region[0])*0.65
        dx = 0.03*(region[1]-region[0])
        for i, cls in enumerate(all_classes):
            xx = x0
            yy = y0 + i*(region[3]-region[2])*0.03
            color = colors_map.get(cls, "#808080")
            fig.plot(x=xx, y=yy, style="c0.14c", pen="0.3p,black", fill=color)
            fig.text(x=xx + dx, y=yy, text=str(int(cls)), font="10p,Helvetica,black", justify="LM")
    
    # 保存为 SVG 格式
    fig.savefig(output_img, dpi=300)
    
    # 保存边界为 shapefile 格式
    base_name = os.path.splitext(output_img)[0]
    shp_path = f"{base_name}_boundaries.shp"
    save_boundaries_shapefile(polygons_dict, shp_path)
    
    print(f"地图绘制完成并保存: {output_img}")
    print(f"边界 shapefile 已保存: {shp_path}")

# ==========================================
# 主函数
# ==========================================
if __name__ == "__main__":
    # 配置路径
    base_path = "E:/_00_Master_research/_0_Research_data/"
    
    # 输入文件：聚类结果
    input_csv = os.path.join(base_path, "_1_EarthData_IO/OIBID/_fig/weighted_results/final_data_with_labels.csv")
    
    # 输出目录
    output_dir = os.path.join(base_path, "_1_EarthData_IO/OIBID/_fig/boundary_maps/")
    os.makedirs(output_dir, exist_ok=True)

    # Indian shp 路径
    indian_shp = os.path.join(base_path, "GEBCO_2024/shp/IndianOcean_Moll.shp")

    # 区域边界
    region_bounds = [-5, 130, -67, 20]

    # 加载聚类数据
    df = load_cluster_data(input_csv)
    if df is not None:
        print(f"数据加载成功，共 {len(df)} 个样本")
        print(f"聚类类别: {sorted(df['Cluster_k5'].unique())}")
        
        # 生成网格
        grid_da = generate_cluster_grid(df, region_bounds, resolution=0.1, 
                                       indian_shp=indian_shp, min_points_per_class=3)
        
        # 输出文件路径
        out_img = os.path.join(output_dir, "Indian_Ocean_Clusters.svg")
        
        # 绘制地图
        plot_pygmt_map(df, grid_da, region_bounds, 
                      indian_shp=indian_shp,
                      output_img=out_img,
                      smooth_sigma=0.5,
                      majority_size=2)
        
        print("边界划分完成！")
    else:
        print("无法加载聚类数据，请先运行主聚类程序")
