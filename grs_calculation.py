# ============================================
# Master Thesis Appendix Script
# Geopolitical Risk Assessment Along Subsea Cable Path
# Author: Abel Klaassens
# ============================================

# === SECTION 1: Imports & Setup ===
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import pickle
from rasterio.warp import reproject, Resampling
from rasterio.plot import plotting_extent
from scipy.ndimage import distance_transform_edt
from pyproj import Geod
import geopandas as gpd
from rasterio.features import rasterize
import contextily as ctx
import os

# === SECTION 2: Load Base Data ===
bathymetry_path = "/Users/abelklaassens/Downloads/Master Thesis/A_Python/Master Thesis/D6_2024_resampled_1000m.tif"
shipping_raster_path = "/Users/abelklaassens/Downloads/Master Thesis/A_Python/Master Thesis/2020 All shiptype AIS Shipping Density.tif"
cable_path_file = "/Users/abelklaassens/Downloads/Master Thesis/A_Python/Master Thesis/Results/optimal_path.pkl"
cables_shapefile = "/Users/abelklaassens/Downloads/Master Thesis/A_Python/Master Thesis/Cables_2018.shp"

with rasterio.open(bathymetry_path) as bathy:
    depth_array = bathy.read(1)
    depth_array = np.where(depth_array == bathy.nodata, np.nan, depth_array)

    with open(cable_path_file, "rb") as f:
        path = pickle.load(f)

    with rasterio.open(shipping_raster_path) as src_ship:
        ship_data = src_ship.read(1)
        ship_data = np.where(ship_data < 0, np.nan, ship_data)
        shipping_reprojected = np.zeros(bathy.shape, dtype=np.float32)
        reproject(
            source=ship_data,
            destination=shipping_reprojected,
            src_transform=src_ship.transform,
            src_crs=src_ship.crs,
            dst_transform=bathy.transform,
            dst_crs=bathy.crs,
            resampling=Resampling.nearest
        )
        shipping_data = np.clip(shipping_reprojected, 0, 5000)

# === SECTION 3: Extract Risk Layers Along Path ===
rows, cols = zip(*path)
depth_values = np.array([depth_array[r, c] for r, c in path], dtype=np.float32)
depth_values[np.isnan(depth_values)] = np.nanmedian(depth_values)
depth_norm = 1 - (depth_values - np.nanmin(depth_values)) / (np.nanmax(depth_values) - np.nanmin(depth_values))

shipping_density_values = np.array([shipping_data[r, c] for r, c in path], dtype=np.float32)
shipping_density_values[np.isnan(shipping_density_values)] = 0
log_scaled = np.log1p(shipping_density_values)
shipping_density_norm = (log_scaled - log_scaled.min()) / (log_scaled.max() - log_scaled.min())

plt.figure(figsize=(10, 3))
plt.plot(shipping_density_norm, label="Normalized Shipping Density", color="darkblue")
plt.xlabel("Path Index")
plt.ylabel("Normalized Density [0–1]")
plt.title("Shipping Density Along Cable Route")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(depth_norm, label="Normalized Depth Risk", color="teal")
plt.xlabel("Path Index")
plt.ylabel("Depth Risk [0–1]")
plt.title("Normalized Depth-Based Risk Along Cable Route")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

extent = plotting_extent(bathy)
fig, ax = plt.subplots(figsize=(12, 10))
img = ax.imshow(depth_array, cmap="Blues_r", origin="upper", extent=extent)
ax.set_title("Water Depth (Bathymetry) with Cable Path Overlay")
fig.colorbar(img, ax=ax, label="Depth (m)")
for i, (r, c) in enumerate(path):
    lon, lat = bathy.xy(r, c)
    ax.plot(lon, lat, marker='o', markersize=2, color='cyan')
    if i % max(len(path) // 25, 1) == 0:
        ax.text(lon + 0.01, lat, str(i), color='black', fontsize=6)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_axis_on()
plt.tight_layout()
plt.show()

land_mask = np.isnan(depth_array)
distance_pixels = distance_transform_edt(~land_mask)
distance_from_shore_norm = np.clip((distance_pixels * 1000) / 50000, 0, 1)
shore_distance_along_path = np.array([distance_from_shore_norm[r, c] for r, c in path])

plt.figure(figsize=(10, 3))
plt.plot(shore_distance_along_path, label="Normalized Distance from Shore", color="darkgreen")
plt.xlabel("Path Index")
plt.ylabel("Normalized Distance [0–1]")
plt.title("Distance from Shore Along Cable Route")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

east_risk = np.zeros_like(depth_array)
for r, c in zip(*np.where(~np.isnan(depth_array))):
    lon, _ = bathy.xy(r, c)
    if lon < 18:
        risk = 0.0
    elif lon <= 21:
        risk = (lon - 18) / 3
    else:
        risk = 1.0
    east_risk[r, c] = risk
eastern_risk_values = np.array([east_risk[r, c] for r, c in path], dtype=np.float32)

plt.figure(figsize=(10, 3))
plt.plot(eastern_risk_values, color="purple", label="Eastern Risk Factor")
plt.xlabel("Path Index")
plt.ylabel("Eastern Risk [0–1]")
plt.title("Eastern Geopolitical Exposure Along Cable Route")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

cables_gdf = gpd.read_file(cables_shapefile)
cables_gdf = cables_gdf.to_crs(bathy.crs)
cables_proj = cables_gdf.to_crs(epsg=3035)
buffered = cables_proj.buffer(5000).to_crs(bathy.crs)
shapes = [(geom, 1) for geom in buffered if geom.is_valid]
cable_cluster_raster = rasterize(shapes, out_shape=depth_array.shape, transform=bathy.transform, fill=0, all_touched=True, dtype="int32")
cable_cluster_factor = (cable_cluster_raster > 0).astype(float)
cluster_risk_values = np.array([cable_cluster_factor[r, c] for r, c in path], dtype=np.float32)

plt.figure(figsize=(10, 3))
plt.plot(cluster_risk_values, color="darkred", label="Cable Clustering Factor")
plt.xlabel("Path Index")
plt.ylabel("Cluster Risk [0–1]")
plt.title("Cable Clustering Risk Along Cable Route")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

weights = {"depth": 0.2, "shipping": 0.2, "shore": 0.2, "eastern": 0.2, "clustering": 0.2}

grp = (
    weights["depth"] * depth_norm +
    weights["shipping"] * shipping_density_norm +
    weights["shore"] * shore_distance_along_path +
    weights["eastern"] * eastern_risk_values +
    weights["clustering"] * cluster_risk_values
)

plt.figure(figsize=(10, 3))
plt.plot(grp, color="black", label="Composite Geopolitical Risk Score")
plt.xlabel("Path Index")
plt.ylabel("Risk Score [0–1]")
plt.title("Weighted Composite Geopolitical Risk Along Cable Route")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

output_path = "/Users/abelklaassens/Downloads/Master Thesis/A_Python/Master Thesis/Results/pareto_paths/geopolitical_risk_score.pkl"
risk_along_path = [
    {"index": i, "row": r, "col": c, "score": float(score)}
    for i, ((r, c), score) in enumerate(zip(path, grp))
]
with open(output_path, "wb") as f:
    pickle.dump(risk_along_path, f)
print(f"\n✅ Saved {len(risk_along_path)} risk scores to:\n{output_path}")