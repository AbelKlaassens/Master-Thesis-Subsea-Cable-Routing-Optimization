# ============================================
# Master Thesis Script 1
# Optimal Subsea Cable Routing via TCO Analysis with Visualizations
# Author: Abel Klaassens
# Institution: University of Groningen
# ============================================

# === SECTION 1: Import Required Libraries ===
import geopandas as gpd
import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Point, box, LineString
from pyproj import Geod
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import pandas as pd
from rasterio.features import rasterize
import pickle
import os
import contextily as ctx

# === SECTION 2: Load Bathymetry Data ===
bathymetry_path = "/Users/abelklaassens/Downloads/Master Thesis/A_Python/Master Thesis/D6_2024_resampled_1000m.tif"
bathymetry = rasterio.open(bathymetry_path)
depth_array = bathymetry.read(1)
depth_array = np.where(depth_array == bathymetry.nodata, np.nan, depth_array)
depth_display = depth_array.copy()

# === SECTION 3: Rasterize Fishing Activity (Fault Rate Proxy) ===
fishing_df = pd.read_csv("/Users/abelklaassens/Downloads/Master Thesis/A_Python/Master Thesis/Ship Activity/filtered_baltic/avg_fishing_by_cell_and_gear.csv")
grouped = fishing_df.groupby(["cell_ll_lat", "cell_ll_lon"]).agg({"fishing_hours": "sum"}).reset_index()
cell_size = 0.1
fishing_shapes = [
    (box(lon, lat, lon + cell_size, lat + cell_size), hours)
    for lat, lon, hours in zip(grouped["cell_ll_lat"], grouped["cell_ll_lon"], grouped["fishing_hours"])
]
fishing_raster = rasterize(
    fishing_shapes,
    out_shape=depth_array.shape,
    transform=bathymetry.transform,
    fill=0,
    dtype="float32"
)

# Normalize fishing effort using log scale
log_fishing = np.log1p(fishing_raster)
fishing_normalized = log_fishing / np.nanmax(log_fishing)

# Define depth-based cap on fault rate (values from literature)
def get_max_fault_rate(depth):
    if depth < 300:
        return 0.069
    elif depth < 1000:
        return 0.011
    else:
        return 0.004

# Vectorized map of max fault rate per cell
depth_based_max_fault_rate = np.vectorize(get_max_fault_rate)(depth_array)

# Final base fault rate = normalized fishing intensity × depth-based cap
base_fault_rate = fishing_normalized * depth_based_max_fault_rate

# === SECTION 4: Rasterize Seabed Substrate Costs ===
seabed_gdf = gpd.read_file("/Users/abelklaassens/Downloads/Master Thesis/A_Python/Master Thesis/baltic_substrate_folk5_rasterready.gpkg")
seabed_gdf["Folk_5cl_txt"] = seabed_gdf["Folk_5cl_txt"].str.replace(r"^\d+\.\s*", "", regex=True).str.strip()
folk_txt_to_code = {
    'Mud to muddy Sand': 1,
    'Sand': 2,
    'Coarse-grained sediment': 3,
    'Mixed sediment': 4,
    'Rock & boulders': 5,
    'No data at this level': 6
}
seabed_penalty_map = {1: 0, 2: 0, 3: 25000, 4: 15000, 5: 50000, 6: 15000}
seabed_gdf["folk_code"] = seabed_gdf["Folk_5cl_txt"].map(folk_txt_to_code)
seabed_gdf = seabed_gdf.to_crs(bathymetry.crs)
seabed_shapes = [
    (geom, seabed_penalty_map[cls])
    for geom, cls in zip(seabed_gdf.geometry, seabed_gdf["folk_code"])
    if not pd.isna(cls) and geom.is_valid
]
seabed_raster = rasterize(
    seabed_shapes,
    out_shape=depth_array.shape,
    transform=bathymetry.transform,
    fill=0,
    dtype='float32'
)

# === SECTION 5: Preprocessing – Land Mask Buffering ===
land_mask = np.isnan(depth_array)
distance_pixels = distance_transform_edt(~land_mask)
depth_array[distance_pixels < 5] = np.nan

# === SECTION 6: Define Cable Endpoints and Nearest Valid Cells ===
start_coord_orig = (18.640237, 59.080677)
end_coord_orig = (19.073931, 54.433272)
valid_mask = (~np.isnan(depth_array)) & (~np.isnan(fishing_raster)) & (~np.isnan(seabed_raster))
valid_indices = np.column_stack(np.where(valid_mask))
tree = cKDTree(valid_indices)
start_rc = bathymetry.index(*start_coord_orig)
end_rc = bathymetry.index(*end_coord_orig)
_, idx_start = tree.query(start_rc)
_, idx_end = tree.query(end_rc)
start_row, start_col = valid_indices[idx_start]
end_row, end_col = valid_indices[idx_end]
start_coord = bathymetry.xy(start_row, start_col)
end_coord = bathymetry.xy(end_row, end_col)

# === SECTION 7: Build Graph with TCO-based Edge Weights ===
G = nx.DiGraph()
geod = Geod(ellps="WGS84")
C_LW = 40000
C_DA = 60000
C_REPAIR = 3_000_000
LIFE_YEARS = 25
DEPTH_THRESHOLD = -500
PROTECTION_FACTORS = {
    "LW_noburial": 1.00,
    "DA_noburial": 0.70,
    "LW_buried": 0.12,
    "DA_buried": 0.10
}

for x in range(depth_array.shape[0]):
    for y in range(depth_array.shape[1]):
        if not valid_mask[x, y]:
            continue
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx_, ny_ = x + dx, y + dy
            if not (0 <= nx_ < depth_array.shape[0] and 0 <= ny_ < depth_array.shape[1]):
                continue
            if not valid_mask[nx_, ny_]:
                continue
            lon1, lat1 = bathymetry.xy(x, y)
            lon2, lat2 = bathymetry.xy(nx_, ny_)
            _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
            dist_km = dist_m / 1000
            avg_depth = np.nanmean([depth_array[x, y], depth_array[nx_, ny_]])
            burial_feasible = avg_depth >= DEPTH_THRESHOLD
            use_DA = avg_depth >= DEPTH_THRESHOLD
            cable_cost = C_DA if use_DA else C_LW
            burial_cost = np.nanmean([seabed_raster[x, y], seabed_raster[nx_, ny_]]) if burial_feasible else 0
            pf_key = ("DA" if use_DA else "LW") + ("_buried" if burial_feasible else "_noburial")
            protection_factor = PROTECTION_FACTORS[pf_key]
            fault_rate = np.nanmean([base_fault_rate[x, y], base_fault_rate[nx_, ny_]])
            expected_failures = fault_rate * protection_factor * dist_km * LIFE_YEARS
            repair_cost = expected_failures * C_REPAIR
            edge_tco = (cable_cost + burial_cost) * dist_km + repair_cost
            G.add_edge((x, y), (nx_, ny_), weight=edge_tco, protection=pf_key, length_km=dist_km)

# === SECTION 8: Compute Optimal Path ===
try:
    path = nx.shortest_path(G, source=(start_row, start_col), target=(end_row, end_col), weight='weight')
except nx.NetworkXNoPath:
    print("❌ No path could be found.")
    path = []

# === SECTION 9: Save Path ===
save_path = "/Users/abelklaassens/Downloads/Master Thesis/A_Python/Master Thesis/Results/optimal_path.pkl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "wb") as f:
    pickle.dump(path, f)
print(f"✅ Path saved to: {save_path}")

# === SECTION 10: Visualization – Fault Risk and Seabed ===
fig, ax = plt.subplots(figsize=(12, 10))
show(base_fault_rate, transform=bathymetry.transform, ax=ax, cmap="YlOrRd", alpha=0.5)
show(depth_display, transform=bathymetry.transform, ax=ax, cmap="Blues_r", alpha=0.5)
plt.imshow(seabed_raster, cmap="inferno")
plt.title("Rasterized Seabed Cost Layer")
plt.colorbar(label="€/km Penalty")
plt.show()

# === SECTION 11: Visualization – Fishing Activity + Path Overlay ===
fig, ax = plt.subplots(figsize=(12, 10))
show(depth_display, transform=bathymetry.transform, ax=ax, cmap="Blues_r", alpha=1)
show(base_fault_rate, transform=bathymetry.transform, ax=ax, cmap="hot", alpha=0.6)
for i in range(len(path) - 1):
    (r1, c1), (r2, c2) = path[i], path[i + 1]
    lon1, lat1 = bathymetry.xy(r1, c1)
    lon2, lat2 = bathymetry.xy(r2, c2)
    ax.plot([lon1, lon2], [lat1, lat2], color="cyan", linewidth=2, label="Cable Path" if i == 0 else "")
ax.scatter(*start_coord, color="green", marker="o", label="Stockholm")
ax.scatter(*end_coord, color="orange", marker="X", label="Gdansk")
ax.set_title("Fishing-Induced Fault Risk and Cable Path on Bathymetry", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
plt.tight_layout()
plt.show()

# === SECTION 12: Visualization – Seabed Cost and Cable Overlay ===
fig, ax = plt.subplots(figsize=(12, 10))
show(depth_display, transform=bathymetry.transform, ax=ax, cmap="Blues_r", alpha=0.3)
seabed_img = show(seabed_raster, transform=bathymetry.transform, ax=ax, cmap="inferno", alpha=0.6)
for i in range(len(path) - 1):
    (r1, c1), (r2, c2) = path[i], path[i + 1]
    lon1, lat1 = bathymetry.xy(r1, c1)
    lon2, lat2 = bathymetry.xy(r2, c2)
    ax.plot([lon1, lon2], [lat1, lat2], color="cyan", linewidth=2, label="Cable Path" if i == 0 else "")
ax.scatter(*start_coord, color="green", marker="o", label="Start")
ax.scatter(*end_coord, color="red", marker="X", label="End")
ax.set_title("Seabed Substrate Cost with Cable Path Overlay")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
fig.colorbar(seabed_img.get_images()[0], ax=ax, label="€/km Penalty")
plt.tight_layout()
plt.show()

# === SECTION 13: Visualization – Cable Type by Depth Segment ===
fig, ax = plt.subplots(figsize=(12, 10))
show(depth_display, transform=bathymetry.transform, ax=ax, cmap="Blues_r")
light_km = 0
double_km = 0
total_cost = 0
burial_total = 0
repair_total = 0
for i in range(len(path) - 1):
    (r1, c1), (r2, c2) = path[i], path[i + 1]
    lon1, lat1 = bathymetry.xy(r1, c1)
    lon2, lat2 = bathymetry.xy(r2, c2)
    _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
    dist_km = dist_m / 1000
    avg_depth = np.nanmean([depth_array[r1, c1], depth_array[r2, c2]])
    seabed_penalty = np.nanmean([seabed_raster[r1, c1], seabed_raster[r2, c2]])
    fault_rate = np.nanmean([base_fault_rate[r1, c1], base_fault_rate[r2, c2]])
    use_DA = avg_depth >= DEPTH_THRESHOLD
    cost_km = C_DA if use_DA else C_LW
    burial_feasible = avg_depth >= DEPTH_THRESHOLD
    protection_key = ("DA" if use_DA else "LW") + ("_buried" if burial_feasible else "_noburial")
    pf = PROTECTION_FACTORS[protection_key]
    if use_DA:
        double_km += dist_km
        ax.plot([lon1, lon2], [lat1, lat2], color="red", linewidth=2)
    else:
        light_km += dist_km
        ax.plot([lon1, lon2], [lat1, lat2], color="green", linewidth=2)
    total_cost += cost_km * dist_km
    burial_total += seabed_penalty * dist_km
    repair_total += fault_rate * pf * dist_km * LIFE_YEARS * C_REPAIR
ax.scatter(*start_coord, color="green", marker="o", label="Stockholm")
ax.scatter(*end_coord, color="orange", marker="X", label="Gdansk")
ax.set_title("Cable Route with Depth-Based Segments", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
plt.tight_layout()
plt.show()

# === SECTION 14: Print Final Stats ===
total_km = light_km + double_km
print("\n=== DEBUG: REPAIR COST CHECK ===")
print(f"Avg Fault Rate × Protection Factor × KM × Years: {repair_total / (C_REPAIR):.2f} expected failures")
print(f"Repair Cost Per Failure: €{C_REPAIR:,}")
print(f"Implied Average Fault Rate × PF: {(repair_total / (C_REPAIR * LIFE_YEARS * total_km)):.6f}")
print(f"Total Length: {total_km:.1f} km")
print(f"Lightweight Cable: {light_km:.1f} km")
print(f"Double Armored Cable: {double_km:.1f} km")
print(f"Construction Cost: €{total_cost:,.0f}")
print(f"Burial Penalty Cost: €{burial_total:,.0f}")
print(f"Repair Cost (lifetime): €{repair_total:,.0f}")
print(f"Total Estimated TCO: €{(total_cost + burial_total + repair_total):,.0f}")

# === SECTION 15: Square Land Basemap View with Path ===
path_file = "/Users/abelklaassens/Downloads/Master Thesis/A_Python/Master Thesis/Results/optimal_path.pkl"
bathymetry_file = "/Users/abelklaassens/Downloads/Master Thesis/A_Python/Master Thesis/D6_2024_resampled_1000m.tif"
with open(path_file, "rb") as f:
    path = pickle.load(f)
with rasterio.open(bathymetry_file) as bathy:
    path_coords = [bathy.xy(r, c)[::-1] for r, c in path]
line = LineString([(lon, lat) for lat, lon in path_coords])
gdf_path = gpd.GeoDataFrame(geometry=[line], crs="EPSG:4326").to_crs(epsg=3857)
fig, ax = plt.subplots(figsize=(12, 12))
gdf_path.plot(ax=ax, color="red", linewidth=2, label="Optimal Cable Route")
minx, miny, maxx, maxy = gdf_path.total_bounds
width, height = maxx - minx, maxy - miny
side = max(width, height)
cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
square_bounds = (cx - side/2, cy - side/2, cx + side/2, cy + side/2)
ax.set_xlim(square_bounds[0], square_bounds[2])
ax.set_ylim(square_bounds[1], square_bounds[3])
ax.set_aspect("equal")
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_title("Optimal Cable Route on Square Basemap", fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()
