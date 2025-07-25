import numpy as np
import laspy
import rasterio
from scipy.ndimage import map_coordinates
from pathlib import Path

# === Input paths ===
las_input_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
aligned_dem_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dem_5cm_normalized_resampled_aligned_to_micasense.tif")
flow_u = np.load("flow_u.npy")
flow_v = np.load("flow_v.npy")

# === Load geotransform from aligned DEM ===
with rasterio.open(aligned_dem_path) as src:
    transform = src.transform
    width, height = src.width, src.height
    pixel_size_x = transform[0]
    pixel_size_y = -transform[4]  # Y pixel size is usually negative
    origin_x, origin_y = transform[2], transform[5]

# === Load LiDAR point cloud ===
las = laspy.read(str(las_input_path))
x = np.array(las.x)
y = np.array(las.y)
z = np.array(las.z)

# === Convert world (x, y) to raster (row, col) coordinates ===
col = ((x - origin_x) / pixel_size_x).astype(np.float32)
row = ((origin_y - y) / pixel_size_y).astype(np.float32)  # flip Y

# === Filter points inside raster bounds ===
valid = (
    (col >= 0) & (col < width - 1) &
    (row >= 0) & (row < height - 1)
)

# === Interpolate optical flow at valid positions ===
coords = np.vstack((row[valid], col[valid]))
dx_pix = map_coordinates(flow_u, coords, order=1, mode='nearest')
dy_pix = map_coordinates(flow_v, coords, order=1, mode='nearest')

# === Convert pixel flow to meters ===
dx_m = dx_pix * pixel_size_x
dy_m = -dy_pix * pixel_size_y  # Flip Y axis

# === Apply displacement ===
x_new = x.copy()
y_new = y.copy()
x_new[valid] += dx_m
y_new[valid] += dy_m

# === Save displaced LiDAR to new LAS file ===
las.x = x_new
las.y = y_new
output_path = las_input_path.with_name(las_input_path.stem + "_displaced_by_efolki.las")
las.write(str(output_path))

print(f"✅ Displaced LiDAR saved to: {output_path}")
