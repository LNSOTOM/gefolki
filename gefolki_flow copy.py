import sys
import numpy as np
import pylab as pl
import time
import rasterio

# Add the path to GeFolki Python source
sys.path.append('/media/laura/laura_usb/code/fvc_structure/gefolki/python')

# Import functional EFolki
from python.algorithm import EFolki
from python.tools import wrapData

# === Input files ===
micasense_dsm_path = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_micasense_dsm_5cm.tif"
lidar_dem_path     = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dem_5cm_normalized_resampled.tif"

# === Load raster data ===
def load_and_mask(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
        nodata = src.nodata if src.nodata is not None else -9999
        mask = arr != nodata
        arr[~mask] = 0
        return arr, mask, profile

master_dsm, mask_master, profile = load_and_mask(micasense_dsm_path)
slave_dem, mask_slave, _         = load_and_mask(lidar_dem_path)

# === Ensure same shape ===
if master_dsm.shape != slave_dem.shape:
    raise ValueError("Master and slave DEMs must be the same shape. Resample one if needed.")

# === Run GeFolki registration ===
print("🔄 Calculating optical flow with EFolki...\n")
start = time.perf_counter()

u, v = EFolki(
    master_dsm,
    slave_dem,
    iteration=8,
    radius=[16, 8, 4],
    rank=4,
    levels=6
)

# === Save flow components ===
np.save("flow_u.npy", u)
np.save("flow_v.npy", v)
print("💾 Saved optical flow fields to: flow_u.npy, flow_v.npy")


elapsed_min = (time.perf_counter() - start) / 60
print(f"✅ Flow calculated in {elapsed_min:.2f} minutes\n")

# === Optional: plot flow norm ===
N = np.sqrt(u**2 + v**2)
pl.figure()
pl.imshow(N, cmap='viridis')
pl.title('Norm of Optical Flow (EFolki)')
pl.colorbar()
pl.tight_layout()
pl.savefig("flow_norm.png", dpi=300)

# === Warp LiDAR DEM to align ===
aligned_lidar = wrapData(slave_dem, u, v)

# === Save aligned DEM ===
aligned_output = lidar_dem_path.replace(".tif", "_aligned_to_micasense.tif")
profile.update(dtype='float32')

with rasterio.open(aligned_output, "w", **profile) as dst:
    dst.write(aligned_lidar.astype('float32'), 1)

print(f"✅ Aligned DEM saved to: {aligned_output}")


