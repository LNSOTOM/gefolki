#%%
import rasterio

with rasterio.open("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif") as msrc, \
     rasterio.open("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_DSM_3cm.tif") as lsrc:
    
    print("📏 MicaSense DSM shape:", msrc.height, "x", msrc.width)
    print("📏 LiDAR DEM shape    :", lsrc.height, "x", lsrc.width)
    print("🧭 MicaSense bounds   :", msrc.bounds)
    print("🧭 LiDAR bounds       :", lsrc.bounds)
    print("📐 MicaSense transform:", msrc.transform)
    print("📐 LiDAR transform    :", lsrc.transform)

#%%
## check if the two rasters have the same shape and CRS
with rasterio.open("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif") as msrc, \
     rasterio.open("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm.tif") as lsrc:
    
    print("📏 MicaSense DSM shape:", msrc.height, "x", msrc.width)
    print("📏 LiDAR DEM shape    :", lsrc.height, "x", lsrc.width)
    print("🧭 MicaSense bounds   :", msrc.bounds)
    print("🧭 LiDAR bounds       :", lsrc.bounds)
    print("📐 MicaSense transform:", msrc.transform)
    print("📐 LiDAR transform    :", lsrc.transform)


#%%
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

def resample_lidar_to_dsm_shape_preserve_crs(src_path, ref_path, dst_path):
    with rasterio.open(src_path) as src, rasterio.open(ref_path) as ref:
        dst_profile = src.profile.copy()  # Preserve LiDAR CRS
        dst_profile.update(
            height=ref.height,
            width=ref.width,
            transform=ref.transform,
            dtype='float32'
        )

        # Empty array with shape of DSM
        resampled = np.empty((ref.height, ref.width), dtype='float32')

        # Reproject
        reproject(
            source=rasterio.band(src, 1),
            destination=resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref.transform,
            dst_crs=src.crs,  # Keep original LiDAR CRS
            resampling=Resampling.bilinear
        )

        with rasterio.open(dst_path, 'w', **dst_profile) as dst:
            dst.write(resampled, 1)

        print(f"✅ Resampled LiDAR DEM (with original CRS) saved to: {dst_path}")




#%%
resample_lidar_to_dsm_shape_preserve_crs(
    src_path="/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_DSM_3cm.tif",
    ref_path="/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif",
    dst_path="/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm.tif"
)


#########################
### NOP
#%%
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import gefolki

# === Input files ===
rgb_dsm_path = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_micasense_dsm_5cm.tif"
lidar_dem_path = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dem_5cm_normalized.tif"  # normalized to [0,1]

# === Load raster data ===
def load_and_mask(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
        nodata = src.nodata if src.nodata is not None else -9999
        mask = arr != nodata
        arr[~mask] = 0
        return arr, mask, profile

master_dsm, mask_master, profile = load_and_mask(rgb_dsm_path)
slave_dem, mask_slave, _ = load_and_mask(lidar_dem_path)

# === Ensure same shape (optional resampling step if mismatched) ===
if master_dsm.shape != slave_dem.shape:
    raise ValueError("Master and slave DEMs must be the same shape. Resample one if needed.")

# === Run GeFolki registration ===
print("🔄 Running GeFolki registration...")
flow_u, flow_v = gefolki.gefolki(
    master=master_dsm,
    slave=slave_dem,
    radius=[32, 24, 16, 8],
    levels=4,
    iter=2,
    rank=4,
    contrast_adapt=True
)

# === Warp LiDAR DEM to align with MicaSense DSM ===
aligned_lidar = gefolki.tools.wrapData(slave_dem, flow_u, flow_v)

# === Save aligned raster ===
aligned_output = lidar_dem_path.replace(".tif", "_aligned_to_micasense.tif")
profile.update(dtype='float32')

with rasterio.open(aligned_output, "w", **profile) as dst:
    dst.write(aligned_lidar.astype('float32'), 1)

print(f"✅ Aligned DEM saved to: {aligned_output}")
