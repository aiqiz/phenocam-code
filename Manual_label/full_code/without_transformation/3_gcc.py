import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime, timedelta
import pandas as pd
import time
import json
from labelme.utils import shape_to_mask
from skimage.draw import polygon
from skimage.measure import regionprops, label
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from matplotlib.colors import LinearSegmentedColormap
from pvlib.location import Location


def calculate_sza(dt, latitude_deg, longitude_deg, altitude_m=0):
    site = Location(latitude=latitude_deg, longitude=longitude_deg, altitude=altitude_m)
    solar_position = site.get_solarposition(times=dt)
    sza_deg = solar_position['zenith'].values[0]
    return sza_deg

def ensure_grayscale(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def calculate_sza(dt, latitude_deg, longitude_deg, altitude_m=0):
    site = Location(latitude=latitude_deg, longitude=longitude_deg, altitude=altitude_m)
    solar_position = site.get_solarposition(times=dt)
    sza_deg = solar_position['zenith'].values[0]
    return sza_deg

def readin_time(fname, UTC=True):
    dt = False

    # Match formats:
    # 1. ...YYYYMMDD_HHMMSS.jpg or .png
    # 2. ...YYYYMMDD_HHMM.jpg
    # 3. HIP_oblique_01_2023_05_29_12_00_00
    match = re.search(
        r'(?:(\d{8})[_-](\d{6}|\d{4}))|(?:_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2}))',
        fname
    )

    if match:
        try:
            if match.group(1) and match.group(2):
                date_part = match.group(1)
                time_part = match.group(2)
                if len(time_part) == 4:
                    dt_str = date_part + time_part + "00"
                else:
                    dt_str = date_part + time_part
                dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
            elif match.group(3):
                # Format: _YYYY_MM_DD_HH_MM_SS
                dt_str = f"{match.group(3)}{match.group(4)}{match.group(5)}{match.group(6)}{match.group(7)}{match.group(8)}"
                dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")

            if UTC:
                dt -= timedelta(hours=4)  # Convert UTC to Toronto local time
        except ValueError as e:
            print(f"Failed to parse time: {dt_str}, error: {e}")
    else:
        print(f"No match for: {fname}")

    return dt

def check_time_SZA(fname, hour_threshold, sza_threshold, latitude_deg, longitude_deg):
    dt = readin_time(fname)
    if dt == False:
        return False
    sza = calculate_sza(dt, latitude_deg, longitude_deg)
    if dt.hour <= hour_threshold and sza <= sza_threshold:
        return True
    return False

def edge_density(image):
    gray = ensure_grayscale(image)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    #print("edge_density", edge_density)
    return edge_density

def GCC_calc_global(image, data, width, height):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    denominator = R + G + B
    denominator[denominator == 0] = np.nan
    GCC = G / denominator

    mask_union = np.zeros((height, width), dtype=bool)
    for shape in data['shapes']:
        mask = shape_to_mask((height, width), shape['points'], shape_type=shape['shape_type'])
        mask_union |= mask  # logical OR for union

    gcc_all_roi_pixels = GCC[mask_union]

    global_mean = np.nanmean(gcc_all_roi_pixels)
    global_std = np.nanstd(gcc_all_roi_pixels)
    global_min = np.nanmin(gcc_all_roi_pixels)
    global_max = np.nanmax(gcc_all_roi_pixels)
    print(f"Mean GCC : {global_mean:.4f}")
    return global_mean, global_std, global_min, global_max

def plot_dual_shadded(region_labels, mean_vals, std_vals, min_vals, max_vals):
    mean_vals = np.array(mean_vals)
    std_vals = np.array(std_vals)
    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)

    # Plot mean line
    plt.figure(figsize=(8, 6))
    plt.plot(region_labels, mean_vals, label='Mean GCC', color='green', linewidth=2)
    plt.fill_between(region_labels, mean_vals - std_vals, mean_vals + std_vals,
                    color='green', alpha=0.3, label='±1 Std Dev')

    # Fill min-max range
    #plt.fill_between(region_labels, min_vals, max_vals, color='green', alpha=0.15, label='MinMax Range')
    plt.title('GCC statistics time series')
    plt.xlabel('Date')
    plt.ylabel('GCC')
    plt.tight_layout()
    plt.legend()
    plt.show()

def get_union_mask_from_json(data, width, height):
    union_mask = np.zeros((height, width), dtype=bool)

    for shape in data['shapes']:
        mask = shape_to_mask((height, width), shape['points'], shape_type=shape['shape_type'])
        union_mask |= mask

    return union_mask

def soft_interior_weight(d, e_mean=0, e_std=0, min_weight=0.05):
    w = np.ones_like(d)
    transition_zone = (d >= 0) & (d < e_mean)
    falloff = np.exp(-0.5 * ((e_mean - d[transition_zone]) / e_std) ** 2)
    w[transition_zone] = np.clip(falloff, min_weight, 1)
    w[d < 0] = 0
    return w

def generate_soft_interior_mask(union_mask, e_mean=0, e_std=0, show_plot=False):
    """
    Create soft mask from inside a polygon ROI to its boundary.
    Center = weight 1; edge = 0; smooth increase from 0→1 over e_mean, controlled by e_std.
    """
    height, width = union_mask.shape
    weight_mask = np.zeros((height, width), dtype=float)

    labeled = label(union_mask)
    props = regionprops(labeled)

    for region in props:
        minr, minc, maxr, maxc = region.bbox
        region_mask = region.image.astype(bool)

        boundaries = find_boundaries(region_mask, mode='inner')
        dist_to_edge = distance_transform_edt(~boundaries)

        local_weight = soft_interior_weight(dist_to_edge, e_mean, e_std)

        local_weight *= region_mask

        weight_mask[minr:maxr, minc:maxc] = np.maximum(
            weight_mask[minr:maxr, minc:maxc],
            local_weight
        )

    if show_plot:
        black_to_white = LinearSegmentedColormap.from_list("black_to_white", ["brown", "green"])
        plt.figure(figsize=(6, 5))
        plt.imshow(weight_mask, cmap=black_to_white, vmin=0, vmax=1)
        plt.title(f"Soft Interior Mask (e_mean={e_mean:.2f}, e_std={e_std:.2f})")
        plt.colorbar(label="Weight")
        plt.tight_layout()
        plt.show()

    return weight_mask

def compute_weighted_gcc(image, weight_mask, show_heatmap=False):
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    black_mask = (R == 0) & (G == 0) & (B == 0)

    denominator = R + G + B
    denominator[denominator == 0] = np.nan
    GCC = G / denominator
    GCC[black_mask] = np.nan  # Optional: mark black pixels explicitly

    valid_mask = (weight_mask > 0) & ~black_mask & ~np.isnan(GCC)
    gcc_values = GCC[valid_mask]
    weights = weight_mask[valid_mask]

    if len(gcc_values) == 0:
        print("No valid pixels in ROI")
        return np.nan, np.nan, np.nan, np.nan
    
    if show_heatmap:
        green_to_white = LinearSegmentedColormap.from_list("green_to_white", ["brown", "green"])
        heatmap = np.full_like(GCC, np.nan)
        heatmap[valid_mask] = GCC[valid_mask]
        plt.figure(figsize=(6, 5))
        plt.imshow(heatmap, cmap=green_to_white, vmin=0.25, vmax=0.4)
        plt.colorbar(label="GCC value")
        plt.title("Local GCC Heatmap (Green → Brown)")
        plt.tight_layout()
        plt.show()

    weighted_mean = np.average(gcc_values, weights=weights)
    weighted_std = np.sqrt(np.average((gcc_values - weighted_mean) ** 2, weights=weights))
    gcc_min = np.min(gcc_values)
    gcc_max = np.max(gcc_values)

    return weighted_mean, weighted_std, gcc_min, gcc_max

def gcc(folder_dir, image_folder_name, ref_image_name, roi_json,
        time_threshold, sza_threshold, edge_density_threshold, 
        local_lattitude, local_longitude, inlier_ratio_threshold):
    VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    
    ### Load Affine Transformation Reference Image
    ref_img = cv2.imread(folder_dir + ref_image_name)
    ref_img_width = ref_img.shape[1]
    ref_img_height = ref_img.shape[0]

    ### Load mask
    with open(folder_dir + roi_json, 'r') as f:
        data = json.load(f)
    
    i = 0
    date_list = []
    global_mean = []
    global_std = []
    global_min = []
    global_max = []

    union_mask = get_union_mask_from_json(data, ref_img_width, ref_img_height)    

    cutoff1 = datetime(2023, 5, 5)
    cutoff2 = datetime(2023, 5, 26)

    for filename in sorted(os.listdir(folder_dir + image_folder_name)):
        i += 1
        filepath = os.path.join(folder_dir + image_folder_name, filename)

        ### 3). SZA & time check
        if not filename.lower().endswith(VALID_EXTENSIONS):
            continue  # Skip .DS_Store and other non-image files
        if not check_time_SZA(filename, time_threshold, sza_threshold, local_lattitude, local_longitude):
            continue # Skip if not in correct SZA threshold and daytime

        file_time = readin_time(filename)
        if file_time < cutoff2:
           continue

        img = cv2.imread(filepath)
        if img is None:
            continue
        if edge_density(img) <= edge_density_threshold:
            continue

        img = img.astype(np.float32)
        
        weighted_mask = generate_soft_interior_mask(union_mask)
        mean, std, min_val, max_val = compute_weighted_gcc(img, weighted_mask)

        global_mean.append(mean)
        global_std.append(std)
        global_min.append(min_val)
        global_max.append(max_val)
        date_list.append(file_time)
        print(file_time, mean)

    df = pd.DataFrame({
        "datetime": date_list,
        "mean": global_mean,
        "std": global_std,
        "min": global_min,
        "max": global_max
    })

    return df


if __name__ == "__main__":
    folder_dir = "/Volumes/Aiqi_02/phenocams/tame_phenocams/leftandright/right_camera/Bushnell_Aug_2_19_2023/"
    image_folder_name = ''
    ref_image_name = "08020001.JPG"
    roi_json = "08020001.json"

    local_lattitude = 43.7847  # Toronto
    local_longitude = -79.1859  # Toronto

    time_threshold = 14
    sza_threshold = 80

    edge_density_threshold = 0.005
    inlier_ratio_threshold = 0.7

    start = time.time()

    df = gcc(folder_dir, image_folder_name, ref_image_name, roi_json,
        time_threshold, sza_threshold, edge_density_threshold, 
        local_lattitude, local_longitude, inlier_ratio_threshold)
    
    #df.to_csv(folder_dir + "gcc_global_jun25.csv", index=False)

    end = time.time()
    print(f"Elapsed time: {end - start:.6f} seconds")


#local_lattitude = 43.663889  # Toronto
#local_longitude = -79.395656  # Toronto