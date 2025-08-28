import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime, timedelta
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import time
from skimage.restoration import wiener
from pvlib.location import Location


def calculate_sza(dt, latitude_deg, longitude_deg, altitude_m=0):
    site = Location(latitude=latitude_deg, longitude=longitude_deg, altitude=altitude_m)
    solar_position = site.get_solarposition(times=dt)
    sza_deg = solar_position['zenith'].values[0]
    return sza_deg

def readin_time(folder_path, UTC=False):
    files = []
    time_list = []

    for fname in os.listdir(folder_path):
        # Match formats:
        # 1. ...YYYYMMDD_HHMMSS.jpg or .png
        # 2. ...YYYYMMDD_HHMM.jpg
        # 3. HIP_oblique_01_2023_05_29_12_00_00
        match = re.search(
            r'(?:(\d{8})[_-](\d{6}|\d{4}))|(?:_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2}))',
            fname
        )
        if match:
            if match.group(1) and match.group(2):
                # Format: YYYYMMDD_HHMMSS or YYYYMMDD_HHMM
                date_part = match.group(1)
                time_part = match.group(2)
                if len(time_part) == 4:
                    time_part += "00"
                dt_str = date_part + time_part
            elif match.group(3):
                # Format: _YYYY_MM_DD_HH_MM_SS
                dt_str = f"{match.group(3)}{match.group(4)}{match.group(5)}{match.group(6)}{match.group(7)}{match.group(8)}"

            try:
                dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
                if UTC:
                    dt -= timedelta(hours=4)  # Convert UTC to Toronto time (UTC-4)
                files.append((dt, fname))
                time_list.append(dt)
            except ValueError:
                print(f"Skipping invalid datetime: {fname} -> {dt_str}")

    files.sort()
    time_list.sort()
    return files, time_list

def edge_density(image, print_edge_density=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    if print_edge_density:
        print("edge density", edge_density)
    return edge_density

def apply_clahe(img, imshow = False):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    
    if imshow:
        cv2.imshow("Inlier Matches", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_eq

def wiener_deblur(img_gray, psf_size=5, imshow = False):
    # Estimate a basic blur kernel (Gaussian)
    psf = np.ones((psf_size, psf_size)) / (psf_size ** 2)
    img_float = img_gray.astype(np.float32) / 255.0
    
    # Apply Wiener deconvolution
    deblurred = wiener(img_float, psf, balance=0.1)
    deblurred = np.clip(deblurred * 255, 0, 255).astype(np.uint8)
    if imshow:
        cv2.imshow("Inlier Matches", deblurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return deblurred

def ensure_grayscale(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def akaze_affine_transformation(ref_img, cur_img, k, ransac_prog_threshold, ratio_thresh, imshow = False):
    print("yes")
    h, w = ref_img.shape[:2]
    ref_img = apply_clahe(ref_img)
    cur_img = apply_clahe(cur_img)
    ref_img = wiener_deblur(ensure_grayscale(ref_img))
    cur_img = wiener_deblur(ensure_grayscale(cur_img))

    # Masks to ignore upper 1/k and lower 1/k
    mask_ref = np.zeros((h, w), dtype=np.uint8)
    mask_cur = np.zeros((h, w), dtype=np.uint8)
    mask_ref[h//k: (h//k)*(k-1), :] = 255
    mask_cur[h//k: (h//k)*(k-1), :] = 255

    akaze = cv2.AKAZE_create(
        threshold=0.0002,
        nOctaves=6,
        nOctaveLayers=8,
        diffusivity=cv2.KAZE_DIFF_CHARBONNIER
    )

    # After CLAHE + Sharpening
    kp1, des1 = akaze.detectAndCompute(ref_img, mask_ref)
    kp2, des2 = akaze.detectAndCompute(cur_img, mask_cur)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    try:
        matches_knn = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches_knn if m.distance < ratio_thresh * n.distance]
    except cv2.error as e:
        print("OpenCV error during knnMatch:", e)
        return None, None, None, None

    if len(good_matches) < 4:
        print("Too few good matches:", len(good_matches))
        return None, None, None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate 6-DoF affine transformation
    A, inlier_mask = cv2.findHomography(
        dst_pts, src_pts, method=cv2.USAC_MAGSAC, ransacReprojThreshold=ransac_prog_threshold
    )

    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inlier_mask[i]]
    if imshow:
        img_inliers = cv2.drawMatches(ref_img, kp1, cur_img, kp2, inlier_matches, None, flags=2)
        cv2.imshow("Inlier Matches", img_inliers)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if A is None or inlier_mask is None:
        print(f"Warning: Affine estimation failed for k={k}")
        return None, None, None, None
    
    inlier_sharpness = [
    kp1[m.queryIdx].response
        for i, m in enumerate(good_matches)
        if inlier_mask[i]
    ]
    
    return A, dst_pts, src_pts, inlier_mask

def goodness_of_match_homography(H, dst_pts, src_pts, inlier_mask, k):
    inliers_dst = dst_pts[inlier_mask.ravel() == 1]
    inliers_src = src_pts[inlier_mask.ravel() == 1]

    # Filter to bottom 1/k of the image height (optional)
    h = np.max(inliers_src[:, :, 1])
    keep_indices = inliers_src[:, 0, 1] > (h / k)
    inliers_dst = inliers_dst[keep_indices]
    inliers_src = inliers_src[keep_indices]

    # Apply homography transformation
    dst_pts_proj = cv2.perspectiveTransform(inliers_dst, H)

    # Compute Euclidean error
    errors = np.linalg.norm(dst_pts_proj.squeeze() - inliers_src.squeeze(), axis=1)

    # Percentage of points with error < 2 pixels
    below = np.sum(errors < 2) / len(errors) * 100
    inlier_ratio = np.sum(inlier_mask) / len(inlier_mask)

    return np.mean(errors), np.median(errors), np.max(errors), np.std(errors), inlier_ratio

def image_alignment(folder_dir, image_folder_name, ref_image_name, region_to_ignore, 
                    local_lattitude, local_longitude, sample_interval, time_threshold,
                    sza_threshold, edge_density_threshold, ransac_prog_threshold, ratio_thresh,
                    imshow = False):
    VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_folder_path = folder_dir + image_folder_name
    ref_image_path = folder_dir + ref_image_name

    ref_img = cv2.imread(ref_image_path, cv2.IMREAD_COLOR)

    files, dt = readin_time(image_folder_path)
    sza = [calculate_sza(t, local_lattitude, local_longitude) for t in dt]
    columns = ['Filename', 'Datetime', 'SZA']
    readin_df = pd.DataFrame(columns=columns)

    for i, (t, name) in enumerate(files):
        if t.hour <= time_threshold and sza[i] <= sza_threshold:
            readin_df.loc[len(readin_df)] = [name, t, sza[i]]

    readin_df["Filename"] = image_folder_path + readin_df["Filename"]
    readin_df_sampled = readin_df.iloc[::sample_interval].reset_index(drop=True)

    columns = ['datetime', 'e_mean', 'e_median', 'e_max', 'e_std', 'inlier_ratio'] + \
              [f'A{i}{j}' for i in range(3) for j in range(3)]  # 2x3 affine matrix
    error_df = pd.DataFrame(columns=columns)

    cutoff1 = datetime(2023, 5, 5)
    cutoff2 = datetime(2023, 5, 26)
    
    for index, row in readin_df_sampled.iterrows():
        filename = row['Filename']
        file_time = row['Datetime']
        if not filename.lower().endswith(VALID_EXTENSIONS):
            print(f"Failed to read image: {filename}")
            continue

        # set cutoff date -> starting from cutoff date (in case of camera moves totally to the otherside)
        # if row["Datetime"] < cutoff1:
        #    continue
        if row["Datetime"] < cutoff2:
           continue

        cur_img = cv2.imread(filename)
        if edge_density(cur_img) <= edge_density_threshold:
            continue

        result = akaze_affine_transformation(ref_img, cur_img, region_to_ignore, ransac_prog_threshold, ratio_thresh)
        if result is None or any(r is None for r in result):
            continue
        A, dst_pts, src_pts, inlier_mask = result
        e_mean, e_median, e_max, e_std, inlier_ratio = goodness_of_match_homography(A, dst_pts, src_pts, inlier_mask, region_to_ignore)

        if imshow:
            aligned_img = cv2.warpPerspective(cur_img, A, (ref_img.shape[1], ref_img.shape[0]))
            blended = cv2.addWeighted(ref_img, 0.5, aligned_img, 0.5, 0)
            cv2.imshow("Overlap of Reference and Aligned Image", blended)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        A = A.flatten()
        row = [file_time, e_mean, e_median, e_max, e_std, inlier_ratio] + A.tolist()
        error_df.loc[len(error_df)] = row

        print(e_mean, e_median, e_max, e_std, inlier_ratio)

    return error_df
    

if __name__ == "__main__":
    folder_dir = ".."
    image_folder_name = 'data/'
    ref_image_name = 'reference_image_02.jpg'

    region_to_ignore = 8 # ignore upper 1/x of the figure (sky & metadata of camera)

    local_lattitude = 43.663889  # Toronto
    local_longitude = -79.395656  # Toronto

    time_threshold = 16
    sza_threshold = 80

    edge_density_threshold = 0.003
    ransac_prog_threshold = 5 #pixel
    ratio_thresh = 0.6

    sample_interval = 200
    
    start = time.time()
    error_df = image_alignment(folder_dir, image_folder_name, ref_image_name, region_to_ignore, 
                    local_lattitude, local_longitude, sample_interval, time_threshold, sza_threshold,
                    edge_density_threshold, ransac_prog_threshold, ratio_thresh)

    error_df.to_csv(folder_dir + 'transformation_reference_projective_02.csv', index=False)
    end = time.time()
    print(f"Elapsed time: {end - start:.6f} seconds")


'''
pheocam01
edge_density_threshold = 0.01
ransac_prog_threshold = 10 #pixel
ratio_thresh = 0.7

pheocam02
edge_density_threshold = 0.005
ransac_prog_threshold = 5 #pixel
ratio_thresh = 0.6

cutoff1 = datetime(2023, 4, 5)
cutoff2 = datetime(2024, 4, 5)

phenocam03
edge_density_threshold = 0.005
ransac_prog_threshold = 15 #pixel
ratio_thresh = 0.7

phenocam04
edge_density_threshold = 0.004
ransac_prog_threshold = 8 #pixel
ratio_thresh = 0.75

phenocam05
edge_density_threshold = 0.004
ransac_prog_threshold = 8 #pixel
ratio_thresh = 0.75

highpark
edge_density_threshold = 0.004
ransac_prog_threshold = 8 #pixel
ratio_thresh = 0.7
'''