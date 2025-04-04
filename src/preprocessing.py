# src/preprocessing.py

import cv2
import numpy as np
import math
import os
import time # Added for timing prints
import matplotlib.pyplot as plt # Keep for standalone testing/debugging

# Optional dependency for thinning (preferred method)
try:
    import cv2.ximgproc
    XIMGPROC_AVAILABLE = True
    print("DEBUG: cv2.ximgproc found. Thinning will use THINNING_ZHANGSUEN.")
except ImportError:
    XIMGPROC_AVAILABLE = False
    print("Warning: cv2.ximgproc not found. Install 'opencv-contrib-python' for optimal thinning.")
    print("Falling back to a basic morphological thinning (less effective).")

# --- Configuration ---
ORIENTATION_BLOCK_SIZE = 16
FREQUENCY_BLOCK_SIZE = 32
GABOR_KERNEL_SIZE = 16
GABOR_SIGMA = 4.0
GABOR_GAMMA = 0.5
GABOR_PSI = 0

# --- Helper Functions (plot_image) ---
# ... (keep plot_image as before) ...
def plot_image(img, title="Image"):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    except ImportError:
        print(f"Info: Matplotlib not found. Cannot plot image '{title}'.")

# --- Core Preprocessing Steps ---
# ... (keep load_image, normalize_image as before) ...
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image file: {image_path}")
    return img

def normalize_image(img, target_mean=100.0, target_variance=100.0):
    mean_val = np.mean(img)
    std_val = np.std(img)
    if std_val < 1e-6:
        print("Warning: Image standard deviation is near zero during normalization.")
        return np.full_like(img, int(target_mean), dtype=np.uint8)
    norm_img = target_mean + ( (img - mean_val) / std_val ) * np.sqrt(target_variance)
    norm_img = np.clip(norm_img, 0, 255)
    return norm_img.astype(np.uint8)

# ... (keep segment_fingerprint as corrected before) ...
def segment_fingerprint(img, block_size=16, threshold_ratio=0.15):
    h, w = img.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    border_value = int(np.mean(img))
    padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=border_value)
    variances = np.zeros(((h + pad_h) // block_size, (w + pad_w) // block_size))
    for r in range(0, h + pad_h, block_size):
        for c in range(0, w + pad_w, block_size):
            block = padded_img[r:r+block_size, c:c+block_size]
            if block.size > 0: variances[r//block_size, c//block_size] = np.var(block)
            else: variances[r//block_size, c//block_size] = 0
    min_var, max_var = np.min(variances), np.max(variances)
    if max_var - min_var < 1e-6: variance_threshold = min_var - 1e-6
    else: variance_threshold = min_var + (max_var - min_var) * threshold_ratio
    mask_blocks = variances > variance_threshold
    mask_full_padded = cv2.resize(mask_blocks.astype(np.uint8), (w + pad_w, h + pad_h), interpolation=cv2.INTER_NEAREST)
    mask_full = mask_full_padded[0:h, 0:w]
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    try:
        cleaned_mask = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, kernel_open, iterations=1)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    except cv2.error as morph_err:
         print(f"Warning: OpenCV error during mask morphology ({morph_err}). Using uncleaned mask.")
         cleaned_mask = mask_full
    segmented_img = np.full_like(img, 255, dtype=np.uint8)
    segmented_img[cleaned_mask == 1] = img[cleaned_mask == 1]
    return segmented_img, cleaned_mask

# ... (keep estimate_orientation_field as before - returns smoothed block map) ...
def estimate_orientation_field(img, block_size=ORIENTATION_BLOCK_SIZE, smooth_kernel_size=5):
    h, w = img.shape
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    Gxx = sobel_x**2; Gyy = sobel_y**2; Gxy = sobel_x * sobel_y
    kernel = np.ones((block_size, block_size), np.float64) / (block_size * block_size)
    sum_Gxx = cv2.filter2D(Gxx, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    sum_Gyy = cv2.filter2D(Gyy, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    sum_Gxy = cv2.filter2D(Gxy, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    orientation_rad = 0.5 * np.arctan2(2 * sum_Gxy + 1e-10, sum_Gxx - sum_Gyy + 1e-10)
    cos_2theta = np.cos(2 * orientation_rad); sin_2theta = np.sin(2 * orientation_rad)
    smooth_sigma = smooth_kernel_size / 3.0
    smoothed_cos_2theta = cv2.GaussianBlur(cos_2theta, (smooth_kernel_size, smooth_kernel_size), smooth_sigma)
    smoothed_sin_2theta = cv2.GaussianBlur(sin_2theta, (smooth_kernel_size, smooth_kernel_size), smooth_sigma)
    smoothed_orientation_rad_pixel = 0.5 * np.arctan2(smoothed_sin_2theta, smoothed_cos_2theta)
    rows_blk, cols_blk = h // block_size, w // block_size
    smoothed_orientation_map_block = np.zeros((rows_blk, cols_blk))
    for r_blk in range(rows_blk):
        for c_blk in range(cols_blk):
             r_center, c_center = int((r_blk + 0.5) * block_size), int((c_blk + 0.5) * block_size)
             smoothed_orientation_map_block[r_blk, c_blk] = smoothed_orientation_rad_pixel[min(r_center, h-1), min(c_center, w-1)]
    return smoothed_orientation_map_block # Return block map

# ... (keep estimate_ridge_frequency as corrected before) ...
def estimate_ridge_frequency(img, orientation_map, mask, block_size=FREQUENCY_BLOCK_SIZE):
    h, w = img.shape
    rows_blk, cols_blk = orientation_map.shape
    ori_block_size = img.shape[0] // rows_blk
    frequency_map = np.zeros_like(orientation_map, dtype=float)
    default_freq = 1.0 / 8.0
    is_foreground_block = np.zeros_like(orientation_map, dtype=bool)
    freq_win_size = block_size
    win_half = freq_win_size // 2
    for r_blk in range(rows_blk):
        for c_blk in range(cols_blk):
            r_center = int((r_blk + 0.5) * ori_block_size)
            c_center = int((c_blk + 0.5) * ori_block_size)
            mask_r_center = min(max(0, r_center), h - 1)
            mask_c_center = min(max(0, c_center), w - 1)
            if mask[mask_r_center, mask_c_center] != 0: is_foreground_block[r_blk, c_blk] = True
            else: frequency_map[r_blk, c_blk] = default_freq; continue
            angle_rad = orientation_map[r_blk, c_blk]
            r_start = max(0, r_center - win_half); r_end = min(h, r_center + win_half + 1)
            c_start = max(0, c_center - win_half); c_end = min(w, c_center + win_half + 1)
            block = img[r_start:r_end, c_start:c_end]
            if block.size < 16: frequency_map[r_blk, c_blk] = default_freq; continue
            block_h, block_w = block.shape
            rot_center = (block_w / 2.0, block_h / 2.0)
            rot_angle_deg = math.degrees(angle_rad + np.pi / 2.0)
            try:
                rotation_matrix = cv2.getRotationMatrix2D(rot_center, rot_angle_deg, 1.0)
                rotated_block = cv2.warpAffine(block, rotation_matrix, (block_w, block_h), flags=cv2.INTER_NEAREST)
            except cv2.error: frequency_map[r_blk, c_blk] = default_freq; continue
            x_signature = np.sum(rotated_block, axis=0)
            if len(x_signature) < 2: frequency_map[r_blk, c_blk] = default_freq; continue
            fft_result = np.fft.fft(x_signature - np.mean(x_signature))
            freqs = np.fft.fftfreq(len(x_signature))
            dominant_freq = 0
            if len(fft_result) > 1:
                 peak_index = np.argmax(np.abs(fft_result[1:])) + 1
                 dominant_freq = abs(freqs[peak_index])
            if dominant_freq > 1e-4:
                 wavelength = 1.0 / dominant_freq
                 if 3 < wavelength < 25: frequency_map[r_blk, c_blk] = dominant_freq
                 else: frequency_map[r_blk, c_blk] = default_freq
            else: frequency_map[r_blk, c_blk] = default_freq
    smooth_sigma_freq = 2.0
    kernel_size_freq = int(smooth_sigma_freq * 3) * 2 + 1
    foreground_block_float_mask = is_foreground_block.astype(float)
    smoothed_freq = cv2.GaussianBlur(frequency_map * foreground_block_float_mask, (kernel_size_freq, kernel_size_freq), smooth_sigma_freq)
    smoothed_freq_mask_norm = cv2.GaussianBlur(foreground_block_float_mask, (kernel_size_freq, kernel_size_freq), smooth_sigma_freq)
    smoothed_freq_mask_norm[smoothed_freq_mask_norm < 1e-6] = 1.0
    smoothed_frequency_map = smoothed_freq / smoothed_freq_mask_norm
    min_freq, max_freq = 1.0 / 25.0, 1.0 / 3.0
    smoothed_frequency_map = np.clip(smoothed_frequency_map, min_freq, max_freq, where=is_foreground_block, out=smoothed_frequency_map)
    smoothed_frequency_map[~is_foreground_block] = default_freq
    return smoothed_frequency_map

# ... (keep enhance_with_gabor as before) ...
def enhance_with_gabor(img, orientation_map, frequency_map, mask, ksize=GABOR_KERNEL_SIZE, sigma=GABOR_SIGMA, gamma=GABOR_GAMMA, psi=GABOR_PSI):
    h, w = img.shape
    enhanced_img = np.zeros_like(img, dtype=float)
    orientation_pixel = cv2.resize(orientation_map, (w, h), interpolation=cv2.INTER_NEAREST)
    frequency_pixel = cv2.resize(frequency_map, (w, h), interpolation=cv2.INTER_NEAREST)
    print("Applying Gabor filters (this may take some time)...")
    k_half = ksize // 2
    for r in range(h):
        for c in range(w):
            if mask[r, c] == 0: enhanced_img[r, c] = 255; continue
            angle = orientation_pixel[r, c] + (np.pi / 2.0)
            freq = frequency_pixel[r, c]
            wavelength = 1.0 / (freq + 1e-6)
            try: gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, angle, wavelength, gamma, psi, ktype=cv2.CV_64F)
            except cv2.error as gabor_err: enhanced_img[r,c] = img[r,c]; continue
            r_start = max(0, r - k_half); r_end = min(h, r + k_half + 1)
            c_start = max(0, c - k_half); c_end = min(w, c + k_half + 1)
            neighborhood = img[r_start:r_end, c_start:c_end].astype(float)
            kernel_r_start = k_half - (r - r_start); kernel_r_end = kernel_r_start + (r_end - r_start)
            kernel_c_start = k_half - (c - c_start); kernel_c_end = kernel_c_start + (c_end - c_start)
            try:
                adjusted_kernel = gabor_kernel[kernel_r_start:kernel_r_end, kernel_c_start:kernel_c_end]
                if neighborhood.shape == adjusted_kernel.shape: enhanced_img[r, c] = np.sum(neighborhood * adjusted_kernel)
                else: enhanced_img[r, c] = img[r, c]
            except Exception as filter_err: enhanced_img[r, c] = img[r, c]
    print("Gabor filtering complete.")
    min_val, max_val = np.min(enhanced_img), np.max(enhanced_img)
    if max_val - min_val < 1e-6: enhanced_img_norm = np.full_like(img, 128, dtype=np.uint8)
    else: enhanced_img_norm = cv2.normalize(enhanced_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return enhanced_img_norm.astype(np.uint8)

# ... (keep binarize_image as before) ...
def binarize_image(img, block_size=17, C=2):
    block_size = max(3, block_size if block_size % 2 != 0 else block_size + 1)
    try: binary_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)
    except cv2.error as bin_err:
        print(f"Error during adaptive thresholding: {bin_err}. Falling back to Otsu.")
        _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.sum(binary_img == 0) > np.sum(binary_img == 255): binary_img = cv2.bitwise_not(binary_img)
    return binary_img

# ... (keep thin_image as before - prioritizing ximgproc) ...
def thin_image(binary_img):
    if binary_img.dtype != np.uint8: binary_img = binary_img.astype(np.uint8)
    if np.max(binary_img) == 1: binary_img = (binary_img * 255).astype(np.uint8)
    non_zero_count = cv2.countNonZero(binary_img)
    total_pixels = binary_img.size
    zero_count = total_pixels - non_zero_count
    if non_zero_count > 0 and zero_count > non_zero_count:
        print("Warning: Input to thinning seems to have black ridges. Inverting image.")
        binary_img = cv2.bitwise_not(binary_img)
    elif non_zero_count == 0:
        print("Warning: Input image to thinning is entirely black. Returning empty image.")
        return binary_img
    thinned_img = None
    if XIMGPROC_AVAILABLE:
        try:
            thinned_img = cv2.ximgproc.thinning(binary_img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            print("DEBUG: Used cv2.ximgproc.thinning (Zhang-Suen).")
        except AttributeError:
             print("ERROR: cv2.ximgproc.thinning attribute not found despite XIMGPROC_AVAILABLE=True.")
             print("       Falling back to morphological thinning.")
             # Fallback logic below will execute
        except Exception as e:
            print(f"Error during cv2.ximgproc.thinning: {e}")
            print("Falling back to basic morphological thinning.")
    if thinned_img is None:
         print("DEBUG: Using fallback morphological thinning.")
         thinned_img_morph = binary_img.copy()
         skeleton = np.zeros(thinned_img_morph.shape, np.uint8)
         kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
         iterations = 0
         max_iterations = 150
         while iterations < max_iterations:
             eroded = cv2.erode(thinned_img_morph, kernel)
             temp = cv2.dilate(eroded, kernel)
             temp = cv2.bitwise_and(thinned_img_morph, cv2.bitwise_not(temp))
             skeleton = cv2.bitwise_or(skeleton, temp)
             thinned_img_morph = eroded.copy()
             if cv2.countNonZero(thinned_img_morph) == 0: break
             iterations += 1
         else: print(f"Warning: Morphological thinning reached max iterations ({max_iterations}).")
         thinned_img = skeleton
    return thinned_img

# --- Main Preprocessing Pipeline Function ---

def preprocess_fingerprint(image_path, display_steps=False, output_dir=None):
    """
    Applies the full preprocessing pipeline to a fingerprint image.

    Args:
        image_path (str): Path to the input fingerprint image.
        display_steps (bool): If True, displays intermediate images using Matplotlib.
        output_dir (str, optional): If provided, saves intermediate images to this directory.

    Returns:
        dict: A dictionary containing 'thinned_image', 'mask', and 'orientation_map',
              or None if a critical error occurs.
    """
    # ... (Setup: intermediate_results, start_time_total, output_dir check, save_or_show def) ...
    intermediate_results = {}
    start_time_total = time.time()
    if output_dir:
        try: os.makedirs(output_dir, exist_ok=True)
        except OSError as e: output_dir = None

    def save_or_show(img, name, step_num):
        # ... (save_or_show implementation remains the same) ...
        if img.dtype != np.uint8:
             if np.max(img) <= 1.0 and np.min(img) >= 0.0: img_display = (img * 255).astype(np.uint8)
             else: img_display = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else: img_display = img
        filename_base = os.path.splitext(os.path.basename(image_path))[0]
        safe_name = name.lower().replace(' ', '_').replace('/', '_')
        title = f"{step_num}: {name} ({filename_base})"
        intermediate_results[name] = img_display
        if output_dir:
            save_path = os.path.join(output_dir, f"{filename_base}_{step_num}_{safe_name}.png")
            try: cv2.imwrite(save_path, img_display)
            except Exception as write_err: print(f"Warning: Failed to save image {save_path}: {write_err}")
        if display_steps: plot_image(img_display, title)

    try:
        step_start_time = time.time()
        # 1. Load Image
        print("  Loading image...")
        img_gray = load_image(image_path)
        save_or_show(img_gray, "Original Grayscale", 1)
        print(f"    (Load time: {time.time() - step_start_time:.2f}s)")

        step_start_time = time.time()
        # 2. Normalization
        print("  Normalizing image...")
        img_norm = normalize_image(img_gray)
        save_or_show(img_norm, "Normalized", 2)
        print(f"    (Normalization time: {time.time() - step_start_time:.2f}s)")

        step_start_time = time.time()
        # 3. Segmentation
        print("  Segmenting fingerprint...")
        img_segmented, mask = segment_fingerprint(img_norm) # Get final cleaned mask
        save_or_show(img_segmented, "Segmented", 3)
        save_or_show((mask * 255), "Foreground Mask", 3)
        print(f"    (Segmentation time: {time.time() - step_start_time:.2f}s)")

        step_start_time = time.time()
        # 4. Orientation Field Estimation
        print("  Estimating orientation field...")
        orientation_map = estimate_orientation_field(img_norm, block_size=ORIENTATION_BLOCK_SIZE) # Get final smoothed block map
        print(f"    (Orientation estimation time: {time.time() - step_start_time:.2f}s)")

        step_start_time = time.time()
        # 5. Ridge Frequency Estimation
        print("  Estimating ridge frequency...")
        frequency_map = estimate_ridge_frequency(img_norm, orientation_map, mask, block_size=FREQUENCY_BLOCK_SIZE)
        print(f"    (Frequency estimation time: {time.time() - step_start_time:.2f}s)")

        step_start_time = time.time()
        # 6. Gabor Filtering Enhancement
        print("  Enhancing with Gabor filters...")
        img_enhanced = enhance_with_gabor(img_norm, orientation_map, frequency_map, mask)
        img_enhanced[mask == 0] = 255
        save_or_show(img_enhanced, "Gabor Enhanced", 6)
        print(f"    (Gabor enhancement time: {time.time() - step_start_time:.2f}s)")

        step_start_time = time.time()
        # 7. Binarization
        print("  Binarizing image...")
        img_binary = binarize_image(img_enhanced)
        img_binary[mask == 0] = 0
        save_or_show(img_binary, "Binarized", 7)
        print(f"    (Binarization time: {time.time() - step_start_time:.2f}s)")

        step_start_time = time.time()
        # 8. Thinning (Skeletonization)
        print("  Thinning image...")
        img_thinned = thin_image(img_binary)
        img_thinned[mask == 0] = 0
        save_or_show(img_thinned, "Thinned", 8)
        print(f"    (Thinning time: {time.time() - step_start_time:.2f}s)")

        if display_steps:
            print("Displaying intermediate steps...")
            if 'matplotlib' in sys.modules: plt.show()

        print(f"Preprocessing pipeline finished. Total time: {time.time() - start_time_total:.2f}s")

        # --- MODIFIED RETURN VALUE ---
        return {
            'thinned_image': img_thinned,
            'mask': mask, # Return the final cleaned mask used
            'orientation_map': orientation_map # Return the smoothed block map
        }
        # --- END MODIFICATION ---

    # ... (keep except blocks as before) ...
    except FileNotFoundError as e: print(f"[Error] in preprocess_fingerprint: {e}"); return None
    except ValueError as e: print(f"[Error] in preprocess_fingerprint: {e}"); return None
    except Exception as e:
        print(f"[Error] An unexpected error occurred during preprocessing pipeline: {e}")
        import traceback
        traceback.print_exc(); return None


# --- Standalone Test (Optional) ---
# ... (keep standalone test block as before) ...
if __name__ == "__main__":
    # ... (standalone test code remains the same, but check the return type) ...
    print("Running preprocessing module test...")
    _SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.abspath(os.path.join(_SRC_DIR, "../data"))
    test_results_dir = os.path.abspath(os.path.join(_SRC_DIR, "../results"))
    os.makedirs(test_data_dir, exist_ok=True)
    dummy_image_path = os.path.join(test_data_dir, "test_fingerprint.png")
    if not os.path.exists(dummy_image_path):
        print(f"Creating dummy test image at: {dummy_image_path}")
        dummy_img = np.full((200, 150), 200, dtype=np.uint8)
        cv2.line(dummy_img, (20, 30), (130, 50), 50, 5); cv2.line(dummy_img, (20, 70), (130, 90), 60, 4)
        cv2.line(dummy_img, (20, 110), (130, 130), 70, 6); cv2.line(dummy_img, (20, 150), (130, 170), 80, 5)
        dummy_img = cv2.GaussianBlur(dummy_img, (5, 5), 0)
        try: cv2.imwrite(dummy_image_path, dummy_img)
        except Exception as write_err: dummy_image_path = None
    if test_image and os.path.exists(test_image):
        output_folder = os.path.join(test_results_dir, "preprocessing_test")
        print(f"\nProcessing test image: {test_image}")
        result_dict = preprocess_fingerprint(test_image, display_steps=True, output_dir=output_folder) # Get dict
        if result_dict is not None and isinstance(result_dict, dict):
            print(f"\nPreprocessing completed successfully.")
            print(f"Returned keys: {result_dict.keys()}")
            print(f"Thinned image dimensions: {result_dict.get('thinned_image', np.array([])).shape}")
            print(f"Mask dimensions: {result_dict.get('mask', np.array([])).shape}")
            print(f"Orientation map dimensions: {result_dict.get('orientation_map', np.array([])).shape}")
            print(f"Intermediate results saved to: {output_folder}")
        else: print("\nPreprocessing failed.")
    else: print("\nCould not find or create test image. Skipping pipeline test.")
    print("\nPreprocessing module test finished.")