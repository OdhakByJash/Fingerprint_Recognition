# src/feature_extraction.py

import numpy as np
import cv2
import json
import math
import os
import matplotlib.pyplot as plt # For visualization

# --- Configuration for Filtering ---
# Minutiae closer than this distance to the mask border will be removed
MASK_BORDER_MARGIN = 10
# Minutiae pairs (any type) closer than this distance will be removed
MIN_DISTANCE_THRESHOLD = 6 # Pixels - adjust based on resolution and skeleton quality

# --- Helper Functions ---

def calculate_cn(window):
    """Calculates Crossing Number (CN) for a 3x3 binary window (1=ridge)."""
    window_bin = (window > 0).astype(int)
    p = [window_bin[0, 1], window_bin[0, 2], window_bin[1, 2], window_bin[2, 2],
         window_bin[2, 1], window_bin[2, 0], window_bin[1, 0], window_bin[0, 0]]
    cn = 0.5 * sum(abs(p[i] - p[(i + 1) % 8]) for i in range(8))
    return cn

def get_angle_at(x, y, orientation_map_rad, block_size):
    """
    Retrieves the smoothed orientation angle (in degrees, 0-180) for a minutia.
    Maps pixel coordinates to the block-based orientation map.
    """
    rows_blk, cols_blk = orientation_map_rad.shape
    # Map pixel coordinates to block indices, clamping to valid range
    block_r = min(max(0, y // block_size), rows_blk - 1)
    block_c = min(max(0, x // block_size), cols_blk - 1)

    # Retrieve angle in radians from the smoothed map
    angle_rad = orientation_map_rad[block_r, block_c]

    # Convert radians (-pi/2 to pi/2) to degrees (0 to 180)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 180.0

    return round(angle_deg, 2) # Return rounded degrees

def _is_near_mask_border(x, y, mask, margin=MASK_BORDER_MARGIN):
    """Checks if a point (x, y) is within 'margin' pixels of the mask border (value 0)."""
    h, w = mask.shape
    # Define local neighborhood to check
    y_start, y_end = max(0, y - margin), min(h, y + margin + 1)
    x_start, x_end = max(0, x - margin), min(w, x + margin + 1)
    neighborhood = mask[y_start:y_end, x_start:x_end]
    # If any pixel in the neighborhood is background (0), the point is near the border
    return 0 in neighborhood

# --- Minutiae Filtering Functions ---

def filter_spurious_minutiae(minutiae_list, mask):
    """
    Applies post-processing filters to remove spurious minutiae.

    Args:
        minutiae_list (list): The raw list of detected minutiae dictionaries.
        mask (numpy.ndarray): The foreground mask (1=foreground, 0=background).

    Returns:
        list: The filtered list of minutiae dictionaries.
    """
    if not minutiae_list:
        return []

    print(f"  Filtering {len(minutiae_list)} raw minutiae...")
    filtered_minutiae = minutiae_list[:] # Work on a copy

    # --- Filter 1: Remove minutiae too close to the mask border ---
    print(f"    Applying mask border filter (margin={MASK_BORDER_MARGIN})...")
    minutiae_after_border_filter = []
    removed_border = 0
    for m in filtered_minutiae:
        if not _is_near_mask_border(m['x'], m['y'], mask, MASK_BORDER_MARGIN):
            minutiae_after_border_filter.append(m)
        else:
            removed_border += 1
    print(f"      Removed {removed_border} minutiae near mask border.")
    filtered_minutiae = minutiae_after_border_filter
    if not filtered_minutiae: return [] # Stop if all removed

    # --- Filter 2: Remove minutiae pairs that are too close ---
    print(f"    Applying distance filter (min_distance={MIN_DISTANCE_THRESHOLD})...")
    n = len(filtered_minutiae)
    # Create distance matrix (consider optimizing for large N if needed)
    dist_matrix = np.full((n, n), float('inf'))
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt((filtered_minutiae[i]['x'] - filtered_minutiae[j]['x'])**2 +
                             (filtered_minutiae[i]['y'] - filtered_minutiae[j]['y'])**2)
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    # Identify indices of minutiae to remove
    indices_to_remove = set()
    for i in range(n):
        if i in indices_to_remove: continue # Skip if already marked for removal
        for j in range(i + 1, n):
            if j in indices_to_remove: continue
            if dist_matrix[i, j] < MIN_DISTANCE_THRESHOLD:
                # Found a pair too close, mark both for removal
                indices_to_remove.add(i)
                indices_to_remove.add(j)

    # Create the final list excluding the removed indices
    minutiae_after_distance_filter = []
    for i, m in enumerate(filtered_minutiae):
        if i not in indices_to_remove:
            minutiae_after_distance_filter.append(m)

    removed_distance = len(filtered_minutiae) - len(minutiae_after_distance_filter)
    print(f"      Removed {removed_distance} minutiae pairs too close.")
    filtered_minutiae = minutiae_after_distance_filter

    # --- Add more advanced filters here if needed (e.g., short bridge/spur removal) ---

    print(f"  Filtering complete. {len(filtered_minutiae)} minutiae remaining.")
    return filtered_minutiae


# --- Core Feature Extraction ---

def extract_minutiae(thinned_img, orientation_map, mask, block_size):
    """
    Extracts, calculates angles, and filters minutiae points.

    Args:
        thinned_img (numpy.ndarray): Thinned binary image (ridges are 255).
        orientation_map (numpy.ndarray): Smoothed orientation field in radians (block map).
        mask (numpy.ndarray): Foreground mask (1=foreground, 0=background).
        block_size (int): Block size used for orientation map calculation.

    Returns:
        list: A filtered list of minutiae dictionaries with 'x', 'y', 'type', 'angle'.
    """
    h, w = thinned_img.shape
    if np.max(thinned_img) == 1: thinned_img = (thinned_img * 255).astype(np.uint8)

    raw_minutiae = []
    print(" Extracting raw minutiae points using Crossing Number...")
    # Iterate checking CN, avoid image borders where CN is unreliable
    border = 5 # Stricter border margin for CN calculation itself
    for y in range(border, h - border):
        for x in range(border, w - border):
            if thinned_img[y, x] == 255: # Check only ridge pixels
                window = thinned_img[y-1 : y+2, x-1 : x+2]
                cn = calculate_cn(window)
                minutia_type = None
                if cn == 1: minutia_type = 'ending'
                elif cn == 3: minutia_type = 'bifurcation'

                if minutia_type:
                    # Calculate angle using the orientation map
                    try:
                        angle = get_angle_at(x, y, orientation_map, block_size)
                    except Exception as angle_err:
                        print(f"Warning: Error calculating angle at ({x},{y}): {angle_err}. Setting angle to None.")
                        angle = None

                    raw_minutiae.append({'x': x, 'y': y, 'type': minutia_type, 'angle': angle})

    print(f" Extracted {len(raw_minutiae)} raw minutiae points.")

    # Apply post-processing filters
    filtered_minutiae = filter_spurious_minutiae(raw_minutiae, mask)

    return filtered_minutiae


# --- Template Creation ---
# ... (create_template function remains the same) ...
def create_template(minutiae_list, image_width=None, image_height=None):
    if not minutiae_list and minutiae_list is not None : # Allow empty list but not None
        print("Warning: Creating template from empty minutiae list.")

    template = {
        "metadata": { "image_width": image_width, "image_height": image_height },
        "features": { "minutiae": minutiae_list, "num_minutiae": len(minutiae_list)}
    }
    try:
        template_json = json.dumps(template, indent=2)
        return template_json
    except TypeError as e:
        print(f"Error serializing template to JSON: {e}")
        return None


# --- Visualization ---
# ... (visualize_minutiae function remains the same) ...
def visualize_minutiae(image, minutiae_list, title_suffix="", save_path=None):
    if len(image.shape) == 2 or image.shape[2] == 1: vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else: vis_img = image.copy()
    ending_color = (0, 0, 255); bifurcation_color = (0, 255, 0); radius = 3; thickness = 1; angle_line_length = 10
    for m in minutiae_list:
        x, y = m['x'], m['y']; color = ending_color if m.get('type') == 'ending' else bifurcation_color
        cv2.circle(vis_img, (x, y), radius, color, thickness)
        if m.get('angle') is not None:
            angle_rad = math.radians(m['angle'])
            end_x = int(x + angle_line_length * math.cos(angle_rad))
            end_y = int(y + angle_line_length * math.sin(angle_rad)) # Standard angle convention
            cv2.line(vis_img, (x, y), (end_x, end_y), color, thickness)
    title = f"Minutiae Visualization ({len(minutiae_list)} points){title_suffix}"
    if save_path:
        try: cv2.imwrite(save_path, vis_img); print(f"Minutiae visualization saved to: {save_path}")
        except Exception as write_err: print(f"Error saving visualization {save_path}: {write_err}")
    else:
        try:
             import matplotlib.pyplot as plt
             plt.figure(figsize=(10, 10)); plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
             plt.title(title); plt.axis('off'); plt.show()
        except ImportError: print("Info: Matplotlib not found. Cannot display visualization.")


# --- Standalone Test ---
if __name__ == "__main__":
    # ... (Keep or enhance the standalone test block) ...
    print("Running feature extraction module test...")
    # Create dummy data (thinned image, orientation map, mask)
    h, w = 100, 150; block_size_test = 16
    dummy_thinned = np.zeros((h, w), dtype=np.uint8)
    dummy_mask = np.zeros((h, w), dtype=np.uint8)
    dummy_orient_map = np.random.rand(h // block_size_test, w // block_size_test) * np.pi - (np.pi / 2)
    # Draw some features on thinned and set mask
    cv2.line(dummy_thinned, (20, 30), (40, 30), 255, 1) # Ending
    cv2.line(dummy_thinned, (60, 50), (80, 50), 255, 1) # Bifurcation base
    cv2.line(dummy_thinned, (70, 50), (70, 70), 255, 1) # Bifurcation branch
    cv2.line(dummy_thinned, (5, 5), (15, 5), 255, 1) # Near border ending (should be removed)
    cv2.line(dummy_thinned, (30, 80), (33, 83), 255, 1) # Close pair 1
    cv2.line(dummy_thinned, (35, 85), (38, 88), 255, 1) # Close pair 2
    # Set mask (e.g., a rectangle inset from borders)
    dummy_mask[5:h-5, 5:w-5] = 1

    print("Processing dummy image...")
    minutiae_points = extract_minutiae(dummy_thinned, dummy_orient_map, dummy_mask, block_size_test)

    print("\nFiltered Minutiae:")
    print(json.dumps(minutiae_points, indent=2))

    # Visualize
    visualize_minutiae(dummy_thinned, minutiae_points, title_suffix=" (Filtered)")

    print("\nFeature extraction module test finished.")