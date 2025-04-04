# src/utils.py

import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os 

# --- Image Plotting ---

def plot_image(img, title="Image", cmap='gray'):
    """
    Displays an image using Matplotlib.

    Args:
        img (numpy.ndarray): The image to display.
        title (str): The title for the image window.
        cmap (str): The colormap to use (e.g., 'gray', 'jet').
                    Defaults to 'gray'.
    """
    plt.figure(figsize=(6, 6)) # Adjust figure size as needed
    # Check if image is color or grayscale for imshow
    if len(img.shape) == 3 and img.shape[2] == 3:
        # It's a color image (BGR from OpenCV), convert to RGB for Matplotlib
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        # Grayscale image
        plt.imshow(img, cmap=cmap)

    plt.title(title)
    plt.axis('off') # Hide axes ticks and labels
    # Note: plt.show() should be called *after* all plots are created in the main script,
    #       unless you want each plot to block execution until closed.


# --- Orientation Field Visualization (Optional Utility) ---

def visualize_orientation_field(image, orientation_map_rad, mask, block_size, line_length=8, skip_step=2, output_path=None):
    vis_img = None
    # Ensure the image is BGR for colored drawing
    if len(image.shape) == 2 or image.shape[2] == 1:
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = image.copy()

    rows_blk, cols_blk = orientation_map_rad.shape
    h, w = image.shape[:2] # Get height and width

    for r_blk in range(0, rows_blk, skip_step):
        for c_blk in range(0, cols_blk, skip_step):
            # Calculate center of the block
            center_y = int((r_blk + 0.5) * block_size)
            center_x = int((c_blk + 0.5) * block_size)

            # Check if the block center is within image bounds and foreground mask
            if 0 <= center_y < h and 0 <= center_x < w and mask[center_y, center_x] != 0:
                angle_rad = orientation_map_rad[r_blk, c_blk]

                # Calculate line endpoints based on angle
                # Angle 0 should point right. Angle pi/2 should point down (image coords).
                dx = line_length * math.cos(angle_rad)
                dy = line_length * math.sin(angle_rad)

                # Start and end points for the line segment centered at the block center
                x1 = int(center_x - dx / 2)
                y1 = int(center_y - dy / 2)
                x2 = int(center_x + dx / 2)
                y2 = int(center_y + dy / 2)

                # Draw the line (e.g., in Red)
                cv2.line(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 1) # Red line, thickness 1

    # Display or save
    if output_path:
        cv2.imwrite(output_path, vis_img)
        print(f"Orientation field visualization saved to: {output_path}")
    else:
        plot_image(vis_img, "Orientation Field Visualization")
        # Note: Requires plt.show() call in the calling script


# --- Other Potential Utilities (Add as needed) ---

# Example: Function to save numpy arrays
def save_numpy_array(data, filename):
    """Saves a NumPy array to a .npy file."""
    try:
        np.save(filename, data)
        print(f"NumPy array saved to: {filename}")
    except Exception as e:
        print(f"Error saving NumPy array to {filename}: {e}")

# Example: Function to load numpy arrays
def load_numpy_array(filename):
    """Loads a NumPy array from a .npy file."""
    if not os.path.exists(filename):
        print(f"Error: NumPy file not found at {filename}")
        return None
    try:
        data = np.load(filename)
        print(f"NumPy array loaded from: {filename}")
        return data
    except Exception as e:
        print(f"Error loading NumPy array from {filename}: {e}")
        return None


# --- Standalone Test (Optional) ---
if __name__ == "__main__":
    print("Running utils module test...")

    # Test plot_image
    dummy_gray = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
    dummy_color = cv2.cvtColor(dummy_gray, cv2.COLOR_GRAY2BGR) # Create a dummy color image
    dummy_color[10:30, 20:50] = [0, 0, 255] # Add a red patch

    plot_image(dummy_gray, "Test Grayscale Image")
    plot_image(dummy_color, "Test Color Image")

    print("Displaying test images (close windows to continue)...")
    plt.show() # Show the plots generated above

    # Test orientation visualization (using dummy data)
    h, w = 100, 150
    block_size = 16
    dummy_orient_map = np.random.rand(h // block_size, w // block_size) * np.pi - (np.pi / 2) # Random angles -pi/2 to pi/2
    dummy_mask = np.ones((h, w), dtype=np.uint8) # Mask covering everything
    dummy_mask[0:20, :] = 0 # Make top part background

    print("Visualizing dummy orientation field...")
    visualize_orientation_field(dummy_gray, dummy_orient_map, dummy_mask, block_size)
    plt.show()

    # Test save/load numpy array
    test_array = np.array([[1, 2, 3], [4, 5, 6]])
    test_filename = "../results/test_array.npy"
    if not os.path.exists("../results"): os.makedirs("../results")
    save_numpy_array(test_array, test_filename)
    loaded_array = load_numpy_array(test_filename)
    if loaded_array is not None:
        print("Loaded array:\n", loaded_array)
        assert np.array_equal(test_array, loaded_array)
        os.remove(test_filename) # Clean up test file

    print("\nUtils module test finished.")