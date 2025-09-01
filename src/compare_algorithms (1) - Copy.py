import cv2
import numpy as np
from preprocessing import preprocess_fingerprint
from zhang_suen_thinning import zhang_suen_thinning, compare_thinning_methods
from ransac_feature_extraction import compare_feature_extraction
import matplotlib.pyplot as plt
import time

def run_comparison(image_path):
    """
    Run comparison of different algorithms
    Args:
        image_path: Path to fingerprint image
    """
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Preprocess image
    normalized_img, segmented_img, enhanced_img, orientation_map = preprocess_fingerprint(img)
    
    # Compare thinning methods
    print("\n=== Thinning Algorithm Comparison ===")
    thinning_metrics = compare_thinning_methods(enhanced_img)
    print(f"Zhang-Suen Pixels: {thinning_metrics['zs_pixels']}")
    print(f"OpenCV Pixels: {thinning_metrics['cv_pixels']}")
    print(f"Zhang-Suen Time: {thinning_metrics['zs_time']:.4f}s")
    print(f"OpenCV Time: {thinning_metrics['cv_time']:.4f}s")
    print("\nZhang-Suen Connectivity:")
    print(f"  Endpoints: {thinning_metrics['zs_connectivity']['endpoints']}")
    print(f"  Bifurcations: {thinning_metrics['zs_connectivity']['bifurcations']}")
    print("\nOpenCV Connectivity:")
    print(f"  Endpoints: {thinning_metrics['cv_connectivity']['endpoints']}")
    print(f"  Bifurcations: {thinning_metrics['cv_connectivity']['bifurcations']}")
    
    # Compare feature extraction methods
    print("\n=== Feature Extraction Comparison ===")
    feature_metrics = compare_feature_extraction(enhanced_img, orientation_map)
    print(f"Current Method Minutiae: {feature_metrics['current_minutiae_count']}")
    print(f"RANSAC Method Minutiae: {feature_metrics['ransac_minutiae_count']}")
    print(f"Current Method Time: {feature_metrics['current_time']:.4f}s")
    print(f"RANSAC Method Time: {feature_metrics['ransac_time']:.4f}s")
    print("\nCurrent Method Features:")
    print(f"  Endpoints: {feature_metrics['current_endpoints']}")
    print(f"  Bifurcations: {feature_metrics['current_bifurcations']}")
    print("\nRANSAC Method Features:")
    print(f"  Endpoints: {feature_metrics['ransac_endpoints']}")
    print(f"  Bifurcations: {feature_metrics['ransac_bifurcations']}")
    
    # Visualize results
    visualize_comparison(img, enhanced_img, thinning_metrics, feature_metrics)

def visualize_comparison(original_img, enhanced_img, thinning_metrics, feature_metrics):
    """
    Visualize comparison results
    """
    plt.figure(figsize=(15, 10))
    
    # Original and enhanced images
    plt.subplot(2, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(2, 2, 2)
    plt.imshow(enhanced_img, cmap='gray')
    plt.title('Enhanced Image')
    
    # Thinning comparison
    plt.subplot(2, 2, 3)
    bars = plt.bar(['Zhang-Suen', 'OpenCV'], 
                  [thinning_metrics['zs_time'], thinning_metrics['cv_time']])
    plt.title('Thinning Algorithm Performance')
    plt.ylabel('Execution Time (s)')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s', ha='center', va='bottom')
    
    # Feature extraction comparison
    plt.subplot(2, 2, 4)
    bars = plt.bar(['Current', 'RANSAC'], 
                  [feature_metrics['current_time'], feature_metrics['ransac_time']])
    plt.title('Feature Extraction Performance')
    plt.ylabel('Execution Time (s)')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png')
    plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python compare_algorithms.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    run_comparison(image_path) 