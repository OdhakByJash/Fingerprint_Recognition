import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

def extract_ridge_curves(img, orientation_map, block_size=16):
    """
    Extract ridge curves using RANSAC
    Args:
        img: Thinned binary image
        orientation_map: Orientation map
        block_size: Size of blocks for curve fitting
    Returns:
        List of ridge curves
    """
    curves = []
    height, width = img.shape
    
    # Divide image into blocks
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = img[y:y+block_size, x:x+block_size]
            orientation = orientation_map[y:y+block_size, x:x+block_size]
            
            # Get ridge points in block
            points = np.where(block > 0)
            if len(points[0]) < 3:  # Need at least 3 points for curve fitting
                continue
            
            # Prepare data for RANSAC
            X = points[1].reshape(-1, 1)  # x coordinates
            y = points[0].reshape(-1, 1)  # y coordinates
            
            # Create polynomial features
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            
            # Fit curve using RANSAC
            ransac = RANSACRegressor(
                base_estimator=make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
                min_samples=3,
                residual_threshold=1.0,
                max_trials=100
            )
            
            try:
                ransac.fit(X_poly, y)
                inlier_mask = ransac.inlier_mask_
                
                if np.sum(inlier_mask) > 3:  # Only keep curves with enough inliers
                    curve_points = np.column_stack((X[inlier_mask], y[inlier_mask]))
                    curves.append({
                        'points': curve_points,
                        'model': ransac.estimator_,
                        'orientation': np.mean(orientation)
                    })
            except:
                continue
    
    return curves

def find_minutiae_from_curves(curves, img):
    """
    Find minutiae points from ridge curves
    Args:
        curves: List of ridge curves
        img: Original binary image
    Returns:
        List of minutiae points
    """
    minutiae = []
    
    for curve in curves:
        points = curve['points']
        
        # Find endpoints
        if len(points) > 0:
            start_point = points[0]
            end_point = points[-1]
            
            # Check if points are actual endpoints
            if is_endpoint(img, start_point):
                minutiae.append({
                    'type': 'endpoint',
                    'position': start_point,
                    'orientation': curve['orientation']
                })
            
            if is_endpoint(img, end_point):
                minutiae.append({
                    'type': 'endpoint',
                    'position': end_point,
                    'orientation': curve['orientation']
                })
        
        # Find bifurcations
        for i in range(1, len(points)-1):
            point = points[i]
            if is_bifurcation(img, point):
                minutiae.append({
                    'type': 'bifurcation',
                    'position': point,
                    'orientation': curve['orientation']
                })
    
    return minutiae

def is_endpoint(img, point):
    """
    Check if a point is an endpoint
    """
    x, y = int(point[0]), int(point[1])
    if x <= 0 or y <= 0 or x >= img.shape[1]-1 or y >= img.shape[0]-1:
        return False
    
    # Count neighbors
    neighbors = img[y-1:y+2, x-1:x+2].sum() - img[y, x]
    return neighbors == 1

def is_bifurcation(img, point):
    """
    Check if a point is a bifurcation
    """
    x, y = int(point[0]), int(point[1])
    if x <= 0 or y <= 0 or x >= img.shape[1]-1 or y >= img.shape[0]-1:
        return False
    
    # Count neighbors
    neighbors = img[y-1:y+2, x-1:x+2].sum() - img[y, x]
    return neighbors > 2

def compare_feature_extraction(img, orientation_map):
    """
    Compare RANSAC-based feature extraction with current method
    Args:
        img: Thinned binary image
        orientation_map: Orientation map
    Returns:
        Dictionary with comparison metrics
    """
    import time
    from feature_extraction import extract_minutiae  # Import current method
    
    # Current method
    start_time = time.time()
    current_minutiae = extract_minutiae(img, orientation_map)
    current_time = time.time() - start_time
    
    # RANSAC method
    start_time = time.time()
    curves = extract_ridge_curves(img, orientation_map)
    ransac_minutiae = find_minutiae_from_curves(curves, img)
    ransac_time = time.time() - start_time
    
    # Calculate metrics
    metrics = {
        'current_minutiae_count': len(current_minutiae),
        'ransac_minutiae_count': len(ransac_minutiae),
        'current_time': current_time,
        'ransac_time': ransac_time,
        'current_endpoints': sum(1 for m in current_minutiae if m['type'] == 'endpoint'),
        'current_bifurcations': sum(1 for m in current_minutiae if m['type'] == 'bifurcation'),
        'ransac_endpoints': sum(1 for m in ransac_minutiae if m['type'] == 'endpoint'),
        'ransac_bifurcations': sum(1 for m in ransac_minutiae if m['type'] == 'bifurcation')
    }
    
    return metrics 