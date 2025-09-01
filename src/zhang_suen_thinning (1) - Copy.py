import numpy as np
import cv2

def zhang_suen_thinning(img):
    """
    Implementation of Zhang-Suen thinning algorithm
    Args:
        img: Binary image (0 or 255)
    Returns:
        Thinned binary image
    """
    # Convert to binary (0 or 1)
    img = (img / 255).astype(np.uint8)
    prev = np.zeros(img.shape, np.uint8)
    diff = None
    
    while True:
        # Step 1
        marker = np.zeros(img.shape, np.uint8)
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                p2 = img[i-1, j]
                p3 = img[i-1, j+1]
                p4 = img[i, j+1]
                p5 = img[i+1, j+1]
                p6 = img[i+1, j]
                p7 = img[i+1, j-1]
                p8 = img[i, j-1]
                p9 = img[i-1, j-1]
                
                # Condition 1: P1 is black
                if img[i, j] == 0:
                    continue
                
                # Condition 2: 2 <= B(P1) <= 6
                b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                if b < 2 or b > 6:
                    continue
                
                # Condition 3: A(P1) = 1
                a = 0
                if p2 == 0 and p3 == 1: a += 1
                if p3 == 0 and p4 == 1: a += 1
                if p4 == 0 and p5 == 1: a += 1
                if p5 == 0 and p6 == 1: a += 1
                if p6 == 0 and p7 == 1: a += 1
                if p7 == 0 and p8 == 1: a += 1
                if p8 == 0 and p9 == 1: a += 1
                if p9 == 0 and p2 == 1: a += 1
                
                if a != 1:
                    continue
                
                # Condition 4: P2 * P4 * P6 = 0
                if p2 * p4 * p6 != 0:
                    continue
                
                # Condition 5: P4 * P6 * P8 = 0
                if p4 * p6 * p8 != 0:
                    continue
                
                marker[i, j] = 1
        
        img = img * (1 - marker)
        
        # Step 2
        marker = np.zeros(img.shape, np.uint8)
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                p2 = img[i-1, j]
                p3 = img[i-1, j+1]
                p4 = img[i, j+1]
                p5 = img[i+1, j+1]
                p6 = img[i+1, j]
                p7 = img[i+1, j-1]
                p8 = img[i, j-1]
                p9 = img[i-1, j-1]
                
                # Condition 1: P1 is black
                if img[i, j] == 0:
                    continue
                
                # Condition 2: 2 <= B(P1) <= 6
                b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                if b < 2 or b > 6:
                    continue
                
                # Condition 3: A(P1) = 1
                a = 0
                if p2 == 0 and p3 == 1: a += 1
                if p3 == 0 and p4 == 1: a += 1
                if p4 == 0 and p5 == 1: a += 1
                if p5 == 0 and p6 == 1: a += 1
                if p6 == 0 and p7 == 1: a += 1
                if p7 == 0 and p8 == 1: a += 1
                if p8 == 0 and p9 == 1: a += 1
                if p9 == 0 and p2 == 1: a += 1
                
                if a != 1:
                    continue
                
                # Condition 4: P2 * P4 * P8 = 0
                if p2 * p4 * p8 != 0:
                    continue
                
                # Condition 5: P2 * P6 * P8 = 0
                if p2 * p6 * p8 != 0:
                    continue
                
                marker[i, j] = 1
        
        img = img * (1 - marker)
        
        # Check if the image has changed
        diff = np.sum(np.abs(prev - img))
        if diff == 0:
            break
        prev = img.copy()
    
    return img * 255

def compare_thinning_methods(img):
    """
    Compare Zhang-Suen thinning with OpenCV's thinning
    Args:
        img: Input binary image
    Returns:
        Dictionary with comparison metrics
    """
    # Zhang-Suen thinning
    zs_thinned = zhang_suen_thinning(img.copy())
    
    # OpenCV thinning (current method)
    cv_thinned = cv2.ximgproc.thinning(img.copy())
    
    # Calculate metrics
    metrics = {
        'zs_pixels': np.sum(zs_thinned > 0),
        'cv_pixels': np.sum(cv_thinned > 0),
        'zs_connectivity': calculate_connectivity(zs_thinned),
        'cv_connectivity': calculate_connectivity(cv_thinned),
        'zs_time': measure_thinning_time(img, zhang_suen_thinning),
        'cv_time': measure_thinning_time(img, cv2.ximgproc.thinning)
    }
    
    return metrics

def calculate_connectivity(img):
    """
    Calculate connectivity of thinned image
    """
    # Count number of endpoints and bifurcations
    endpoints = 0
    bifurcations = 0
    
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if img[i, j] > 0:
                neighbors = img[i-1:i+2, j-1:j+2].sum() - img[i, j]
                if neighbors == 1:
                    endpoints += 1
                elif neighbors > 2:
                    bifurcations += 1
    
    return {
        'endpoints': endpoints,
        'bifurcations': bifurcations,
        'total_features': endpoints + bifurcations
    }

def measure_thinning_time(img, thinning_func):
    """
    Measure execution time of thinning function
    """
    import time
    start_time = time.time()
    thinning_func(img.copy())
    return time.time() - start_time 