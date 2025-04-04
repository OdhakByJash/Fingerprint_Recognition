# src/matching.py

import math
import numpy as np
import json # For standalone test readability
 
# --- Configuration ---
# Thresholds for considering two minutiae as a potential match.
# Distance is measured in pixels. Angle is measured in degrees.
DISTANCE_THRESHOLD = 15  # Max distance between matched minutiae locations (pixels)
ANGLE_THRESHOLD = 15     # Max difference between matched minutiae angles (degrees) - ONLY used if angles are available

# --- Helper Functions ---

def calculate_distance(point1, point2):
    """Calculates Euclidean distance between two points."""
    # Ensure points have 'x' and 'y' keys
    if 'x' not in point1 or 'y' not in point1 or 'x' not in point2 or 'y' not in point2:
        print("Warning: Missing 'x' or 'y' coordinate in point for distance calculation.")
        return float('inf') # Return infinite distance if coords are missing
    dx = point1['x'] - point2['x']
    dy = point1['y'] - point2['y']
    return math.sqrt(dx**2 + dy**2)

def calculate_angle_difference(angle1_deg, angle2_deg):
    """
    Calculates the absolute difference between two angles (in degrees),
    handling wraparound (e.g., difference between 170 and 10 should be 20).
    Angles are assumed to be in the 0-180 range typical for fingerprint orientation.

    Returns:
        float: The angle difference (0-90).
        Returns 0.0 if either angle is None (treating missing angles as a non-blocking condition).
    """
    # --- MODIFIED TO HANDLE None ---
    # If either angle is None, we cannot compare. Return 0 difference
    # so it doesn't prevent a match based solely on missing angles.
    if angle1_deg is None or angle2_deg is None:
        return 0.0 # Treat as perfect angle match if angles are missing
    # --- END MODIFICATION ---

    try:
        # Ensure angles are numeric before calculation
        a1 = float(angle1_deg)
        a2 = float(angle2_deg)
        diff = abs(a1 - a2)
        # Handle wraparound (difference across the 0/180 boundary)
        return min(diff, 180 - diff)
    except (ValueError, TypeError):
        print(f"Warning: Could not convert angles ({angle1_deg}, {angle2_deg}) to float.")
        return 180.0 # Return max difference on error


# --- Basic Minutiae Matching Algorithm ---

def match_minutiae(minutiae_list1, minutiae_list2):
    """
    Performs a basic comparison between two sets of minutiae.

    Counts pairs of minutiae (one from each list) that are close in distance.
    Angle is only checked if BOTH minutiae have a non-None angle value.
    This simple algorithm doesn't perform optimal alignment.

    Args:
        minutiae_list1 (list): List of minutiae dictionaries from the first fingerprint.
                               Each dict should have 'x', 'y'. Optional: 'type', 'angle'.
        minutiae_list2 (list): List of minutiae dictionaries from the second fingerprint.

    Returns:
        int: The number of potentially matching minutiae pairs found based on thresholds.
        float: A normalized score (0-1), e.g., matches / min(len1, len2).
    """
    matches = 0
    # Keep track of minutiae in list2 that have been matched to prevent one-to-many matches
    used_indices_list2 = set()

    len1, len2 = len(minutiae_list1), len(minutiae_list2)

    # Handle empty lists
    if len1 == 0 or len2 == 0:
        return 0, 0.0 # No matches possible if either list is empty

    # Iterate through each minutia in the first list
    for i, m1 in enumerate(minutiae_list1):
        best_match_found_for_m1 = False
        min_dist = float('inf') # Track distance of the best match for m1
        best_match_idx = -1     # Track index in list2 of the best match for m1

        # Find the closest potential match in the second list
        for j, m2 in enumerate(minutiae_list2):
            # Skip if this minutia from list2 has already been used in a match
            if j in used_indices_list2:
                continue

            # 1. Check distance
            dist = calculate_distance(m1, m2)
            if dist < DISTANCE_THRESHOLD:
                # 2. Check angle difference *only if* both angles are available
                # --- MODIFIED CONDITION ---
                angles_available = m1.get('angle') is not None and m2.get('angle') is not None
                angle_match = False # Assume angles don't match unless proven otherwise

                if angles_available:
                    # Both angles exist, calculate difference and compare to threshold
                    angle_diff = calculate_angle_difference(m1.get('angle'), m2.get('angle'))
                    if angle_diff < ANGLE_THRESHOLD:
                        angle_match = True # Angles match
                else:
                    # At least one angle is None, so we skip the angle check
                    # and consider it an "angle match" by default.
                    angle_match = True
                # --- END MODIFICATION ---

                # 3. Check if this is a potential match (passes distance and angle conditions)
                if angle_match:
                    # Is it the *closest* potential match found *so far* for m1?
                    # This greedy approach helps prevent one minutia matching multiple others.
                    if dist < min_dist:
                         min_dist = dist
                         best_match_idx = j
                         best_match_found_for_m1 = True # Mark that we found at least one potential match for m1

        # After checking all m2 for the current m1:
        # If a best match was found for m1 within thresholds, count it and mark m2 as used
        if best_match_found_for_m1:
            # Ensure the best_match_idx is valid and hasn't been sneakily used
            # (Shouldn't happen with the check at the start of the inner loop, but belt-and-suspenders)
            if best_match_idx != -1 and best_match_idx not in used_indices_list2:
                 matches += 1
                 used_indices_list2.add(best_match_idx) # Mark the matched minutia in list2 as used

    # Calculate a normalized score (e.g., based on the smaller minutiae set size)
    min_len = min(len1, len2)
    # Avoid division by zero if min_len is 0 (though handled by initial check)
    normalized_score = float(matches) / min_len if min_len > 0 else 0.0

    # Clamp score to be between 0 and 1 (shouldn't be necessary if logic is correct, but safe)
    normalized_score = max(0.0, min(1.0, normalized_score))

    # Return raw match count and normalized score
    return matches, normalized_score


# --- Standalone Test (Optional) ---
if __name__ == "__main__":
    print("Running matching module test...")

    # --- Test Case 1: Similar lists with angles ---
    m_list_a = [
        {'x': 100, 'y': 100, 'type': 'ending', 'angle': 30},
        {'x': 150, 'y': 200, 'type': 'bifurcation', 'angle': 120},
        {'x': 80, 'y': 250, 'type': 'ending', 'angle': 90},
    ]
    m_list_b = [
        {'x': 105, 'y': 102, 'type': 'ending', 'angle': 35},      # Should match m_list_a[0]
        {'x': 148, 'y': 195, 'type': 'bifurcation', 'angle': 110},# Should match m_list_a[1]
        {'x': 250, 'y': 100, 'type': 'ending', 'angle': 45},      # Should not match anything in A
    ]
    print(f"\nComparing List A vs List B (Angles Present):")
    count_ab, score_ab = match_minutiae(m_list_a, m_list_b)
    print(f"  Raw Matches: {count_ab}") # Expected: 2
    print(f"  Normalized Score: {score_ab:.4f}")
    assert count_ab == 2, "Test A vs B failed: Expected 2 matches"

    # --- Test Case 2: Similar lists, NO angles (angle=None) ---
    m_list_a_no_angle = [
        {'x': 100, 'y': 100, 'type': 'ending', 'angle': None},
        {'x': 150, 'y': 200, 'type': 'bifurcation', 'angle': None},
        {'x': 80, 'y': 250, 'type': 'ending', 'angle': None},
    ]
    m_list_b_no_angle = [
        {'x': 105, 'y': 102, 'type': 'ending', 'angle': None},      # Should match based on distance
        {'x': 148, 'y': 195, 'type': 'bifurcation', 'angle': None},# Should match based on distance
        {'x': 250, 'y': 100, 'type': 'ending', 'angle': None},
    ]
    print(f"\nComparing List A vs List B (Angles = None):")
    count_ab_na, score_ab_na = match_minutiae(m_list_a_no_angle, m_list_b_no_angle)
    print(f"  Raw Matches: {count_ab_na}") # Expected: 2
    print(f"  Normalized Score: {score_ab_na:.4f}")
    assert count_ab_na == 2, "Test A vs B (No Angle) failed: Expected 2 matches"

    # --- Test Case 3: Identical lists, NO angles (self-match test) ---
    m_list_self_no_angle = [
        {'x': 100, 'y': 100, 'type': 'ending', 'angle': None},
        {'x': 150, 'y': 200, 'type': 'bifurcation', 'angle': None},
        {'x': 80, 'y': 250, 'type': 'ending', 'angle': None},
    ]
    print(f"\nComparing List Self vs Self (Angles = None):")
    count_self_na, score_self_na = match_minutiae(m_list_self_no_angle, m_list_self_no_angle)
    print(f"  Raw Matches: {count_self_na}") # Expected: 3
    print(f"  Normalized Score: {score_self_na:.4f}") # Expected: 1.0
    assert count_self_na == 3, "Test Self vs Self (No Angle) failed: Expected 3 matches"
    assert abs(score_self_na - 1.0) < 1e-6, "Test Self vs Self (No Angle) failed: Expected score 1.0"


    # --- Test Case 4: Different lists ---
    m_list_c = [
        {'x': 500, 'y': 500, 'type': 'ending', 'angle': 0},
        {'x': 600, 'y': 600, 'type': 'bifurcation', 'angle': 90},
    ]
    print(f"\nComparing List A vs List C (Different):")
    count_ac, score_ac = match_minutiae(m_list_a, m_list_c)
    print(f"  Raw Matches: {count_ac}") # Expected: 0
    print(f"  Normalized Score: {score_ac:.4f}")
    assert count_ac == 0, "Test A vs C failed: Expected 0 matches"

    # --- Test Case 5: Empty list ---
    m_list_d = []
    print(f"\nComparing List A vs List D (Empty):")
    count_ad, score_ad = match_minutiae(m_list_a, m_list_d)
    print(f"  Raw Matches: {count_ad}") # Expected: 0
    print(f"  Normalized Score: {score_ad:.4f}") # Expected: 0.0
    assert count_ad == 0 and score_ad == 0.0, "Test A vs D failed: Expected 0 score"


    print("\nMatching module test finished.")