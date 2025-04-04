# src/identify.py

import argparse
import os
import sys
import time
import traceback
import json

# --- Add src directory to Python path ---
try:
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path: sys.path.append(src_dir)
except Exception as e: print(f"Error modifying sys.path: {e}"); sys.exit(1)

# --- Import Project Modules ---
try:
    # === IMPORTING SHARED MODULES ===
    # These imports ensure the *same* functions are used as in enroll.py/verify.py
    from preprocessing import preprocess_fingerprint, ORIENTATION_BLOCK_SIZE
    from feature_extraction import extract_minutiae, visualize_minutiae # Using enhanced version
    from database_operations import get_all_templates, DB_PATH, initialize_database
    from matching import match_minutiae # Using version handling None angles
    from utils import plot_image
    # === END IMPORTS ===
except ImportError as e: print(f"Error importing project module: {e}"); traceback.print_exc(); sys.exit(1)
except Exception as e: print(f"An unexpected error during imports: {e}"); traceback.print_exc(); sys.exit(1)

try: import matplotlib.pyplot as plt
except ImportError: plt = None

# --- Identification Configuration ---
# Threshold tuning is CRITICAL after improving features.
IDENTIFICATION_THRESHOLD_RAW_COUNT = 8 # Example: Adjust based on testing

# --- Main Identification Function ---
def identify_fingerprint(image_path, display_steps=False):
    """
    Identifies a fingerprint using the consistent, enhanced feature extraction pipeline.
    """
    start_time = time.time()
    print(f"\n{'='*15} Starting Identification Process {'='*15}")
    abs_image_path = os.path.abspath(image_path); print(f"Identification Image: {abs_image_path}")
    abs_db_path = os.path.abspath(DB_PATH); print(f"Database:     {abs_db_path}")

    if not os.path.exists(image_path): print(f"\n[Error] Image file not found: {image_path}"); return None, 0.0

    # --- Step 1: Preprocessing (Using SHARED, Improved Function) ---
    print("\n[Step 1/3] Preprocessing input image...")
    preprocess_start = time.time()
    try:
        results_dir = os.path.abspath(os.path.join(src_dir, "../results"))
        os.makedirs(results_dir, exist_ok=True)
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        preprocess_output_dir = os.path.join(results_dir, f"identify_preprocess_{image_basename}")

        # === CALLING SHARED PREPROCESSING ===
        preprocess_result = preprocess_fingerprint(image_path, display_steps=display_steps, output_dir=preprocess_output_dir)
        # === END CALL ===

        preprocess_end = time.time()
        # --- HANDLE DICTIONARY RETURN (Consistent with enroll/verify) ---
        if preprocess_result is None: print("[Error] Preprocessing failed."); return None, 0.0
        thinned_img = preprocess_result.get('thinned_image')
        mask = preprocess_result.get('mask')
        orientation_map = preprocess_result.get('orientation_map')
        if thinned_img is None or mask is None or orientation_map is None:
             print("[Error] Preprocessing missing required data."); return None, 0.0
        # --- END HANDLING ---
        print(f"Preprocessing complete. (Time: {preprocess_end - preprocess_start:.2f}s)")
        if display_steps and plt: plt.show()

    except Exception as e: print(f"[Error] Preprocessing exception:"); traceback.print_exc(); return None, 0.0

    # --- Step 2: Feature Extraction (Using SHARED, Improved Function) ---
    print("\n[Step 2/3] Extracting and filtering features from input image...")
    feature_start = time.time()
    input_minutiae = []
    try:
        # === CALLING SHARED FEATURE EXTRACTION ===
        # Passes the mask and orientation map needed for angle calculation and filtering
        input_minutiae = extract_minutiae(thinned_img, orientation_map, mask, ORIENTATION_BLOCK_SIZE)
        # === END CALL ===
        feature_end = time.time()

        if not input_minutiae: print("[Warning] No valid minutiae extracted. Cannot identify."); return None, 0.0
        print(f"Extracted and filtered {len(input_minutiae)} minutiae points. (Time: {feature_end - feature_start:.2f}s)")

        if display_steps and input_minutiae and plt:
             visualize_minutiae(thinned_img, input_minutiae, title_suffix=" (Identify Input - Filtered)")
             plt.show()

    except Exception as e: print(f"[Error] Feature extraction exception:"); traceback.print_exc(); return None, 0.0

    # --- Step 3: Retrieve ALL Stored Templates and Match ---
    print("\n[Step 3/3] Retrieving all stored templates and matching...")
    db_match_start = time.time()
    try:
        all_stored_data = get_all_templates() # Uses shared DB operations

        if all_stored_data is None: print("[Error] DB error retrieving templates."); return None, 0.0
        if not all_stored_data: print("[Error] No templates in database."); return None, 0.0

        best_overall_score = -1 # Raw count
        best_match_user_id = None
        best_match_fp_id = None

        print(f"Comparing input ({len(input_minutiae)} minutiae) against {len(all_stored_data)} stored templates...")
        for stored_user_id, stored_fp_id, stored_template_data in all_stored_data:
            stored_minutiae = []
            if 'features' in stored_template_data and 'minutiae' in stored_template_data['features']:
                 stored_minutiae = stored_template_data['features']['minutiae']
                 if not stored_minutiae: continue

                 # === CALLING SHARED MATCHING LOGIC ===
                 raw_count, norm_score = match_minutiae(input_minutiae, stored_minutiae)
                 # === END CALL ===

                 if raw_count > best_overall_score:
                    best_overall_score = raw_count
                    best_match_user_id = stored_user_id
                    best_match_fp_id = stored_fp_id

        db_match_end = time.time()
        print(f"Comparison finished. (Time: {db_match_end - db_match_start:.2f}s)")

        # --- Decision ---
        print("\n--- Identification Result ---")
        if best_match_user_id is None: best_overall_score = 0

        print(f"Best Match Score Found: {best_overall_score} (Raw Count)")
        print(f"Best Matching Template: User '{best_match_user_id}', Fingerprint DB ID '{best_match_fp_id}'")
        print(f"Identification Threshold (Raw Count): {IDENTIFICATION_THRESHOLD_RAW_COUNT}")

        total_time = time.time() - start_time
        identified_user = None
        if best_overall_score >= IDENTIFICATION_THRESHOLD_RAW_COUNT:
            print(f"Decision: IDENTIFIED as User '{best_match_user_id}'")
            identified_user = best_match_user_id
            print(f"{'='*15} Identification Successful {'='*15}")
        else:
            print(f"Decision: NOT IDENTIFIED (Score below threshold)")
            print(f"{'='*15} Identification Failed {'='*15}")

        print(f"Total time: {total_time:.2f}s")
        return identified_user, best_overall_score

    except Exception as e:
        print(f"[Error] Exception during DB retrieval/matching:"); traceback.print_exc(); return None, 0.0

# --- Command-Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser( # ... rest of parser setup ...
        description="Identify a fingerprint...", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("image_path", help="Path to fingerprint image.")
    parser.add_argument("--display", action="store_true", help="Display intermediate images.")
    parser.add_argument("-t", "--threshold", type=int, default=IDENTIFICATION_THRESHOLD_RAW_COUNT, help=f"Identification threshold. Default={IDENTIFICATION_THRESHOLD_RAW_COUNT}")
    args = parser.parse_args()
    IDENTIFICATION_THRESHOLD_RAW_COUNT = args.threshold
    try: initialize_database() # Ensures DB exists and table is structured
    except Exception as e: print(f"DB Init Error: {e}"); sys.exit(1)
    # Check if DB has entries before proceeding? Optional.
    identified_user_id, best_score = identify_fingerprint(args.image_path, args.display)
    sys.exit(0 if identified_user_id is not None else 1)