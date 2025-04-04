# src/verify.py

import argparse
import os
import sys
import time
import traceback
import json
 
# ... (Keep sys.path modification and imports) ...
try:
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path: sys.path.append(src_dir)
except Exception as e: print(f"Error modifying sys.path: {e}"); sys.exit(1)

try:
    from preprocessing import preprocess_fingerprint, ORIENTATION_BLOCK_SIZE
    from feature_extraction import extract_minutiae, visualize_minutiae
    from database_operations import get_templates_for_user, DB_PATH, initialize_database
    from matching import match_minutiae # Use the version handling None angles
    from utils import plot_image
except ImportError as e: print(f"Error importing project module: {e}"); traceback.print_exc(); sys.exit(1)
except Exception as e: print(f"An unexpected error during imports: {e}"); traceback.print_exc(); sys.exit(1)

try: import matplotlib.pyplot as plt
except ImportError: plt = None

# --- Verification Configuration ---
# Adjust this threshold based on tests with *filtered* minutiae scores
VERIFICATION_THRESHOLD_RAW_COUNT = 8 # Example: Lower threshold might be suitable now

# --- Main Verification Function ---
def verify_fingerprint(user_id, image_path, display_steps=False):
    start_time = time.time()
    print(f"\n{'='*15} Starting Verification Process {'='*15}")
    # ... (print paths) ...
    abs_image_path = os.path.abspath(image_path); print(f"Verification Image: {abs_image_path}")
    abs_db_path = os.path.abspath(DB_PATH); print(f"Database:     {abs_db_path}")

    if not os.path.exists(image_path): print(f"\n[Error] Image file not found: {image_path}"); return False, 0.0

    # --- Step 1: Preprocessing ---
    print("\n[Step 1/3] Preprocessing verification image...")
    preprocess_start = time.time()
    try:
        results_dir = os.path.abspath(os.path.join(src_dir, "../results"))
        os.makedirs(results_dir, exist_ok=True)
        safe_user_id = "".join(c if c.isalnum() else "_" for c in user_id)
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        preprocess_output_dir = os.path.join(results_dir, f"verify_preprocess_{safe_user_id}_{image_basename}")

        # --- MODIFIED PREPROCESSING CALL ---
        preprocess_result = preprocess_fingerprint(image_path, display_steps=display_steps, output_dir=preprocess_output_dir)
        # --- END MODIFICATION ---

        preprocess_end = time.time()

        # --- MODIFIED RESULT HANDLING ---
        if preprocess_result is None:
            print("[Error] Preprocessing failed for verification image.")
            print(f"{'='*15} Verification Failed {'='*15}"); return False, 0.0
        thinned_img = preprocess_result.get('thinned_image')
        mask = preprocess_result.get('mask')
        orientation_map = preprocess_result.get('orientation_map')
        if thinned_img is None or mask is None or orientation_map is None:
             print("[Error] Preprocessing did not return all required data."); return False, 0.0
        # --- END MODIFICATION ---

        print(f"Preprocessing complete. (Time: {preprocess_end - preprocess_start:.2f}s)")
        if display_steps and plt: plt.show()

    except Exception as e:
        print(f"[Error] An exception occurred during preprocessing:"); traceback.print_exc(); return False, 0.0

    # --- Step 2: Feature Extraction ---
    print("\n[Step 2/3] Extracting and filtering features from verification image...")
    feature_start = time.time()
    input_minutiae = []
    try:
        # --- MODIFIED FEATURE EXTRACTION CALL ---
        input_minutiae = extract_minutiae(thinned_img, orientation_map, mask, ORIENTATION_BLOCK_SIZE)
        # --- END MODIFICATION ---
        feature_end = time.time()

        if not input_minutiae:
            print("[Warning] No valid minutiae extracted from verification image after filtering.")
            print(f"{'='*15} Verification Failed {'='*15}"); return False, 0.0
        print(f"Extracted and filtered {len(input_minutiae)} minutiae points. (Time: {feature_end - feature_start:.2f}s)")

        if display_steps and input_minutiae and plt:
             visualize_minutiae(thinned_img, input_minutiae, title_suffix=" (Verification - Filtered)")
             plt.show()

    except Exception as e:
        print(f"[Error] An exception occurred during feature extraction:"); traceback.print_exc(); return False, 0.0

    # --- Step 3: Retrieve Stored Template(s) and Match ---
    print("\n[Step 3/3] Retrieving stored template(s) and matching...")
    db_match_start = time.time()
    try:
        stored_templates = get_templates_for_user(user_id)

        if stored_templates is None: print("[Error] Database error."); return False, 0.0
        if not stored_templates: print(f"[Error] User ID '{user_id}' not found."); return False, 0.0

        best_match_score = -1 # Raw count score
        matched_template_index = -1

        print(f"Comparing against {len(stored_templates)} stored template(s) for user '{user_id}'...")
        for idx, template in enumerate(stored_templates):
            # ... (Extract stored_minutiae as before) ...
            stored_minutiae = []
            if 'features' in template and 'minutiae' in template['features']:
                stored_minutiae = template['features']['minutiae']
                if not stored_minutiae: print(f"  Skipping template {idx+1} (no minutiae)."); continue

                print(f"\n  --- Matching Input vs Stored Template {idx+1} ---")
                print(f"  Input Count (Filtered): {len(input_minutiae)}")
                print(f"  Stored Count (Filtered): {len(stored_minutiae)}")

                # REMOVED the explicit check for count difference causing failure
                # Counts are *expected* to differ slightly now

                print(f"\n  Running match_minutiae...")
                raw_count, norm_score = match_minutiae(input_minutiae, stored_minutiae)
                print(f"    Match Result - Raw Count: {raw_count}, Normalized Score: {norm_score:.4f}")
                print(f"  --- End Match {idx+1} ---\n")

                if raw_count > best_match_score:
                    best_match_score = raw_count
                    matched_template_index = idx + 1
            else: print(f"  Skipping template {idx+1} (invalid format).")

        db_match_end = time.time()
        print(f"Matching process finished. (Time: {db_match_end - db_match_start:.2f}s)")

        # --- Decision ---
        print("\n--- Verification Result ---")
        if matched_template_index == -1: best_match_score = 0 # Ensure score is 0 if no valid matching occurred

        print(f"Best Raw Match Count Found: {best_match_score}" + (f" (against stored template {matched_template_index})" if matched_template_index != -1 else ""))
        print(f"Verification Threshold (Raw Count): {VERIFICATION_THRESHOLD_RAW_COUNT}")

        final_result = best_match_score >= VERIFICATION_THRESHOLD_RAW_COUNT

        total_time = time.time() - start_time
        if final_result:
            print(f"Decision: VERIFIED")
            print(f"{'='*15} Verification Successful {'='*15}")
        else:
            print(f"Decision: NOT VERIFIED")
            if matched_template_index != -1: print("  Reason: Best match score below threshold.")
            else: print("  Reason: No valid templates compared or matched.")
            print(f"{'='*15} Verification Failed {'='*15}")
        print(f"Total time: {total_time:.2f}s")
        return final_result, best_match_score

    except Exception as e:
        print(f"[Error] An exception occurred during DB retrieval or matching:"); traceback.print_exc(); return False, 0.0

# --- Command-Line Interface ---
# ... (if __name__ == "__main__": block remains the same) ...
if __name__ == "__main__":
    parser = argparse.ArgumentParser( # ... rest of parser setup ...
         description="Verify a fingerprint...", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("user_id", help="Claimed user identifier.")
    parser.add_argument("image_path", help="Path to fingerprint image for verification.")
    parser.add_argument("--display", action="store_true", help="Display intermediate images.")
    parser.add_argument("-t", "--threshold", type=int, default=VERIFICATION_THRESHOLD_RAW_COUNT, help=f"Verification threshold (raw count). Default={VERIFICATION_THRESHOLD_RAW_COUNT}")
    args = parser.parse_args()
    VERIFICATION_THRESHOLD_RAW_COUNT = args.threshold
    try: initialize_database()
    except Exception as e: print(f"DB Init Error: {e}"); sys.exit(1)
    verified, score = verify_fingerprint(args.user_id, args.image_path, args.display)
    sys.exit(0 if verified else 1)