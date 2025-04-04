# src/enroll.py

import argparse
import os
import sys
import time
import traceback

# ... (Keep sys.path modification and imports - ensure they work) ...
try:
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path: sys.path.append(src_dir)
    print(f"\nDEBUG: Current sys.path:\n{sys.path}\n") # Print path for verification
except Exception as e: print(f"Error modifying sys.path: {e}"); sys.exit(1)

try:
    from preprocessing import preprocess_fingerprint, ORIENTATION_BLOCK_SIZE # Import block size
    from feature_extraction import extract_minutiae, create_template, visualize_minutiae
    from database_operations import initialize_database, store_template, check_user_exists, DB_PATH
    from utils import plot_image
except ImportError as e: print(f"Error importing project module: {e}"); traceback.print_exc(); sys.exit(1)
except Exception as e: print(f"An unexpected error during imports: {e}"); traceback.print_exc(); sys.exit(1)

try: import matplotlib.pyplot as plt
except ImportError: plt = None

# ... (Keep enroll_fingerprint function signature) ...
def enroll_fingerprint(user_id, image_path, display_steps=False, output_vis=False):
    start_time = time.time()
    print(f"\n{'='*15} Starting Enrollment Process {'='*15}")
    # ... (print paths) ...
    abs_image_path = os.path.abspath(image_path); print(f"Image Path:   {abs_image_path}")
    abs_db_path = os.path.abspath(DB_PATH); print(f"Database:     {abs_db_path}")

    if not os.path.exists(image_path): print(f"\n[Error] Image file not found: {image_path}"); return False

    # --- Step 1: Preprocessing ---
    print("\n[Step 1/4] Preprocessing fingerprint image...")
    preprocess_start = time.time()
    try:
        results_dir = os.path.abspath(os.path.join(src_dir, "../results"))
        os.makedirs(results_dir, exist_ok=True)
        safe_user_id = "".join(c if c.isalnum() else "_" for c in user_id)
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        preprocess_output_dir = os.path.join(results_dir, f"enroll_preprocess_{safe_user_id}_{image_basename}")

        # --- MODIFIED PREPROCESSING CALL ---
        preprocess_result = preprocess_fingerprint(image_path, display_steps=display_steps, output_dir=preprocess_output_dir)
        # --- END MODIFICATION ---

        preprocess_end = time.time()

        # --- MODIFIED RESULT HANDLING ---
        if preprocess_result is None:
            print("[Error] Preprocessing failed. Check logs or image quality.")
            print(f"{'='*15} Enrollment Failed {'='*15}")
            return False
        # Extract results from dictionary
        thinned_img = preprocess_result.get('thinned_image')
        mask = preprocess_result.get('mask')
        orientation_map = preprocess_result.get('orientation_map')
        if thinned_img is None or mask is None or orientation_map is None:
             print("[Error] Preprocessing did not return all required data (thinned_image, mask, orientation_map).")
             print(f"{'='*15} Enrollment Failed {'='*15}")
             return False
        # --- END MODIFICATION ---

        print(f"Preprocessing complete. (Time: {preprocess_end - preprocess_start:.2f}s)")
        if display_steps and plt: plt.show()
        h, w = thinned_img.shape

    except Exception as e:
        print(f"[Error] An exception occurred during preprocessing:")
        traceback.print_exc(); print(f"{'='*15} Enrollment Failed {'='*15}"); return False

    # --- Step 2: Feature Extraction (Minutiae) ---
    print("\n[Step 2/4] Extracting and filtering minutiae features...")
    feature_start = time.time()
    try:
        # --- MODIFIED FEATURE EXTRACTION CALL ---
        # Pass the required arguments extracted from preprocessing result
        minutiae = extract_minutiae(thinned_img, orientation_map, mask, ORIENTATION_BLOCK_SIZE)
        # --- END MODIFICATION ---
        feature_end = time.time()

        # Check the number of minutiae AFTER filtering
        if not minutiae:
            print("[Warning] No valid minutiae found after filtering. Enrollment template will be empty.")
        else:
             print(f"Extracted and filtered {len(minutiae)} minutiae points. (Time: {feature_end - feature_start:.2f}s)")

        # Optional: Visualize filtered minutiae
        if output_vis and minutiae:
            vis_filename = f"{safe_user_id}_{image_basename}_minutiae_enroll.png"
            vis_path = os.path.join(results_dir, vis_filename)
            visualize_minutiae(thinned_img, minutiae, title_suffix=" (Enrollment - Filtered)", save_path=vis_path)
        elif display_steps and minutiae and plt:
             visualize_minutiae(thinned_img, minutiae, title_suffix=" (Enrollment - Filtered)")
             plt.show()

    except Exception as e:
        print(f"[Error] An exception occurred during feature extraction:")
        traceback.print_exc(); print(f"{'='*15} Enrollment Failed {'='*15}"); return False

    # --- Step 3: Create Template ---
    print("\n[Step 3/4] Creating fingerprint template...")
    # ... (template creation call remains the same) ...
    template_json = create_template(minutiae, image_width=w, image_height=h)
    if template_json is None: print("[Error] Failed to create template JSON."); return False
    print("Template created.")


    # --- Step 4: Store Template in Database ---
    print("\n[Step 4/4] Storing template in database...")
    # ... (database storage call remains the same) ...
    fingerprint_id = store_template(user_id, template_json)
    if fingerprint_id is not None:
        total_time = time.time() - start_time
        print(f"Template stored successfully for User '{user_id}'. Record ID: {fingerprint_id}")
        print(f"\n{'='*15} Enrollment Successful {'='*15}")
        print(f"Total time: {total_time:.2f}s")
        return True
    else:
        print("[Error] Failed to store template in the database.")
        print(f"{'='*15} Enrollment Failed {'='*15}")
        return False

# --- Command-Line Interface ---
# ... (if __name__ == "__main__": block remains the same) ...
if __name__ == "__main__":
    parser = argparse.ArgumentParser( # ... rest of parser setup ...
        description="Enroll a fingerprint...", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("user_id", help="User identifier.")
    parser.add_argument("image_path", help="Path to fingerprint image.")
    parser.add_argument("--display", action="store_true", help="Display intermediate images.")
    parser.add_argument("--savevis", action="store_true", help="Save minutiae visualization.")
    args = parser.parse_args()
    try: initialize_database()
    except Exception as e: print(f"DB Init Error: {e}"); sys.exit(1)
    success = enroll_fingerprint(args.user_id, args.image_path, args.display, args.savevis)
    sys.exit(0 if success else 1)