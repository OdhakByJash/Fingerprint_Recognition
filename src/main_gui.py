# src/main_gui.py

import tkinter as tk
from tkinter import ttk # Themed widgets
from tkinter import filedialog, messagebox
import os
import sys
import threading # To prevent GUI freezing during long operations
import time 
import traceback # For detailed error logging

# --- Add src directory to Python path ---
try:
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path: sys.path.append(src_dir)
except Exception as e: print(f"Error modifying sys.path: {e}"); sys.exit(1)

# --- Import Core Backend Functions ---
# Import the necessary functions directly from the modules
# This avoids modifying the original CLI scripts (enroll.py, etc.)
try:
    from preprocessing import preprocess_fingerprint, ORIENTATION_BLOCK_SIZE
    from feature_extraction import extract_minutiae, create_template # Only need these two for enroll
    from database_operations import initialize_database, store_template, get_templates_for_user, get_all_templates, DB_PATH
    from matching import match_minutiae
    # We don't need to import enroll.py, verify.py, identify.py itself
except ImportError as e:
    messagebox.showerror("Import Error", f"Failed to import backend module: {e}\n\nPlease ensure all required .py files are in the 'src' directory.")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    messagebox.showerror("Import Error", f"An unexpected error occurred during imports: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Constants ---
WINDOW_WIDTH = 650
WINDOW_HEIGHT = 550
STATUS_AREA_HEIGHT = 15 # In text lines

# --- GUI Application Class ---

class FingerprintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fingerprint Authentication System")
        # Set window size
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(WINDOW_WIDTH, WINDOW_HEIGHT) # Prevent excessive shrinking

        # --- Style ---
        self.style = ttk.Style()
        # Available themes depend on OS: 'clam', 'alt', 'default', 'classic', 'vista', 'xpnative'
        try:
            self.style.theme_use('vista') # Try a potentially nicer theme
        except tk.TclError:
            print("Info: 'vista' theme not available, using default.")
            self.style.theme_use('default')

        # Style configurations
        self.style.configure('TLabel', padding=5, font=('Segoe UI', 10))
        self.style.configure('TButton', padding=6, font=('Segoe UI', 10, 'bold'))
        self.style.configure('TEntry', padding=5, font=('Segoe UI', 10))
        self.style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'), padding=(10, 15))
        self.style.configure('Status.TLabel', font=('Segoe UI', 10, 'italic'), padding=(5, 2))

        # --- Variables ---
        self.user_id_var = tk.StringVar()
        self.file_path_var = tk.StringVar()
        self.file_path_var.set("No file selected")

        # --- Initialize Database ---
        try:
            initialize_database()
            print("Database initialized successfully.")
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to initialize database at {DB_PATH}:\n{e}")
            self.root.quit() # Exit if DB can't be initialized

        # --- Build GUI ---
        self.create_widgets()

    def create_widgets(self):
        # --- Main Frame ---
        main_frame = ttk.Frame(self.root, padding="15 15 15 15")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # --- Header ---
        header_label = ttk.Label(main_frame, text="Fingerprint System", style='Header.TLabel')
        header_label.pack(pady=(0, 15))

        # --- Input Frame ---
        input_frame = ttk.Frame(main_frame, padding="10 10 10 10", relief=tk.GROOVE)
        input_frame.pack(fill=tk.X, pady=10)
        input_frame.columnconfigure(1, weight=1) # Make entry/label column expand

        # User ID Input
        user_id_label = ttk.Label(input_frame, text="User ID (e.g., 101):")
        user_id_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.entry_user_id = ttk.Entry(input_frame, textvariable=self.user_id_var, width=40)
        self.entry_user_id.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        # File Selection
        file_label = ttk.Label(input_frame, text="Fingerprint File:")
        file_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.file_path_display = ttk.Label(input_frame, textvariable=self.file_path_var, relief=tk.SUNKEN, width=40, anchor=tk.W)
        self.file_path_display.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        browse_button = ttk.Button(input_frame, text="Browse...", command=self.browse_file)
        browse_button.grid(row=1, column=2, padx=5, pady=5)

        # --- Action Buttons Frame ---
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(pady=15)

        enroll_button = ttk.Button(action_frame, text="Enroll Fingerprint", command=self.run_enrollment_thread)
        enroll_button.grid(row=0, column=0, padx=10, pady=5)

        verify_button = ttk.Button(action_frame, text="Verify Fingerprint (1:1)", command=self.run_verification_thread)
        verify_button.grid(row=0, column=1, padx=10, pady=5)

        identify_button = ttk.Button(action_frame, text="Identify Fingerprint (1:N)", command=self.run_identification_thread)
        identify_button.grid(row=0, column=2, padx=10, pady=5)

        # --- Status Area ---
        status_frame = ttk.LabelFrame(main_frame, text="Status / Output", padding="10 10 10 10")
        status_frame.pack(expand=True, fill=tk.BOTH, pady=10)

        self.status_text = tk.Text(status_frame, height=STATUS_AREA_HEIGHT, wrap=tk.WORD, relief=tk.FLAT, font=('Consolas', 9), state=tk.DISABLED) # Start disabled
        scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.config(yscrollcommand=scrollbar.set)

        self.status_text.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def browse_file(self):
        # Define file types (adjust as needed)
        filetypes = (
            ('Image Files', '*.tif *.tiff *.bmp *.png *.jpg *.jpeg'),
            ('All Files', '*.*')
        )
        # Get initial directory (e.g., one level up from src, in 'data')
        initial_dir = os.path.abspath(os.path.join(src_dir, "../data"))
        if not os.path.isdir(initial_dir): initial_dir = os.path.expanduser("~") # Fallback to home

        filepath = filedialog.askopenfilename(
            title='Select Fingerprint Image',
            initialdir=initial_dir,
            filetypes=filetypes
        )
        if filepath:
            self.file_path_var.set(filepath)
            self.update_status(f"Selected file: {filepath}")
        else:
            self.file_path_var.set("No file selected")

    def update_status(self, message, clear_first=False):
        # Must enable text widget to modify, then disable again
        self.status_text.config(state=tk.NORMAL)
        if clear_first:
            self.status_text.delete('1.0', tk.END) # Clear previous content
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END) # Scroll to the end
        self.status_text.config(state=tk.DISABLED)
        self.root.update_idletasks() # Force GUI update

    def format_user_id(self, raw_id):
        """Prefixes the raw ID with 'subject_'."""
        return f"subject_{raw_id}"

    def run_in_thread(self, target_func, *args):
        """Runs a function in a separate thread to avoid freezing the GUI."""
        thread = threading.Thread(target=target_func, args=args, daemon=True)
        thread.start()

    def run_enrollment_thread(self):
        self.run_in_thread(self._execute_enrollment)

    def run_verification_thread(self):
        self.run_in_thread(self._execute_verification)

    def run_identification_thread(self):
        self.run_in_thread(self._execute_identification)

    # --- Backend Execution Wrappers (Called by Buttons via Threads) ---

    def _execute_enrollment(self):
        self.update_status("--- Starting Enrollment ---", clear_first=True)
        raw_id = self.user_id_var.get().strip()
        filepath = self.file_path_var.get()

        # --- Input Validation ---
        if not raw_id:
            messagebox.showerror("Input Error", "User ID cannot be empty for enrollment.")
            self.update_status("ERROR: User ID required.")
            return
        if not os.path.exists(filepath) or filepath == "No file selected":
            messagebox.showerror("Input Error", "Please select a valid fingerprint image file.")
            self.update_status("ERROR: Fingerprint file required.")
            return
        # --- End Validation ---

        # Format user ID
        user_id = self.format_user_id(raw_id)
        self.update_status(f"Processing enrollment for User ID: {user_id}")
        self.update_status(f"Image: {filepath}")

        try:
            start_time = time.time()
            # 1. Preprocess
            self.update_status("Step 1: Preprocessing...")
            preprocess_result = preprocess_fingerprint(filepath) # Don't display/save from GUI call
            if preprocess_result is None: raise ValueError("Preprocessing failed.")
            thinned_img = preprocess_result.get('thinned_image')
            mask = preprocess_result.get('mask')
            orientation_map = preprocess_result.get('orientation_map')
            if thinned_img is None or mask is None or orientation_map is None:
                raise ValueError("Preprocessing missing required data.")
            h, w = thinned_img.shape
            self.update_status("Preprocessing successful.")

            # 2. Extract Features
            self.update_status("Step 2: Extracting features...")
            minutiae = extract_minutiae(thinned_img, orientation_map, mask, ORIENTATION_BLOCK_SIZE)
            if minutiae is None: # Should return empty list, but check None just in case
                 raise ValueError("Feature extraction failed.")
            self.update_status(f"Extracted {len(minutiae)} filtered minutiae points.")
            if not minutiae:
                 self.update_status("WARNING: No minutiae extracted after filtering.")

            # 3. Create Template
            self.update_status("Step 3: Creating template...")
            template_json = create_template(minutiae, image_width=w, image_height=h)
            if template_json is None: raise ValueError("Template creation failed.")
            self.update_status("Template created.")

            # 4. Store Template
            self.update_status("Step 4: Storing template...")
            fingerprint_id = store_template(user_id, template_json)
            if fingerprint_id is None: raise ValueError("Database storage failed.")

            end_time = time.time()
            self.update_status(f"\n--- Enrollment Successful ---")
            self.update_status(f"Stored template for User '{user_id}'. Record ID: {fingerprint_id}")
            self.update_status(f"Total time: {end_time - start_time:.2f}s")

        except Exception as e:
            self.update_status(f"\n--- Enrollment FAILED ---")
            self.update_status(f"Error: {e}")
            # Optionally log full traceback to console or file
            print("\n--- ERROR DURING ENROLLMENT ---")
            traceback.print_exc()
            print("-----------------------------\n")
            messagebox.showerror("Enrollment Error", f"An error occurred during enrollment:\n{e}")

    def _execute_verification(self):
        self.update_status("--- Starting Verification (1:1) ---", clear_first=True)
        raw_id = self.user_id_var.get().strip()
        filepath = self.file_path_var.get()

        # --- Input Validation ---
        if not raw_id:
            messagebox.showerror("Input Error", "User ID cannot be empty for verification.")
            self.update_status("ERROR: User ID required.")
            return
        if not os.path.exists(filepath) or filepath == "No file selected":
            messagebox.showerror("Input Error", "Please select a valid fingerprint image file.")
            self.update_status("ERROR: Fingerprint file required.")
            return
        # --- End Validation ---

        user_id = self.format_user_id(raw_id)
        self.update_status(f"Processing verification for User ID: {user_id}")
        self.update_status(f"Image: {filepath}")

        try:
            start_time = time.time()
            # 1. Preprocess
            self.update_status("Step 1: Preprocessing...")
            preprocess_result = preprocess_fingerprint(filepath)
            if preprocess_result is None: raise ValueError("Preprocessing failed.")
            thinned_img = preprocess_result.get('thinned_image')
            mask = preprocess_result.get('mask')
            orientation_map = preprocess_result.get('orientation_map')
            if thinned_img is None or mask is None or orientation_map is None:
                raise ValueError("Preprocessing missing required data.")
            self.update_status("Preprocessing successful.")

            # 2. Extract Features
            self.update_status("Step 2: Extracting features...")
            input_minutiae = extract_minutiae(thinned_img, orientation_map, mask, ORIENTATION_BLOCK_SIZE)
            if input_minutiae is None: raise ValueError("Feature extraction failed.")
            self.update_status(f"Extracted {len(input_minutiae)} filtered minutiae points.")
            if not input_minutiae:
                 self.update_status("WARNING: No minutiae extracted. Cannot verify.")
                 # Force failure if no features extracted
                 raise ValueError("No features extracted from input image.")

            # 3. Retrieve & Match
            self.update_status("Step 3: Retrieving template(s) and matching...")
            stored_templates = get_templates_for_user(user_id)
            if stored_templates is None: raise ValueError("Database error retrieving templates.")
            if not stored_templates: raise ValueError(f"User ID '{user_id}' not found in database.")

            best_match_score = -1
            best_raw_count = -1
            matched_template_index = -1

            self.update_status(f"Comparing against {len(stored_templates)} stored template(s)...")
            for idx, template in enumerate(stored_templates):
                stored_minutiae = []
                if 'features' in template and 'minutiae' in template['features']:
                    stored_minutiae = template['features']['minutiae']
                    if not stored_minutiae: continue # Skip empty templates
                    raw_count, norm_score = match_minutiae(input_minutiae, stored_minutiae)
                    self.update_status(f"  Compared with template {idx+1}: Raw Score = {raw_count}")
                    if raw_count > best_raw_count:
                        best_raw_count = raw_count
                        best_match_score = norm_score # Store normalized score if needed
                        matched_template_index = idx + 1
                # else: self.update_status(f"  Skipping template {idx+1} (invalid format)")

            # 4. Decision (using default threshold from matching, could make it configurable)
            # Re-import VERIFICATION_THRESHOLD_RAW_COUNT if it's defined elsewhere
            # Or define it here if it's static for the GUI
            try:
                from verify import VERIFICATION_THRESHOLD_RAW_COUNT as VERIFY_THR # Import from verify.py
            except ImportError:
                 VERIFY_THR = 8 # Fallback if verify.py isn't structured to export it
                 self.update_status(f"Warning: Couldn't import threshold from verify.py, using default {VERIFY_THR}")

            self.update_status(f"\nBest Raw Match Score: {best_raw_count}")
            self.update_status(f"Verification Threshold: {VERIFY_THR}")

            end_time = time.time()
            if best_raw_count >= VERIFY_THR:
                self.update_status(f"\n--- Verification Successful ---")
                self.update_status(f"Decision: VERIFIED")
            else:
                self.update_status(f"\n--- Verification Failed ---")
                self.update_status(f"Decision: NOT VERIFIED (Score below threshold)")
            self.update_status(f"Total time: {end_time - start_time:.2f}s")

        except Exception as e:
            self.update_status(f"\n--- Verification FAILED ---")
            self.update_status(f"Error: {e}")
            print("\n--- ERROR DURING VERIFICATION ---")
            traceback.print_exc()
            print("-----------------------------\n")
            messagebox.showerror("Verification Error", f"An error occurred during verification:\n{e}")

    def _execute_identification(self):
        self.update_status("--- Starting Identification (1:N) ---", clear_first=True)
        # User ID is not needed for identification input
        filepath = self.file_path_var.get()

        # --- Input Validation ---
        if not os.path.exists(filepath) or filepath == "No file selected":
            messagebox.showerror("Input Error", "Please select a valid fingerprint image file.")
            self.update_status("ERROR: Fingerprint file required.")
            return
        # --- End Validation ---

        self.update_status(f"Processing identification for image:")
        self.update_status(f"{filepath}")

        try:
            start_time = time.time()
            # 1. Preprocess
            self.update_status("Step 1: Preprocessing...")
            preprocess_result = preprocess_fingerprint(filepath)
            if preprocess_result is None: raise ValueError("Preprocessing failed.")
            thinned_img = preprocess_result.get('thinned_image')
            mask = preprocess_result.get('mask')
            orientation_map = preprocess_result.get('orientation_map')
            if thinned_img is None or mask is None or orientation_map is None:
                 raise ValueError("Preprocessing missing required data.")
            self.update_status("Preprocessing successful.")

            # 2. Extract Features
            self.update_status("Step 2: Extracting features...")
            input_minutiae = extract_minutiae(thinned_img, orientation_map, mask, ORIENTATION_BLOCK_SIZE)
            if input_minutiae is None: raise ValueError("Feature extraction failed.")
            self.update_status(f"Extracted {len(input_minutiae)} filtered minutiae points.")
            if not input_minutiae:
                 raise ValueError("No features extracted from input image.")

            # 3. Retrieve ALL Templates & Match
            self.update_status("Step 3: Retrieving all templates and matching...")
            all_stored_data = get_all_templates()
            if all_stored_data is None: raise ValueError("Database error retrieving templates.")
            if not all_stored_data: raise ValueError("No templates found in database.")

            best_overall_score = -1
            best_match_user_id = None
            best_match_fp_id = None

            self.update_status(f"Comparing against {len(all_stored_data)} stored templates...")
            for stored_user_id, stored_fp_id, stored_template_data in all_stored_data:
                stored_minutiae = []
                if 'features' in stored_template_data and 'minutiae' in stored_template_data['features']:
                    stored_minutiae = stored_template_data['features']['minutiae']
                    if not stored_minutiae: continue
                    raw_count, norm_score = match_minutiae(input_minutiae, stored_minutiae)
                    if raw_count > best_overall_score:
                        best_overall_score = raw_count
                        best_match_user_id = stored_user_id
                        best_match_fp_id = stored_fp_id
                # else: self.update_status(f" Skipping template fp_id {stored_fp_id} (invalid format)")

            # 4. Decision
            try:
                 from identify import IDENTIFICATION_THRESHOLD_RAW_COUNT as IDENT_THR # Import from identify.py
            except ImportError:
                 IDENT_THR = 8 # Fallback if identify.py isn't structured to export it
                 self.update_status(f"Warning: Couldn't import threshold from identify.py, using default {IDENT_THR}")

            self.update_status(f"\nBest Match Score Found: {best_overall_score} (Raw Count)")
            self.update_status(f"Best Matching Template: User '{best_match_user_id}', FP_ID {best_match_fp_id}")
            self.update_status(f"Identification Threshold: {IDENT_THR}")

            end_time = time.time()
            if best_overall_score >= IDENT_THR:
                self.update_status(f"\n--- Identification Successful ---")
                self.update_status(f"Decision: IDENTIFIED as User '{best_match_user_id}'")
            else:
                self.update_status(f"\n--- Identification Failed ---")
                self.update_status(f"Decision: NOT IDENTIFIED (Score below threshold)")
            self.update_status(f"Total time: {end_time - start_time:.2f}s")

        except Exception as e:
            self.update_status(f"\n--- Identification FAILED ---")
            self.update_status(f"Error: {e}")
            print("\n--- ERROR DURING IDENTIFICATION ---")
            traceback.print_exc()
            print("-------------------------------\n")
            messagebox.showerror("Identification Error", f"An error occurred during identification:\n{e}")


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintApp(root)
    root.mainloop()