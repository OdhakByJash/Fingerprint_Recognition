# Fingerprint Recognition System

This project implements a modular fingerprint recognition system for biometric authentication. It supports three main operations: **Enrollment**, **Verification (1:1 matching)**, and **Identification (1:N matching)**. The system is designed for extensibility, robust feature extraction, and ease of use via both command-line and graphical interfaces.

## Features

- **Enrollment**: Preprocesses a fingerprint image, extracts minutiae features, and stores a template in a SQLite database.
- **Verification**: Compares a fingerprint image against stored templates for a claimed user ID.
- **Identification**: Searches a fingerprint image against all enrolled templates to identify the user.
- **GUI**: A user-friendly Tkinter-based graphical interface for all operations.
- **Algorithm Comparison**: Scripts to compare thinning and feature extraction algorithms (e.g., Zhang-Suen vs. OpenCV thinning, RANSAC-based feature extraction).
- **Robust Preprocessing**: Includes normalization, segmentation, orientation and frequency estimation, Gabor enhancement, binarization, and thinning.
- **Minutiae Extraction**: Uses crossing number method, angle calculation, and post-processing filters for reliable feature extraction.
- **Database Management**: Uses SQLite for template storage, retrieval, and indexing.
- **Visualization**: Matplotlib-based visualization of images and extracted features.

## Technology Stack

- **Programming Language**: Python 3
- **Image Processing**: OpenCV (`cv2`), NumPy
- **Machine Learning (optional)**: scikit-learn (for RANSAC-based feature extraction)
- **Database**: SQLite (via Python's `sqlite3`)
- **GUI**: Tkinter (Python standard library)
- **Plotting/Visualization**: Matplotlib

## Workflow Overview

### 1. Preprocessing

- **Normalization**: Standardizes image intensity.
- **Segmentation**: Isolates fingerprint region using block-wise variance and morphological operations.
- **Orientation Field Estimation**: Computes local ridge orientation using Sobel gradients and smoothing.
- **Ridge Frequency Estimation**: Estimates local ridge spacing via FFT analysis.
- **Gabor Enhancement**: Applies orientation- and frequency-adaptive Gabor filtering for ridge clarity.
- **Binarization**: Converts enhanced image to binary using adaptive thresholding.
- **Thinning**: Skeletonizes ridges using OpenCV's Zhang-Suen thinning (or morphological fallback).

### 2. Feature Extraction

- **Minutiae Detection**: Uses crossing number method to detect ridge endings and bifurcations.
- **Angle Calculation**: Assigns orientation to each minutia based on local orientation field.
- **Filtering**: Removes spurious minutiae near mask borders and close pairs.
- **Template Creation**: Serializes minutiae and metadata into JSON for database storage.

### 3. Matching

- **Minutiae Matching**: Compares sets of minutiae using distance and angle thresholds, with one-to-one greedy matching.
- **Verification**: Compares input features to stored templates for a claimed user.
- **Identification**: Compares input features to all templates to find the best match.

### 4. Database Operations

- **Initialization**: Creates database and tables if not present.
- **Storage**: Inserts new templates with user ID and timestamp.
- **Retrieval**: Fetches templates by user or all templates for identification.
- **Indexing**: Indexes user IDs for efficient lookup.

### 5. GUI

- **User Input**: Allows selection of user ID and fingerprint image.
- **Operation Selection**: Buttons for enrollment, verification, and identification.
- **Status Display**: Shows progress, results, and errors.
- **Threading**: Runs backend operations in separate threads to keep UI responsive.

### 6. Algorithm Comparison

- **Thinning Algorithms**: Compares Zhang-Suen and OpenCV thinning for performance and connectivity.
- **Feature Extraction**: Compares standard crossing number method with RANSAC-based ridge curve fitting.



## File Overview

- `src/preprocessing.py`: Preprocessing pipeline (normalization, segmentation, enhancement, thinning).
- `src/feature_extraction.py`: Minutiae extraction, filtering, visualization, template creation.
- `src/matching.py`: Minutiae matching logic.
- `src/database_operations.py`: SQLite database management.
- `src/enroll.py`, `src/verify.py`, `src/identify.py`: CLI scripts for enrollment, verification, identification.
- `src/main_gui.py`: Tkinter GUI application.
- `src/utils.py`: Utility functions (plotting, array I/O).
- `src/zhang_suen_thinning (1) - Copy.py`: Zhang-Suen thinning implementation and comparison.
- `src/ransac_feature_extraction (1) - Copy.py`: RANSAC-based ridge curve and minutiae extraction.
- `src/compare_algorithms (1) - Copy.py`: Algorithm comparison script.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`, `opencv-contrib-python`)
- NumPy
- Matplotlib
- scikit-learn (for RANSAC comparison)
- Tkinter (standard with Python)
- SQLite (standard with Python)

Install dependencies:
## Notes

- Fingerprint images should be grayscale and of reasonable quality/resolution.
- Database is stored in the `database/` directory as `fingerprints.db`.
- Intermediate results and visualizations are saved in the `results/` directory.
- The system is modular; you can extend or replace algorithms as needed.

## Troubleshooting

- If thinning fails, ensure `opencv-contrib-python` is installed for `cv2.ximgproc`.
- For GUI errors, ensure Tkinter is available (included in most Python installations).
- For database issues, check permissions and existence of the `database/` directory.

## License

This project is intended for educational and research purposes. Please cite appropriately if used in publications.

---

For further details, refer to the source code and comments in each module.