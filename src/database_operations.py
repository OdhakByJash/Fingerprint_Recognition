# src/database_operations.py

import sqlite3
import json
import os
import datetime # To get timestamp for enrollment_date

# --- Database Configuration ---
# Get the directory where this script is located (src)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the database directory relative to the script's location (src -> parent -> database)
DB_DIR = os.path.abspath(os.path.join(_SRC_DIR, "../database"))
DB_NAME = "fingerprints.db"
DB_PATH = os.path.join(DB_DIR, DB_NAME)

# --- Database Functions ---

def initialize_database():
    """
    Initializes the SQLite database.
    - Creates the database directory if it doesn't exist.
    - Creates the database file if it doesn't exist.
    - Creates the 'fingerprints' table if it doesn't exist.
    - Creates an index on user_id for faster lookups if it doesn't exist.
    """
    try:
        # Ensure the directory exists
        os.makedirs(DB_DIR, exist_ok=True)
        print(f"DEBUG: Database directory ensured at: {DB_DIR}")

        # Connect to the database (creates the file if it doesn't exist)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        print(f"DEBUG: Connected to database at: {DB_PATH}")

        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fingerprints (
                fingerprint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                template_data TEXT NOT NULL,
                enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                -- Add other columns here if needed in the future, e.g.:
                -- image_quality REAL,
                -- source_image_path TEXT
            )
        ''')
        print("DEBUG: 'fingerprints' table ensured.")

        # Create an index for faster lookups by user_id, if it doesn't exist
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_user_id ON fingerprints (user_id)
        ''')
        print("DEBUG: 'idx_user_id' index ensured.")

        conn.commit() # Save changes
        conn.close()  # Close connection
        print(f"Database initialized successfully at {DB_PATH}")

    except sqlite3.Error as e:
        print(f"[ERROR] SQLite error during initialization: {e}")
        # Re-raise the exception or handle it as needed
        raise e
    except OSError as e:
        print(f"[ERROR] OS error creating database directory '{DB_DIR}': {e}")
        raise e
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during database initialization: {e}")
        raise e

def store_template(user_id, template_json):
    """
    Stores the fingerprint template JSON string for a given user ID in the database.

    Args:
        user_id (str): The identifier for the user.
        template_json (str): The JSON string representing the fingerprint template.

    Returns:
        int: The row ID (fingerprint_id) of the newly inserted record, or None if an error occurred.
    """
    conn = None # Initialize conn to None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Use CURRENT_TIMESTAMP for the enrollment date (handled by default schema)
        cursor.execute('''
            INSERT INTO fingerprints (user_id, template_data)
            VALUES (?, ?)
        ''', (user_id, template_json))

        conn.commit() # Commit the transaction
        last_id = cursor.lastrowid # Get the ID of the inserted row
        print(f"DEBUG: Stored template for user_id '{user_id}' with fingerprint_id {last_id}")
        return last_id

    except sqlite3.Error as e:
        print(f"[ERROR] SQLite error storing template for user '{user_id}': {e}")
        if conn:
            conn.rollback() # Rollback changes on error
        return None # Indicate failure
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred storing template for user '{user_id}': {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close() # Ensure connection is closed

def check_user_exists(user_id):
     """
     Checks if a user ID already has any enrolled fingerprints in the database.

     Args:
         user_id (str): The user identifier to check.

     Returns:
         bool: True if at least one record exists for the user_id, False otherwise.
     """
     conn = None
     try:
         conn = sqlite3.connect(DB_PATH)
         cursor = conn.cursor()
         # Use EXISTS for efficiency, or COUNT(*) if you need the exact number
         # cursor.execute("SELECT COUNT(*) FROM fingerprints WHERE user_id = ?", (user_id,))
         # count = cursor.fetchone()[0]
         # return count > 0
         cursor.execute("SELECT 1 FROM fingerprints WHERE user_id = ? LIMIT 1", (user_id,))
         exists = cursor.fetchone() is not None
         return exists

     except sqlite3.Error as e:
         print(f"[ERROR] SQLite error checking user '{user_id}': {e}")
         return False # Assume doesn't exist on error, or handle differently
     except Exception as e:
        print(f"[ERROR] An unexpected error occurred checking user '{user_id}': {e}")
        return False
     finally:
         if conn:
             conn.close()

# --- Standalone Test (Optional) ---
if __name__ == "__main__":
    print("Running database operations module test...")

    # 1. Initialize the database
    print("\n--- Testing Initialization ---")
    try:
        initialize_database()
        print("Initialization test successful (check for fingerprints.db in ../database)")
    except Exception as e:
        print(f"Initialization test FAILED: {e}")
        # If init fails, subsequent tests might not work

    # 2. Test Storing a template
    print("\n--- Testing Storage ---")
    test_user = "test_user_001"
    test_template = '{"features": {"minutiae": [{"x": 10, "y": 20, "type": "ending", "angle": 45.0}]}}' # Example JSON
    stored_id = store_template(test_user, test_template)
    if stored_id is not None:
        print(f"Storage test successful. Record ID: {stored_id}")
    else:
        print("Storage test FAILED.")

    # 3. Test Checking user existence
    print("\n--- Testing User Check ---")
    # Check for the user just added
    exists = check_user_exists(test_user)
    print(f"Check for '{test_user}' returned: {exists}")
    assert exists is True, f"Test failed: User '{test_user}' should exist."

    # Check for a non-existent user
    non_existent_user = "non_existent_user_999"
    exists = check_user_exists(non_existent_user)
    print(f"Check for '{non_existent_user}' returned: {exists}")
    assert exists is False, f"Test failed: User '{non_existent_user}' should not exist."

    print("\n--- Manual Verification ---")
    print(f"Please manually check the '{DB_PATH}' file using a tool like DB Browser for SQLite.")
    print(f"Look for the 'fingerprints' table and the record for user '{test_user}'.")

    # Optional: Clean up test data (be careful with this)
    # print("\n--- Cleaning up test data ---")
    # try:
    #     conn = sqlite3.connect(DB_PATH)
    #     cursor = conn.cursor()
    #     cursor.execute("DELETE FROM fingerprints WHERE user_id = ?", (test_user,))
    #     conn.commit()
    #     print(f"Deleted test record for '{test_user}'.")
    #     conn.close()
    # except Exception as e:
    #     print(f"Error during cleanup: {e}")


    print("\nDatabase operations module test finished.")
    
# Add this function to src/database_operations.py
# Make sure json is imported at the top

# ... (keep existing functions: initialize_database, store_template, check_user_exists) ...

def get_templates_for_user(user_id):
    """
    Retrieves all stored fingerprint templates for a given user ID.

    Args:
        user_id (str): The identifier of the user whose templates are needed.

    Returns:
        list: A list of parsed template dictionaries (containing minutiae lists, etc.).
              Returns an empty list if the user is not found or an error occurs.
              Returns None if a database connection error occurs.
    """
    conn = None
    templates = []
    try:
        # Check if DB exists first? Optional, connect will fail anyway if not.
        if not os.path.exists(DB_PATH):
             print(f"[Error] Database file not found at {DB_PATH}")
             return None # Indicate DB connection failed

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Select the template data for the given user_id
        cursor.execute('''
            SELECT template_data FROM fingerprints WHERE user_id = ?
        ''', (user_id,))

        rows = cursor.fetchall() # Fetch all rows matching the user_id

        if not rows:
            print(f"Info: No templates found for user_id '{user_id}'.")
            return [] # Return empty list if user not found

        # Parse the JSON data from each row
        for row in rows:
            template_json = row[0]
            try:
                template_data = json.loads(template_json)
                # Basic validation: Check if expected keys exist
                if 'features' in template_data and 'minutiae' in template_data['features']:
                     templates.append(template_data)
                else:
                     print(f"Warning: Skipping invalid template format for user '{user_id}': Missing 'features' or 'minutiae'.")
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse stored JSON template for user '{user_id}': {e}")
                # Optionally skip this template or handle error differently

        print(f"DEBUG: Retrieved {len(templates)} valid templates for user_id '{user_id}'.")
        return templates

    except sqlite3.Error as e:
        print(f"[ERROR] SQLite error retrieving templates for user '{user_id}': {e}")
        return None # Indicate failure due to DB error
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred retrieving templates for user '{user_id}': {e}")
        return None
    finally:
        if conn:
            conn.close()

# ... (keep standalone test block if desired) ...
# ... (keep existing functions: initialize_database, store_template, check_user_exists, get_templates_for_user) ...

def get_all_templates():
    """
    Retrieves all stored fingerprint templates from the database.

    Returns:
        list: A list of tuples, where each tuple contains:
              (user_id, fingerprint_id, parsed_template_dictionary).
              Returns an empty list if the database is empty or an error occurs retrieving data.
              Returns None if a database connection error occurs.
    """
    conn = None
    all_templates = []
    try:
        if not os.path.exists(DB_PATH):
             print(f"[Error] Database file not found at {DB_PATH}")
             return None # Indicate DB connection failed

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Select user_id, fingerprint_id, and template_data for all records
        cursor.execute('''
            SELECT user_id, fingerprint_id, template_data FROM fingerprints
        ''')

        rows = cursor.fetchall() # Fetch all rows

        if not rows:
            print("Info: Database contains no fingerprint templates.")
            return [] # Return empty list if DB is empty

        # Parse the JSON data from each row
        parsed_count = 0
        skipped_count = 0
        for row in rows:
            user_id, fingerprint_id, template_json = row
            try:
                template_data = json.loads(template_json)
                # Basic validation
                if 'features' in template_data and 'minutiae' in template_data['features']:
                     all_templates.append((user_id, fingerprint_id, template_data))
                     parsed_count += 1
                else:
                     print(f"Warning: Skipping invalid template format for user '{user_id}', fp_id {fingerprint_id}.")
                     skipped_count += 1
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse stored JSON template for user '{user_id}', fp_id {fingerprint_id}: {e}")
                skipped_count += 1

        print(f"DEBUG: Retrieved and parsed {parsed_count} templates (skipped {skipped_count}) from the database.")
        return all_templates

    except sqlite3.Error as e:
        print(f"[ERROR] SQLite error retrieving all templates: {e}")
        return None # Indicate failure due to DB error
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred retrieving all templates: {e}")
        return None
    finally:
        if conn:
            conn.close()
