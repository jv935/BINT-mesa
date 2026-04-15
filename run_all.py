import subprocess
import os
import sys

TOTAL_ITERATIONS = 50
CHUNK_SIZE = 5
CSV_FILE = "bint_batch_results.csv"

if __name__ == "__main__":
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
        print("Deleted old csv file.")

    print(f"--- Starting sweep: {TOTAL_ITERATIONS} iterations ---")

    for start in range(0, TOTAL_ITERATIONS, CHUNK_SIZE):
        subprocess.run([sys.executable, "run.py", str(CHUNK_SIZE), CSV_FILE], check=True)

    print(f"Completed sweep! Saved to {CSV_FILE}")