import subprocess
import os
import sys

TOTAL_ITERATIONS = 100
CHUNK_SIZE = 10
CSV_FILE = "bint_batch_results.csv"

if __name__ == "__main__":
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
        print("Deleted old csv file.")
    
    total_chunks = (TOTAL_ITERATIONS + CHUNK_SIZE - 1) // CHUNK_SIZE
    chunk_num = 1

    print(f"--- Starting sweep: {TOTAL_ITERATIONS} iterations ---")

    for start in range(0, TOTAL_ITERATIONS, CHUNK_SIZE):
        current_chunk = min(CHUNK_SIZE, TOTAL_ITERATIONS-start)
    
        print(f"\n Chunk {chunk_num}/{total_chunks} | Iterations {start + 1} to {start + current_chunk}...")
        subprocess.run([sys.executable, "run.py", str(current_chunk), CSV_FILE], check=True)
        
        chunk_num += 1

    print(f"Completed sweep! Saved to {CSV_FILE}")
