import argparse
import os
import shutil
import subprocess
import sys


TOTAL_ITERATIONS = 15
CHUNK_SIZE = 3
CSV_FILE = "bint_batch_results.csv"
EXPORT_DIR = "exports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BINT sweep chunks.")
    parser.add_argument("--total-iterations", type=int, default=TOTAL_ITERATIONS)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--csv", default=CSV_FILE)
    parser.add_argument("--export-dir", default=EXPORT_DIR)
    parser.add_argument("--suite", choices=["main", "robustness", "ablation", "smoke"], default="main")
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--keep-old-exports", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if os.path.exists(args.csv):
        os.remove(args.csv)
        print(f"Deleted old csv file: {args.csv}")

    if os.path.exists(args.export_dir) and not args.keep_old_exports:
        shutil.rmtree(args.export_dir)
        print(f"Deleted old export directory: {args.export_dir}")

    os.makedirs(args.export_dir, exist_ok=True)

    total_chunks = (args.total_iterations + args.chunk_size - 1) // args.chunk_size
    chunk_num = 1

    print(
        f"--- Starting sweep: {args.total_iterations} seed replications per scenario | "
        f"suite={args.suite} | chunk_size={args.chunk_size} ---"
    )

    for start in range(0, args.total_iterations, args.chunk_size):
        current_chunk = min(args.chunk_size, args.total_iterations - start)

        chunk_seed = None if args.seed is None else args.seed + chunk_num - 1

        print(f"\nChunk {chunk_num}/{total_chunks} | Replications {start + 1} to {start + current_chunk}")

        command = [
            sys.executable,
            "run.py",
            str(current_chunk),
            args.csv,
            "--suite",
            args.suite,
            "--processes",
            str(args.processes),
            "--max-steps",
            str(args.max_steps),
            "--export-dir",
            args.export_dir,
        ]

        if chunk_seed is not None:
            command.extend(["--seed", str(chunk_seed)])

        subprocess.run(command, check=True)
        chunk_num += 1

    print(f"Completed sweep! Saved CSV to {args.csv} and JSON exports to {args.export_dir}/")
