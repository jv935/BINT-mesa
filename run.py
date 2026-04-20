import sys
import os
import csv
import numpy as np
from mesa import batch_run
from model import BintWorldModel

params = {
    "size": [(150, 150)],
    "num_drop_offs": [5, 10, 15],
    "agent_vision_radius": [2],
    "num_delivery": [7, 10],
    "num_map_malicious": range(6),
}

if __name__ == "__main__":
    chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    csv_filename = sys.argv[2] if len(sys.argv) > 2 else "results.csv"

    rng = np.random.default_rng()
    rng_values = rng.integers(0, sys.maxsize, size=chunk_size)

    results = batch_run(
        model_cls=BintWorldModel,
        parameters=params,
        rng=rng_values.tolist(),
        max_steps=1000,
        number_processes=3,
        data_collection_period=1,
        display_progress=True
    )

    print("Complete! Saving to CSV...")

    if results:
        keys = list(results[0].keys())
        write_header = not os.path.exists(csv_filename)

        with open(csv_filename, "a", newline="", buffering=10_485_760) as output_file:
            writer = csv.DictWriter(output_file, fieldnames=keys)
            if write_header:
                writer.writeheader()
            writer.writerows(results)

    print(f"Success! Wrote {len(results)} rows to {csv_filename}")