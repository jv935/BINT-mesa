import sys
import os
import csv
import pandas as pd
import numpy as np
from mesa import batch_run
from model import BintWorldModel

params = {
    "width": [100],
    "height": [100],
    "num_drop_offs": [5, 10, 15],
    "agent_vision_radius": [2],
    "num_delivery": [5, 7, 10],
    "num_map_malicious": range(6),
}

if __name__ == "__main__":
    chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    csv_filename = sys.argv[2] if len(sys.argv) > 2 else "results.csv"

#    seed = int(time.time() * 1000) % 123456789
    rng = np.random.default_rng()
    rng_values = rng.integers(0, sys.maxsize, size=(chunk_size,))

    print(f"Starting chunk of size {chunk_size}")

    results = batch_run(
        model_cls=BintWorldModel,
        parameters=params,
        rng=rng_values.tolist(),
        max_steps=1500,
        number_processes=3,
        data_collection_period=1,
        display_progress=True
    )

    print("Complete! Saving to CSV...")
    results_df = pd.DataFrame(results)

    write_header = not os.path.exists(csv_filename)

    # if results:
    #     keys = results[0].keys()
    #     with open(csv_filename, "a", newline="") as output_file:
    #         dict_writer = csv.DictWriter(output_file, fieldnames=keys)
    #         if write_header:
    #             dict_writer.writeheader()
    #         dict_writer.writerows(results)

    results_df.to_csv(csv_filename, mode="a", index=False, header=write_header)
    print(f"Success!")