import sys
import os
import csv
import pandas as pd
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
    rng_values = rng.integers(0, sys.maxsize, size=(chunk_size,))

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
    # results_df = pd.DataFrame(results)

    write_header = not os.path.exists(csv_filename)

    if results:
        keys = list(results[0].keys())

        # buffering=10485760 gives the OS a 10MB buffer
        with open(csv_filename, "a", newline="", buffering=10485760) as output_file:
            writer = csv.writer(output_file)

            if write_header:
                writer.writerow(keys)

            writer.writerows(row.values() for row in results)

    # if results:
    #     keys = results[0].keys()
    #
    #     with open(csv_filename, "a", newline="") as output_file:
    #         dict_writer = csv.DictWriter(output_file, fieldnames=keys)
    #
    #         if write_header:
    #             dict_writer.writeheader()
    #
    #         dict_writer.writerows(results)

    # results_df.to_csv(csv_filename, mode="a", index=False, header=write_header)
    print(f"Success!")
