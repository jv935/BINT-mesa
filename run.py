import sys

import pandas as pd
import numpy as np
from mesa import batch_run
from model import BintWorldModel

params = {
    "width": [100],
    "height": [100],
    "num_drop_offs": [5, 10, 15],
    "agent_vision_radius": [2],
    "num_delivery": [7],
    "num_map_malicious": [0, 1, 2, 3, 4, 5],
}

if __name__ == "__main__":
    rng = np.random.default_rng(1337)
    rng_values = rng.integers(0, sys.maxsize, size=(25,))

    print("Starting BINT Protocol Evaluation...")
    results = batch_run(
        model_cls=BintWorldModel,
        parameters=params,
        rng=rng_values.tolist(),
        max_steps=1500,
        number_processes=None,
        data_collection_period=4,
        display_progress=True
    )

    print("Simulation complete. Formatting data...")

    results_df = pd.DataFrame(results)

    results_df.to_csv("bint_experiment_results.csv", index=False)
    print("Saved to bint_experiment_results.csv!")