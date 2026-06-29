# BINT-Mesa

**Simulation-Based Evaluation of a Trustee-Centered Multi-Agent Trust Management System**

This repository contains a Mesa-based multi-agent simulation for studying **BINT**: **Bidirectional NFT-based Trust**, a trustee-centered trust management mechanism for open multi-agent systems. The simulation uses a delivery and map-sharing scenario where agents must decide whether to rely on information provided by other agents under uncertainty.

The project was developed during a Master internship at LITIS in the context of the MaestrIoT project, supervised by Dr. Maxime Gueriau and Prof. Laurent Vercouter. It follows Marc Saideh's work on BINT, where trustees are not only passive objects of evaluation but can actively hold and commit verifiable trust capital.

---

## Current project status

The repository currently supports three trust/evaluation modes:

| Mode | Meaning | Main purpose |
|---|---|---|
| `bint` with staking | Full BINT mechanism: TNFT/VTP trust evidence + bilateral staking | Main implemented mechanism |
| `bint` without staking | BINT trust filtering only | Ablation baseline |
| `brs` | Classical Beta Reputation System using success/failure counts | Functional comparison baseline |

Important caveat: the **BRS implementation is functional and integrated**, but it is currently only used through the dashboard and the lightweight `compare_brs_bint.py` smoke comparison. It is **not yet a thesis-scale evaluation** unless someone extends the notebooks to include BRS in the full experimental sweep.

---

## Quick start

### 1. Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project was developed with Python 3.12+. The dependencies are pinned in `requirements.txt`.

### 2. Run the dashboard

```bash
solara run app.py
```

Then open the local URL printed by Solara, usually:

```text
http://localhost:8765
```

Use the **Scenario** dropdown at the top of the dashboard.

| Scenario | Trust model | Staking | Meaning |
|---|---|---:|---|
| `default` | BINT | yes | 3 honest + 2 malicious agents, medium attacks |
| `brs_default` | BRS | no | Same small scenario, but with Beta Reputation System |
| `honest_only` | BINT | yes | 5 honest agents, no malicious agents |
| `aggressive_malicious` | BINT | yes | 3 honest + 2 malicious agents, all attack probabilities set to 1 |

The dashboard has buttons for **Step 1**, **Step 50**, **Step 500**, and **Reset scenario**. Switching scenarios or resetting creates a fresh model instance.

### 3. Run a quick BINT vs BRS check

```bash
python compare_brs_bint.py --steps 1000 --seeds 7 1337 2024
```

This prints a small comparison table for BINT and BRS. It is only a smoke comparison, not a full evaluation. A useful sanity check is that the BRS row should show **0 active TNFTs** and **0 burned TNFTs**, because BRS does not use the token economy.

---

## Fast sanity checks before changing anything

From the repository root, run:

```bash
python -m compileall -q agents.py model.py profiles.py app.py compare_brs_bint.py bint_golden.py
python -m pytest -q test_brs.py test_harness_machinery.py
PYTHONHASHSEED=0 python -m pytest -q
PYTHONHASHSEED=0 python bint_golden.py verify --profile fast
PYTHONHASHSEED=0 python bint_golden.py verify --profile fast --check-caches
python compare_brs_bint.py --steps 1000 --seeds 7 1337 2024
```

Expected outcome:

- Unit tests pass.
- Golden-master verification reports that the current BINT behavior is identical to the saved baseline.
- Cache verification passes.
- The BINT/BRS comparison prints two rows: `bint` and `brs`.
- In that comparison, BRS has no TNFT activity.

Do **not** run `bint_golden.py capture` unless you intentionally want to replace the saved golden baseline. For normal checking, use `verify`.

More detail is in [`README_golden.md`](README_golden.md).

---

## Repository structure

```text
BINT-mesa/
├── agents.py                         # Delivery agents, malicious agents, trust decisions, staking decisions
├── model.py                          # Mesa model, grid, TNFT ledger, interactions, outcomes, BRS evidence
├── profiles.py                       # Dashboard scenario definitions and default parameters
├── app.py                            # Solara dashboard entry point
├── compare_brs_bint.py               # Lightweight BINT-vs-BRS smoke comparison
├── bint_eval_clean.ipynb             # Main thesis evaluation notebook
├── bint_parameter_exploration.ipynb  # Parameter exploration/debugging notebook
├── bint_golden.py                    # Golden-master harness for behavior preservation checks
├── README_golden.md                  # Golden-master workflow documentation
├── test_brs.py                       # Unit tests for BRS integration
├── test_bint_golden.py               # Pytest wrapper for golden verification
├── test_harness_machinery.py         # Unit tests for the golden harness itself
├── bint_golden_baseline/             # Saved golden snapshots and manifest
├── evaluation_outputs/               # Saved evaluation tables/figures from notebook runs
├── results/                          # Runtime output directory, if produced by notebooks/scripts
├── Dockerfile                        # Docker image definition
├── docker-compose.yml                # Dashboard/notebook services
├── requirements.txt                  # Pinned Python environment
├── .gitignore
└── README.md
```

### Core file map for future developers

| Task | Start here |
|---|---|
| Change agent behavior | `agents.py` |
| Change the world/grid/package dispatch logic | `model.py` |
| Change BINT TNFT mint/burn/lock behavior | `model.py` |
| Change BINT trust-score calculation | `agents.py`, especially `calculate_trust_summary(...)` and related helpers |
| Change BRS scoring | `agents.py`, especially `calculate_brs_trust_summary(...)` |
| Change BRS evidence updates | `model.py`, especially `record_brs_outcome(...)` |
| Add or edit dashboard scenarios | `profiles.py` |
| Change dashboard visuals/metrics | `app.py` |
| Run thesis-scale experiments | `bint_eval_clean.ipynb` |
| Explore parameters quickly | `bint_parameter_exploration.ipynb` |
| Check behavior preservation after refactoring | `bint_golden.py` and `README_golden.md` |

---

## Simulation overview

### Scenario

The simulation is a delivery and map-sharing environment:

1. Delivery agents move on a two-dimensional grid.
2. The grid contains fixed drop-off locations.
3. Agents start with only partial map knowledge.
4. When an agent needs a drop-off location it does not know, it may request map information from other agents.
5. The requester is the **truster**; the information provider is the **trustee**.
6. Honest providers share correct known coordinates.
7. Malicious providers may fabricate coordinates.
8. Requesters later discover whether the received coordinate was useful or false.
9. The interaction outcome updates trust evidence.

This creates dependency situations: accepting correct information can improve delivery performance, while accepting false information can waste time and cause failed interactions.

### Agent types

| Agent | Class | Behavior |
|---|---|---|
| Honest delivery agent | `DeliveryAgent` | Shares correct map information when available and reports outcomes truthfully |
| Malicious delivery agent | `MaliciousDeliveryAgent` | May share false coordinates and may submit false positive or false negative reviews |
| Drop-off location | `DropOffLocationAgent` | Passive grid marker for delivery destinations |

### Interaction lifecycle

A typical map-sharing interaction follows this path:

1. A requester receives a delivery task.
2. If it does not know the target coordinate, it asks peers for map information.
3. Candidate providers decide whether and how to respond.
4. The requester evaluates the provider using the active trust model.
5. The provider may also evaluate the requester.
6. If BINT staking is enabled, both sides must lock enough active TNFTs as collateral.
7. If the interaction is accepted, the requester stores the received coordinate as candidate knowledge.
8. The interaction remains pending until the requester can confirm success or failure.
9. On settlement:
   - BINT mints, burns, releases, or burns staked TNFTs depending on the outcome.
   - BRS updates success/failure evidence for the trustee.

---

## Trust mechanisms

### BINT: TNFT/VTP trust evidence

BINT represents trust evidence using **Trust Non-Fungible Tokens (TNFTs)**.

Each delivery agent has a **Verifiable Trust Portfolio (VTP)**: the set of TNFTs owned by that agent. In this implementation, the blockchain layer is abstracted as an in-memory ledger in `model.py`.

In BINT mode:

- Agents start with genesis TNFTs to reduce the cold-start problem.
- Successful interactions mint positive TNFT evidence for the trustee.
- Failed interactions burn trust evidence.
- Same-service evidence is weighted more strongly than other-service evidence.
- A smoothed weighted ratio produces a trust score in `[0, 1]`.
- The two-threshold policy rejects low trust, accepts high trust, and samples probabilistically in the middle region.

The trust thresholds used in the main experiments are:

| Threshold | Value | Meaning |
|---|---:|---|
| Reject threshold | 0.30 | Scores at or below this are always rejected |
| Accept threshold | 0.80 | Scores at or above this are always accepted |

### BINT staking

When staking is enabled, both parties must commit active TNFTs as collateral before an interaction is created.

- Required stake increases with perceived risk.
- Locked TNFTs cannot be reused in another pending interaction.
- On success, locked TNFTs are released and the trustee receives a new TNFT.
- On failure, locked TNFTs are burned.

This is the main trustee-centered part of the implementation: the trustee's trust capital is not only observed, but can also be committed as a costly signal.

### BRS: Beta Reputation System baseline

BRS is the classical reputation baseline added for comparison.

When `trust_model="brs"`:

- TNFTs are not used for trust scoring.
- Staking is automatically disabled.
- The model stores success/failure counts per trustee and service context.
- Reported successes increment positive evidence.
- Reported failures increment negative evidence.
- Optional forgetting can decay older evidence before each new update.
- Trust is computed as:

```text
alpha = brs_prior_success + successes
beta  = brs_prior_failure + failures
trust = alpha / (alpha + beta)
```

Default BRS priors are `1.0` and `1.0`, so an agent with no BRS evidence starts at `0.5`.

BRS is useful as a familiar baseline because it uses the same interaction outcomes as BINT but removes the token portfolio and staking economy.

### Accept-all baseline

The accept-all baseline disables meaningful trust filtering by using very permissive thresholds. It is mainly used as a lower-bound comparison in the thesis evaluation notebook.

---

## Scenarios and parameter sets

There are two different scales to keep in mind.

### Dashboard scale

Defined in `profiles.py`; intended to be fast and visual.

| Parameter | Dashboard default |
|---|---:|
| Grid size | 50 × 50 |
| Drop-off locations | 5 |
| Honest agents | 3 |
| Malicious agents | 2 |
| Max steps | 2,000 |
| Genesis TNFTs | 5 |
| Vision radius | 2 |
| Medium attack probabilities | 0.50 |
| Reviewer check activation | 8 reviews |
| Staking fraction range | 0.25–1.00 |

### Thesis evaluation scale

Defined in `bint_eval_clean.ipynb`; intended for reported results.

| Parameter | Thesis evaluation value |
|---|---:|
| Grid size | 150 × 150 |
| Drop-off locations | 35 |
| Honest agents | 14 |
| Malicious agents | 6 |
| Max steps | 5,000 |
| Genesis TNFTs | 5 |
| Vision radius | 2 |
| Medium attack probabilities | 0.50 |
| Reviewer check activation | 8 reviews |
| Staking fraction range | 0.10–0.90 |

The dashboard is intentionally smaller than the notebook experiments. Do not compare dashboard numbers directly with thesis tables.

---

## Key parameters

Most dashboard defaults live in `profiles.py`. The thesis evaluation notebook has its own `FINAL_BASE_PARAMETERS` dictionary.

| Parameter | Typical value | Meaning |
|---|---:|---|
| `trust_reject_threshold` | 0.30 | Trust score at/below which interactions are rejected |
| `trust_accept_threshold` | 0.80 | Trust score at/above which interactions are accepted |
| `genesis_tokens` | 5 | Initial TNFTs per delivery agent |
| `context_match_weight` | 1.00 | Weight for same-service trust evidence |
| `other_context_weight` | 0.25 | Weight for other-service trust evidence |
| `max_negative_review_rate` | 0.60 | Reviewer credibility cutoff |
| `min_reviews_before_reviewer_check` | 8 | Reviews required before reviewer credibility is enforced |
| `false_map_probability` | 0.50 | Probability a malicious provider fabricates map data |
| `false_negative_review_probability` | 0.50 | Probability a malicious requester reports a success as failure |
| `false_positive_review_probability` | 0.50 | Probability a malicious requester reports a failure as success |
| `staking_min_fraction` | 0.10 or 0.25 | Minimum stake fraction; notebook uses 0.10, dashboard uses 0.25 |
| `staking_max_fraction` | 0.90 or 1.00 | Maximum stake fraction; notebook uses 0.90, dashboard uses 1.00 |
| `brs_prior_success` | 1.00 | BRS positive prior |
| `brs_prior_failure` | 1.00 | BRS negative prior |
| `brs_forgetting_factor` | 1.00 | BRS evidence decay factor; 1.00 means no decay |

---

## Running experiments

### Main thesis notebook

```bash
jupyter lab bint_eval_clean.ipynb
```

This notebook contains the main ablation, sensitivity, and corner-attack experiments used for the thesis/report figures.

High-level structure:

| Section | Content |
|---|---|
| 0–7 | Imports, helper functions, common parameters, metric collection, plotting utilities |
| 8 | Experiment definitions |
| 9 | Output file paths |
| 10 | Full experiment execution |
| 11+ | Loading saved data, summary tables, and figures |

The full run can take a while because it uses many seeds and configurations. Results are saved to disk, usually under `evaluation_outputs/` or `results/`, depending on the notebook version.

### Parameter exploration notebook

```bash
jupyter lab bint_parameter_exploration.ipynb
```

Use this notebook for tuning and debugging. It is not required for reproducing the final thesis figures.

### Lightweight BRS comparison

```bash
python compare_brs_bint.py --steps 1000 --seeds 7 1337 2024
```

Useful optional arguments:

```bash
python compare_brs_bint.py \
  --steps 2000 \
  --width 80 \
  --height 80 \
  --dropoffs 25 \
  --honest-agents 14 \
  --malicious-agents 6 \
  --brs-forgetting-factor 1.0 \
  --seeds 7 1337 2024
```

Again, this is a smoke comparison. To make BRS a real evaluated baseline, add it to the full notebook experiment definitions and regenerate the tables/figures.

---

## Docker alternative

If Docker is installed, you can avoid local Python setup.

```bash
docker compose build
docker compose up dashboard     # dashboard at http://localhost:8765
docker compose up notebooks     # notebooks at http://localhost:8888
```

Results written by notebooks are mounted back to the local project directory.

---

## How to extend the project

### Add a new dashboard scenario

Edit `profiles.py`:

1. Add a new `ScenarioConfig` entry to `SCENARIO_CONFIGS`.
2. Set the number of honest and malicious agents.
3. Set attack probabilities.
4. Choose `trust_model="bint"` or `trust_model="brs"`.
5. For BRS, set `staking_enabled=False`.
6. Add the scenario name to `SCENARIO_OPTIONS` in `app.py` if you want it visible in the dashboard dropdown.

After that, run:

```bash
python -m compileall -q profiles.py app.py
solara run app.py
```

### Add a new trust model

The main integration points are:

| Step | File/function |
|---|---|
| Accept the model name | `model.py`, `_normalise_trust_model(...)` |
| Store model-specific evidence | `model.py`, near the BRS evidence structures |
| Update evidence on settlement | `model.py`, `record_outcome(...)` / `settle_interaction(...)` |
| Compute trust score | `agents.py`, `calculate_trust_summary(...)` |
| Expose metrics | `model.py`, `get_vtp_summary(...)` or a compatible summary function |
| Add a dashboard scenario | `profiles.py` |
| Add quick comparison | `compare_brs_bint.py` or a new runner |
| Add tests | `test_*.py` |

Try to return a summary dictionary compatible with the current BINT/BRS shape. This keeps the dashboard and notebooks easier to reuse.

### Add a new metric

For dashboard-only metrics, edit `app.py`, usually `RunSummary(...)` or `AgentDecisions(...)`.

For experiment metrics, edit the metric collection helpers in `bint_eval_clean.ipynb`. If the metric should also appear in the BINT/BRS smoke runner, update `compare_brs_bint.py`.

### Change staking behavior

Staking is split between agent-side decisions and model-side ledger effects.

- Risk and required stake decisions are mainly in `agents.py`.
- Locking, releasing, and burning TNFT collateral are in `model.py`.
- Settlement rules are in `model.py`, especially `_apply_outcome_to_interaction(...)`.

After touching staking, always run:

```bash
PYTHONHASHSEED=0 python bint_golden.py verify --profile fast --check-caches
```

### Change trust scoring

For BINT trust scoring, look in `agents.py` around:

- `calculate_trust_summary(...)`
- `calculate_trust_summary_from_evidence(...)`
- evidence filtering helpers

For BRS trust scoring, look in:

- `calculate_brs_trust_summary(...)` in `agents.py`
- `record_brs_outcome(...)` and `get_brs_evidence(...)` in `model.py`

If the change is intentional, update tests and decide whether the golden baseline should be recaptured. Do not recapture just to hide an accidental behavior change.

---

## Golden-master testing

This project includes characterization tests that lock the current BINT behavior. They are useful when refactoring because they tell you whether the simulation changed at the behavioral level.

Common commands:

```bash
PYTHONHASHSEED=0 python bint_golden.py selfcheck
PYTHONHASHSEED=0 python bint_golden.py verify --profile fast
PYTHONHASHSEED=0 python bint_golden.py verify --profile fast --check-caches
PYTHONHASHSEED=0 python bint_golden.py verify --profile thorough
```

Use `--profile fast` for everyday checks. Use `--profile thorough` before larger merges or after risky changes.

See [`README_golden.md`](README_golden.md) for the full workflow.

---

## Results summary

The thesis/report evaluation compares four main configurations over 100 independent seeds, using 14 honest and 6 malicious agents over 5,000 steps with medium attack probabilities of 0.50.

| Configuration | Honest pts/agent | Malicious pts/agent | Failure rate | Malicious provider rate |
|---|---:|---:|---:|---:|
| Honest-only | 692 | — | 0.000 | 0.000 |
| Accept-all | 541 | 558 | 0.416 | 0.490 |
| BINT baseline | 527 | 27 | 0.173 | 0.171 |
| BINT | 480 | -1 | 0.105 | 0.115 |

In this simulation setting, trust filtering substantially reduces interactions with malicious providers compared with accept-all. Adding staking further reduces failures and makes the default malicious strategy unprofitable in the main evaluated scenario, although it also lowers interaction volume and honest-agent reward.

These are simulation-specific results, not general guarantees for all multi-agent or IoT environments.

---

## Known limitations and future work

- The blockchain layer is abstracted as an in-memory ledger. Consensus, latency, gas cost, and smart-contract execution are not modeled.
- The simulation scenario is intentionally simplified: delivery/map sharing is a controlled dependency situation, not a full IoT deployment.
- BRS is currently integrated as a functional baseline but not yet included in the full thesis-scale notebook evaluation.
- The current results depend on the selected parameter regime.
- The golden-master tests preserve existing behavior; they do not prove the behavior is theoretically correct.

Possible next steps:

1. Add BRS to the full evaluation notebook and regenerate comparison figures.
2. Evaluate more BRS variants, such as time decay or reviewer-weighted BRS.
3. Add additional malicious strategies, such as on-off behavior or collusion.
4. Model blockchain overhead explicitly.
5. Add more dashboard scenarios for sensitivity testing.
6. Add dashboard-specific unit tests if the dashboard is expected to be maintained long term.
