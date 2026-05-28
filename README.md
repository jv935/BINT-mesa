# BINT-Mesa

**Simulation-Based Evaluation of a Trustee-Centered Multi-Agent Trust Management System**

The project implements and evaluates the **BINT (Bidirectional NFT-based Trust)** model — a trustee-centered trust management mechanism for open multi-agent systems — in a delivery and map-sharing simulation.


---

## Table of Contents

- [Background](#background)
- [What This Project Does](#what-this-project-does)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Project](#running-the-project)
  - [Interactive Dashboard](#interactive-dashboard)
  - [Evaluation Experiments (Notebooks)](#evaluation-experiments-notebooks)
  - [Parameter Exploration](#parameter-exploration)
- [Simulation Overview](#simulation-overview)
  - [Scenario and Agents](#scenario-and-agents)
  - [BINT Mechanisms](#bint-mechanisms)
  - [Scenarios and Configurations](#scenarios-and-configurations)
- [Key Parameters](#key-parameters)
- [Results Summary](#results-summary)

---

## Background

Most computational trust models focus on the *truster*: the agent deciding whether to rely on another. The trusted agent (the *trustee*) is usually passive. **BINT**, proposed in Marc Saideh's PhD thesis, inverts this by giving trustees an active role: agents hold *Trust Non-Fungible Tokens (TNFTs)* as verifiable trust capital, accumulated through past trustworthy behavior, and may *stake* (commit) part of that capital as collateral before an interaction.

This project implements BINT in a controlled Mesa simulation and evaluates whether trust-based filtering and the staking mechanism reduce harmful interactions with malicious agents.

This internship was carried out under the [MaestrIoT project](https://anr.fr/Project-ANR-21-CE23-0016) (ANR-21-CE23-0016), supervised by Dr. Maxime Gueriau and Prof. Laurent Vercouter at LITIS.

---

## What This Project Does

- Implements a **delivery and map-sharing simulation** on a 2D grid using [Mesa](https://mesa.readthedocs.io/) (Python agent-based modelling framework).
- Realises the **TNFT/VTP core** of the BINT model as an in-memory ledger (the blockchain layer is abstracted).
- Introduces **malicious agent behaviors**: false map coordinates and dishonest outcome reviews.
- Implements a **bilateral staking mechanism** where both parties commit trust capital before each interaction.
- Provides an **interactive dashboard** (Solara) to run and visualise the simulation live.
- Provides **Jupyter notebooks** to reproduce all evaluation figures from the thesis.

---

## Repository Structure

```
BINT-mesa/
├── agents.py                      # Agent classes: DeliveryAgent, MaliciousDeliveryAgent, DropOffLocationAgent
├── model.py                       # BintWorldModel: Mesa model, TNFT ledger, interaction & outcome records
├── profiles.py                    # Scenario definitions and default parameter values
├── app.py                         # Solara interactive dashboard (entry point)
├── bint_eval_clean.ipynb          # Main evaluation notebook — reproduces all thesis figures
├── bint_parameter_exploration.ipynb  # Parameter exploration and debugging notebook
├── requirements.txt               # Full pinned dependency list (Python 3.12+)
├── .gitignore
└── README.md
```

**Core modules:**

| File | Role |
|------|------|
| `model.py` | Defines `BintWorldModel` (Mesa model), the in-memory TNFT ledger, `InteractionRecord`, `OutcomeRecord`, and `AgentProfile`. |
| `agents.py` | Defines `DeliveryAgent` (honest), `MaliciousDeliveryAgent`, and `DropOffLocationAgent`. Contains trust-score computation, staking logic, and reviewer-credibility checks. |
| `profiles.py` | Exports three named scenarios (`default`, `honest_only`, `aggressive_malicious`) and all default constants. |
| `app.py` | Interactive Solara dashboard: grid visualisation, live metrics panel, scenario selector, and step controls. |

---

## Requirements

- **Python 3.12** or later (the codebase uses modern type-hint syntax).
- All Python dependencies are listed in `requirements.txt`. Key packages:

| Package | Version | Purpose |
|---------|---------|---------|
| Mesa | 3.5.1 | Agent-based simulation framework |
| Solara | 1.57.3 | Interactive dashboard |
| JupyterLab | 4.5.6 | Notebook interface |
| NumPy | 2.4.3 | Numerical computation |
| pandas | 3.0.1 | Data handling in notebooks |
| matplotlib / seaborn | 3.10.8 / 0.13.2 | Plotting |
| scipy | 1.17.1 | Statistical analysis in notebooks |

> **Note:** `requirements.txt` lists the full environment with pinned versions. Installing into a clean virtual environment is strongly recommended.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/jv935/BINT-mesa.git
cd BINT-mesa

# 2. Create and activate a virtual environment (recommended)
python3.12 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> If you only want to run the interactive dashboard without notebooks, a lighter install is possible with just `mesa`, `solara`, and `matplotlib`. However, the notebooks require all dependencies.

---

### Docker (alternative)

If you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed,
no Python setup is required.

```bash
# First build (takes ~10 minutes the first time)
docker compose build

# Run the dashboard
docker compose up dashboard   # → http://localhost:8765

# Run the notebooks
docker compose up notebooks   # → http://localhost:8888
```

Results written by the notebooks are saved to `results/` on your local machine.

---

## Running the Project

### Interactive Dashboard

The dashboard lets you run the simulation step-by-step or continuously, with a live grid view and real-time metric charts.

```bash
solara run app.py
```

This opens a browser tab at `http://localhost:8765`. You can select a scenario (`default`, `honest_only`, `aggressive_malicious`), adjust trust thresholds, toggle staking, and watch agents move across the grid.

**What you see:**
- Blue circles — honest delivery agents.
- Red crosses — malicious delivery agents.
- Black squares — fixed drop-off locations.
- Side panel — live trust scores, TNFT balances, reward totals, and interaction statistics.

### Evaluation Experiments (Notebooks)

The main notebook, `bint_eval_clean.ipynb`, contains all experiments and figures from the thesis.

```bash
jupyter lab bint_eval_clean.ipynb
```

The notebook is structured as follows:

| Section | Content |
|---------|---------|
| 0–7 | Imports, helpers, base config, metric collection, plotting utilities |
| 8 | Experiment definitions (ablation, sensitivity, corner-attack) |
| 9 | Output file paths |
| **10** | **Run all experiments** (uses `ProcessPoolExecutor`; 100 seeds × 4 configurations; can take 30–60 min on a typical laptop) |
| 11 | Load and prepare saved results |
| 12–24 | Summary tables and all figures (Figures 2–8 in the thesis) |

> **Section 10 saves results to disk.** If result files already exist, the cell skips re-running (skip-if-exists logic). You can re-run individual experiments by deleting the corresponding output files.

### Parameter Exploration

`bint_parameter_exploration.ipynb` is a debugging and tuning notebook, not needed to reproduce the thesis results. It is useful for understanding how individual parameters affect simulation behavior before committing to a configuration.

```bash
jupyter lab bint_parameter_exploration.ipynb
```

---

## Simulation Overview

### Scenario and Agents

The simulation runs on a 150 × 150 grid with 35 fixed drop-off locations. A population of delivery agents moves in discrete steps. Agents start with partial map knowledge; when they need a drop-off location they do not know, they can **request map information** from other agents. This map-sharing interaction is the central service: it creates dependency situations where trust matters.

- **Honest agents** (`DeliveryAgent`): Share correct coordinates and submit truthful outcome reviews.
- **Malicious agents** (`MaliciousDeliveryAgent`): May share fabricated coordinates and/or submit false reviews. Each attack type is governed by an independent probability parameter.

Agents accumulate **reward** through timely deliveries. Full reward (+10 pts) is granted for on-time arrivals; late arrivals receive discounted or negative reward.

### BINT Mechanisms

**TNFTs (Trust Non-Fungible Tokens):** Each agent holds a *Verifiable Trust Portfolio* (VTP). Successful interactions mint new TNFTs; failed interactions burn tokens. All agents start with 5 genesis tokens to avoid a complete cold-start.

**Trust scoring:** A weighted Beta-reputation formula converts the TNFT evidence in the ledger to a trust score in [0, 1]. Same-service evidence is weighted more heavily than cross-service evidence. A two-threshold acceptance policy (reject ≤ 0.30, accept ≥ 0.80) filters interaction partners, with a soft probabilistic region between the thresholds.

**Staking:** Before a staked interaction is created, both parties must commit active TNFTs as collateral. The required stake scales with perceived risk (derived from the partner's trust score and review history). On failure, all committed tokens are burned immediately — making defection economically costly. On success, tokens are unlocked and the trustee receives a new minted TNFT.

**Reviewer credibility:** Agents with a negative review rate above 0.60 (computed once at least 8 reviews are available) are flagged as unreliable reviewers and their evidence is filtered.

### Scenarios and Configurations

`profiles.py` defines three named scenarios that can be selected in the dashboard or notebooks:

| Scenario | Honest agents | Malicious agents | Attack probabilities |
|----------|--------------|-----------------|----------------------|
| `default` | 3 | 2 | map=0.5, neg-review=0.5, pos-review=0.5 |
| `honest_only` | 5 | 0 | — |
| `aggressive_malicious` | 3 | 2 | all=1.0 |

The evaluation notebook uses a larger population (14 honest, 6 malicious) on the 150×150 grid. The scenarios above are for the interactive dashboard.

The four **ablation configurations** evaluated in the thesis:

| Configuration | Trust | Staking | Purpose |
|---------------|-------|---------|---------|
| Honest-only | Yes | No | Upper bound |
| Accept-all | No | No | Lower bound |
| BINT baseline | Yes | No | Trust filtering only |
| BINT | Yes | Yes | Full mechanism |

---

## Key Parameters

All defaults are defined in `profiles.py`. The most important ones:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trust_reject_threshold` | 0.30 | Trust score below which interactions are always rejected |
| `trust_accept_threshold` | 0.80 | Trust score above which interactions are always accepted |
| `genesis_tokens` | 5 | Initial TNFTs per agent (cold-start mitigation) |
| `staking_min_fraction` | 0.10 | Minimum stake as a fraction of available tokens |
| `staking_max_fraction` | 0.90 | Maximum stake fraction (scales with risk) |
| `false_map_probability` | 0.50 | Probability a malicious agent provides a false coordinate |
| `false_negative_review_probability` | 0.50 | Probability a malicious requester falsely reports failure |
| `false_positive_review_probability` | 0.50 | Probability a malicious requester falsely reports success |
| `max_negative_review_rate` | 0.60 | Reviewer credibility threshold |
| `min_reviews_before_reviewer_check` | 5 | Reviews required before the credibility check activates |

---

## Results Summary

Results below are means over 100 independent seeds (14 honest / 6 malicious agents, 5,000 steps, medium attack probabilities of 0.50).

| Configuration | Honest pts/agent | Malicious pts/agent | Failure rate | Mal. provider rate |
|--------------|-----------------|--------------------|--------------|--------------------|
| Honest-only | 692 | — | 0.000 | 0.000 |
| Accept-all | 541 | 558 | 0.416 | 0.490 |
| BINT baseline | 527 | 27 | 0.173 | 0.171 |
| BINT | 480 | −1 | 0.105 | 0.115 |

Trust filtering alone reduces the accepted malicious provider rate from ~49% to ~17% and cuts the failure rate by 58%. Adding staking drives malicious-agent average reward below zero, making the default defection strategy unprofitable in this scenario.

These results characterise the mechanism's behavior in this specific simulation setting. They are not general performance guarantees for all multi-agent or IoT environments.

---