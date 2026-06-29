# BINT golden-master tests

The golden-master tests are characterization tests for the BINT simulation. They lock the **exact current behavior** of the model so that future refactors can be checked safely.

They answer this question:

> Did my code change alter the simulation behavior for the same seeds?

They do **not** answer this question:

> Is the model theoretically correct?

So golden tests are very useful for refactoring, cleanup, optimization, and dashboard work, but they are not a substitute for model validation.

---

## Files

| File/directory | Purpose |
|---|---|
| `bint_golden.py` | Golden-master harness; supports `selfcheck`, `capture`, and `verify` |
| `test_bint_golden.py` | Pytest wrapper around golden verification |
| `test_harness_machinery.py` | Unit tests for the harness machinery itself |
| `bint_golden_baseline/` | Saved baseline snapshots and manifest |

These files must stay in the same directory as `agents.py` and `model.py`.

---

## Everyday workflow

Before changing behavior-sensitive code, run:

```bash
PYTHONHASHSEED=0 python bint_golden.py verify --profile fast
PYTHONHASHSEED=0 python bint_golden.py verify --profile fast --check-caches
```

After your change, run the same commands again.

Expected result:

```text
RESULT: identical to baseline across all runs.
```

If verification fails, the script prints where the first divergence happened. That does not always mean your change is wrong; it means the simulation behavior changed and you need to inspect whether that change was intentional.

---

## Full recommended check sequence

From the repository root:

```bash
python -m compileall -q agents.py model.py profiles.py app.py compare_brs_bint.py bint_golden.py
python -m pytest -q test_brs.py test_harness_machinery.py
PYTHONHASHSEED=0 python -m pytest -q
PYTHONHASHSEED=0 python bint_golden.py verify --profile fast
PYTHONHASHSEED=0 python bint_golden.py verify --profile fast --check-caches
```

Before handing off larger changes, also run:

```bash
PYTHONHASHSEED=0 python bint_golden.py verify --profile thorough
```

The thorough profile is slower but covers larger worlds and more seeds.

---

## Commands

### 1. Self-check determinism

```bash
PYTHONHASHSEED=0 python bint_golden.py selfcheck
```

This confirms that the model is deterministic for a fixed seed. If this fails, golden testing is not reliable until determinism is restored.

### 2. Verify against existing baseline

```bash
PYTHONHASHSEED=0 python bint_golden.py verify --profile fast
```

Use this after refactors or small changes.

### 3. Verify cache correctness too

```bash
PYTHONHASHSEED=0 python bint_golden.py verify --profile fast --check-caches
```

The golden hash intentionally excludes derived lookup caches, because cache layout can change during refactors. `--check-caches` recomputes cache facts from primary state and checks that the caches are still consistent.

### 4. Run the larger profile

```bash
PYTHONHASHSEED=0 python bint_golden.py verify --profile thorough
```

Use this before bigger commits or before handing the repository to someone else.

### 5. Capture a new baseline

```bash
PYTHONHASHSEED=0 python bint_golden.py capture
```

Only do this when a behavior change is intentional and has been reviewed.

After capturing, commit the new baseline:

```bash
git add bint_golden_baseline
git commit -m "Update golden baseline"
```

Do not use `capture` just to make a failing golden test pass.

---

## What the baseline captures

At fixed checkpoints, the harness stores and hashes the observable simulation state, including:

- TNFT ledger records
- interactions
- outcomes
- agent positions
- agent points and delivery state
- known and candidate drop-off knowledge
- active package state
- model counters
- RNG internal state

The RNG state is included because it catches changes in the number or order of random draws, even before those changes visibly affect the model output.

---

## What is excluded from the golden hash

Derived caches and indexes are excluded, for example:

- ledger-by-owner indexes
- trust-score caches
- trust-evidence caches
- cached active/burned token counts where they can be recomputed

This is intentional. A refactor should be allowed to reorganize caches without changing model behavior.

Use `--check-caches` when you want to verify that cache contents still match the primary ledger and record state.

---

## Profiles

| Profile | Use case |
|---|---|
| `fast` | Everyday checks; small dense worlds; runs quickly |
| `thorough` | More paranoid checks; larger worlds and more seeds |

The profiles are defined near the top of `bint_golden.py`.

The scenario matrix currently covers:

| Scenario | What it exercises |
|---|---|
| `honest_only` | Honest delivery, success path, minting behavior |
| `accept_all` | Trust disabled/lower-bound behavior, non-staked burn path |
| `bint_baseline` | Trust filtering without staking |
| `bint_staking` | Stake locking, release, and burn paths |
| `corner_isomap` | Isolated map-fabrication attack |
| `corner_isorev` | Isolated dishonest-review attack |
| `corner_worst` | Map and review attacks together |

If a future developer adds a major mechanism, they should consider extending the scenario matrix so the new path is covered.

---

## Important caveats

- Golden tests are **behavior preservation tests**, not correctness proofs.
- They lock in the current behavior, including any existing bugs.
- Run capture and verify in the same environment when possible: same Python version, Mesa version, and dependency set.
- Always use `PYTHONHASHSEED=0` when running through pytest. The script pins it automatically when run directly, but setting it explicitly avoids confusion.
- If the model intentionally changes, inspect the diff before recapturing the baseline.

---

## Recommended rule of thumb

Use this command often:

```bash
PYTHONHASHSEED=0 python bint_golden.py verify --profile fast --check-caches
```

Use this command rarely and carefully:

```bash
PYTHONHASHSEED=0 python bint_golden.py capture
```
