#!/usr/bin/env python3
"""
bint_golden.py — golden-master / characterization test for the BINT simulation.

Purpose
-------
Lock the *exact* behaviour of the simulation in place so that any refactor of
``agents.py`` / ``model.py`` (for readability or otherwise) can be proven to
leave results byte-for-byte identical for the same seed(s).

How it works
------------
The model is deterministic given a seed, so we:

  1. Run a matrix of (scenario x seed) configurations chosen to exercise every
     major code path.
  2. At fixed step checkpoints, capture a COMPLETE fingerprint of the run:
       * full observable state (ledger, interactions, outcomes, per-agent state,
         model counters) -- the ground truth, not the lossy published metrics;
       * the RNG internal state -- catches any change in the order/count of
         random draws, even before it has changed observable state.
  3. Hash each checkpoint and save the full snapshots to a baseline directory.

After a refactor, re-run and compare. Identical hashes => behaviour unchanged.
On a mismatch, a precise field-level diff shows WHAT changed and at WHICH
checkpoint it first diverged.

Derived lookup caches (``_ledger_by_owner``, ``_trust_score_cache``, the per-agent
``cached_active_tnfts``, etc.) are EXCLUDED from the golden hash, because a
refactor may legitimately restructure or remove them without changing behaviour.
Their correctness is instead validated separately by ``--check-caches``, which
recomputes the underlying facts from primary state.

Usage
-----
    # 1. BEFORE touching anything -- record the baseline:
    python bint_golden.py capture

    # 2. After each change -- prove behaviour is unchanged:
    python bint_golden.py verify

    # Sanity: prove the model is deterministic (no baseline needed):
    python bint_golden.py selfcheck

    # Validate the caching layer against recomputed ground truth:
    python bint_golden.py verify --check-caches

Profiles:  --profile fast (default, seconds)  |  thorough (closer to real config)
Run capture and verify in the SAME environment (same machine + Python + mesa
version); cross-environment float identicality is not guaranteed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Determinism guard: pin the hash seed so set/dict iteration is identical      #
# across separate `capture` and `verify` invocations. Must happen before the   #
# interpreter does any string hashing we care about, so we re-exec once.        #
# --------------------------------------------------------------------------- #
def _pin_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])


# =========================================================================== #
# Scenario matrix                                                             #
# =========================================================================== #
# Common agent parameters (mirrors the notebook's FINAL_BASE_PARAMETERS subset
# that is forwarded to agents). World/seed/step parameters are added per profile.
_AGENT_BASE: dict[str, Any] = {
    "vision_radius": 2,
    "trust_reject_threshold": 0.30,
    "trust_accept_threshold": 0.80,
    "context_match_weight": 1.0,
    "other_context_weight": 0.25,
    "trust_prior_active": 1.0,
    "trust_prior_burned": 1.0,
    "burned_weight_multiplier": 1.0,
    "filter_untrusted_evidence": True,
    "max_negative_review_rate": 0.60,
    "min_reviews_before_reviewer_check": 8,
    "staking_min_fraction": 0.10,
    "staking_max_fraction": 0.90,
    "provider_stake_vtp_weight": 0.8,
    "provider_stake_reviewer_weight": 0.2,
    "requester_stake_vtp_weight": 0.4,
    "requester_stake_reviewer_weight": 0.6,
}
_MALICIOUS_BASE: dict[str, Any] = {
    "false_map_probability": 0.50,
    "false_negative_review_probability": 0.50,
    "false_positive_review_probability": 0.50,
}

# Each scenario is a set of overrides. Together they exercise:
#   - no-malicious / pure delivery + mint-on-success      (honest_only)
#   - always-accept + non-staked burn path                (accept_all)
#   - trust filtering + non-staked burn path              (bint_baseline)
#   - staking: lock / release / burn_interaction_stakes   (bint_staking)
#   - malicious map-fabrication branch only               (corner_isomap)
#   - malicious review-forgery branches only              (corner_isorev)
#   - both attacks at once                                (corner_worst)
SCENARIOS: dict[str, dict[str, Any]] = {
    "honest_only": dict(
        honest_agents=8, malicious_agents=0, staking_enabled=False,
    ),
    "accept_all": dict(
        honest_agents=6, malicious_agents=3, staking_enabled=False,
        trust_reject_threshold=0.0, trust_accept_threshold=0.01,
        max_negative_review_rate=1.0,
    ),
    "bint_baseline": dict(
        honest_agents=6, malicious_agents=3, staking_enabled=False,
    ),
    "bint_staking": dict(
        honest_agents=6, malicious_agents=3, staking_enabled=True,
    ),
    "corner_isomap": dict(
        honest_agents=6, malicious_agents=3, staking_enabled=True,
        false_map_probability=1.0,
        false_negative_review_probability=0.0, false_positive_review_probability=0.0,
    ),
    "corner_isorev": dict(
        honest_agents=6, malicious_agents=3, staking_enabled=True,
        false_map_probability=0.0,
        false_negative_review_probability=1.0, false_positive_review_probability=1.0,
    ),
    "corner_worst": dict(
        honest_agents=6, malicious_agents=3, staking_enabled=True,
        false_map_probability=1.0,
        false_negative_review_probability=1.0, false_positive_review_probability=1.0,
    ),
}

PROFILES: dict[str, dict[str, Any]] = {
    # Small, dense world so interactions happen quickly; runs in seconds.
    "fast": dict(
        width=40, height=40, num_drop_offs=10, genesis_tokens=5,
        max_steps=600, checkpoint_every=100,
        seeds=[1337, 2024, 7],
    ),
    # Larger and longer; still far cheaper than the full notebook sweep.
    "thorough": dict(
        width=80, height=80, num_drop_offs=25, genesis_tokens=5,
        max_steps=2000, checkpoint_every=250,
        seeds=[1337, 2024, 7, 42, 99991],
    ),
}


def _resolve_config(scenario_name: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Merge agent base + scenario overrides + world params into one flat dict."""
    cfg: dict[str, Any] = {}
    cfg.update(_AGENT_BASE)
    cfg.update(_MALICIOUS_BASE)
    cfg.update(SCENARIOS[scenario_name])
    for k in ("width", "height", "num_drop_offs", "genesis_tokens",
              "max_steps", "checkpoint_every"):
        cfg[k] = profile[k]
    return cfg


def _agent_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: cfg[k] for k in _AGENT_BASE}


def _malicious_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    kw = _agent_kwargs(cfg)
    kw.update({k: cfg[k] for k in _MALICIOUS_BASE})
    return kw


def build_model(cfg: dict[str, Any], seed: int):
    """Construct a BintWorldModel for one scenario+seed. Imports lazily so the
    rest of the file (and --help) works even without mesa installed."""
    from agents import DeliveryAgent, MaliciousDeliveryAgent
    from model import AgentProfile, BintWorldModel

    profiles = []
    if int(cfg["honest_agents"]) > 0:
        profiles.append(AgentProfile(
            agent_class=DeliveryAgent,
            count=int(cfg["honest_agents"]),
            kwargs=_agent_kwargs(cfg),
        ))
    if int(cfg["malicious_agents"]) > 0:
        profiles.append(AgentProfile(
            agent_class=MaliciousDeliveryAgent,
            count=int(cfg["malicious_agents"]),
            kwargs=_malicious_kwargs(cfg),
        ))

    model = BintWorldModel(
        rng=seed,
        width=int(cfg["width"]),
        height=int(cfg["height"]),
        num_drop_offs=int(cfg["num_drop_offs"]),
        genesis_tokens=int(cfg["genesis_tokens"]),
        max_steps=int(cfg["max_steps"]),
        staking_enabled=bool(cfg["staking_enabled"]),
        agent_profiles=profiles,
    )
    model.DEBUG_DECISIONS = False
    return model


# =========================================================================== #
# Snapshotting -- turn the live model into a JSON-able ground-truth record     #
# =========================================================================== #
NAN = "__nan__"
POS_INF = "__inf__"
NEG_INF = "__-inf__"


def _num(x: Any) -> Any:
    """Make floats JSON-safe and stable (NaN/inf -> sentinels)."""
    if isinstance(x, float):
        if math.isnan(x):
            return NAN
        if x == math.inf:
            return POS_INF
        if x == -math.inf:
            return NEG_INF
    return x


def _coord(c: Any) -> Any:
    return list(c) if isinstance(c, (tuple, list)) else c


def _jsonable_meta(meta: Any) -> Any:
    """Recursively convert an arbitrary metadata blob to JSON-able primitives."""
    if isinstance(meta, dict):
        return {str(k): _jsonable_meta(v) for k, v in meta.items()}
    if isinstance(meta, (list, tuple)):
        return [_jsonable_meta(v) for v in meta]
    if isinstance(meta, float):
        return _num(meta)
    if isinstance(meta, (str, int, bool)) or meta is None:
        return meta
    return repr(meta)  # last resort, keeps it deterministic


def _tnft_record(t: dict[str, Any]) -> dict[str, Any]:
    """Canonical view of one ledger token. Note: ``staked_for`` IS behavioural
    (it gates collateral reuse), so it is kept; pure index membership is not."""
    return {
        "id": t.get("id"),
        "issuer": t.get("issuer"),
        "owner": t.get("owner"),
        "type": t.get("type"),
        "service_type": t.get("service_type"),
        "interaction_id": t.get("interaction_id"),
        "status": bool(t.get("status")),
        "timestamp": _num(t.get("timestamp")),
        "staked_for": t.get("staked_for"),
        "stake_role": t.get("stake_role"),
        "stake_service_type": t.get("stake_service_type"),
        "burned_by": t.get("burned_by"),
        "burn_reason": t.get("burn_reason"),
        "burn_timestamp": _num(t.get("burn_timestamp")),
        "metadata": _jsonable_meta(t.get("metadata", {})),
    }


def _agent_state(agent: Any) -> dict[str, Any]:
    """Behavioural state of one delivery agent. Maps are emitted as SORTED lists
    so that a harmless change in perception/iteration order (same contents,
    different insertion order) does not register as a difference."""
    internal_map = sorted(
        [[_coord(coord), rec.get("type"), rec.get("source")]
         for coord, rec in agent.internal_map.items()]
    )
    known = sorted([[name, _coord(c)] for name, c in agent.known_drop_offs.items()])
    candidate = sorted([[name, _coord(c)] for name, c in agent.candidate_drop_offs.items()])
    pkg = agent.package
    package = None if pkg is None else {
        "destination": pkg.get("destination"),
        "max_steps": pkg.get("max_steps"),
        "min_steps": pkg.get("min_steps"),
        "steps_taken": pkg.get("steps_taken"),
    }
    return {
        "type": type(agent).__name__,
        "coordinate": _coord(agent.cell.coordinate),
        "points": _num(float(agent.points)),
        "delivery_count": int(agent.delivery_count),
        "state": agent.state,
        "goal_name": agent.goal_name,
        "prev_goal_name": agent.prev_goal_name,
        "target_coordinate": _coord(agent.target_coordinate),
        "current_provider_id": agent.current_provider_id,
        "current_interaction_id": agent.current_interaction_id,
        "package": package,
        "internal_map": internal_map,
        "known_drop_offs": known,
        "candidate_drop_offs": candidate,
        "last_decision_type": agent.last_decision_type,
        "last_decision_reason": agent.last_decision_reason,
        "last_decision_peer_id": agent.last_decision_peer_id,
        "last_decision_service_type": agent.last_decision_service_type,
        "last_checked_trust_score": _num(agent.last_checked_trust_score)
            if agent.last_checked_trust_score is not None else None,
        "last_checked_negative_review_rate": _num(agent.last_checked_negative_review_rate)
            if agent.last_checked_negative_review_rate is not None else None,
    }


def _rng_state(model: Any) -> dict[str, Any]:
    """Capture the RNG internal state(s). Identical state after N steps proves
    the exact same sequence of random draws occurred in the same order."""
    out: dict[str, Any] = {}
    rnd = getattr(model, "random", None)
    if rnd is not None and hasattr(rnd, "getstate"):
        st = rnd.getstate()
        # st = (version, tuple_of_ints, gauss_next_or_None)
        out["stdlib"] = [st[0], list(st[1]), st[2]]
    np_rng = getattr(model, "rng", None)
    if np_rng is not None:
        try:
            bg = np_rng.bit_generator.state  # dict of ints
            out["numpy"] = json.loads(json.dumps(bg, default=lambda o: int(o)))
        except Exception:
            pass
    return out


def snapshot(model: Any, *, step: int, with_rng: bool = True) -> dict[str, Any]:
    """Build the complete ground-truth fingerprint of the model at this step."""
    delivery_agents = sorted(model.cached_delivery_agents, key=lambda a: str(a.unique_id))
    drop_offs = sorted(model.cached_drop_offs, key=lambda a: str(a.unique_id))

    interactions = {
        str(iid): {
            "truster_id": rec.truster_id,
            "trustee_id": rec.trustee_id,
            "service_type": rec.service_type,
            "status": rec.status,
            "timestamp": _num(rec.timestamp),
            "meta": _jsonable_meta(rec.meta),
        }
        for iid, rec in model.interactions.items()
    }
    outcomes = {
        str(iid): {
            "status": rec.status,
            "timestamp": _num(rec.timestamp),
            "meta": _jsonable_meta(rec.meta),
        }
        for iid, rec in model.outcomes.items()
    }

    snap: dict[str, Any] = {
        "step": step,
        "model": {
            "time": _num(getattr(model, "time", None)),
            "steps": getattr(model, "steps", None),
            "nft_counter": model.nft_counter,
            "interaction_counter": model.interaction_counter,
            "n_success": getattr(model, "_n_success", None),
            "n_failure": getattr(model, "_n_failure", None),
            "n_false_reviews": getattr(model, "_n_false_reviews", None),
            "decision_counts": dict(sorted(getattr(model, "_decision_counts", {}).items())),
        },
        "ledger": [_tnft_record(t) for t in model.tnft_ledger],
        "interactions": dict(sorted(interactions.items())),
        "outcomes": dict(sorted(outcomes.items())),
        "agents": {str(a.unique_id): _agent_state(a) for a in delivery_agents},
        "drop_offs": {str(a.unique_id): _coord(a.cell.coordinate) for a in drop_offs},
    }
    if with_rng:
        snap["rng_state"] = _rng_state(model)
    return snap


# =========================================================================== #
# Canonicalisation, hashing, diffing                                          #
# =========================================================================== #
def _canon(obj: Any, ndigits: int | None) -> Any:
    """Walk a JSON-able structure, stringifying floats for a stable hash.
    ``ndigits`` rounds floats first (None => exact)."""
    if isinstance(obj, float):
        if ndigits is not None:
            obj = round(obj, ndigits)
        return "f:" + repr(obj)
    if isinstance(obj, dict):
        return {k: _canon(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v, ndigits) for v in obj]
    return obj


def hash_snapshot(snap: dict[str, Any], ndigits: int | None) -> str:
    payload = json.dumps(_canon(snap, ndigits), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _floats_close(a: float, b: float, tol: float) -> bool:
    if isinstance(a, str) or isinstance(b, str):  # NaN/inf sentinels
        return a == b
    return abs(a - b) <= tol + tol * max(abs(a), abs(b))


def diff(golden: Any, current: Any, *, tol: float, path: str = "",
         out: list[tuple[str, Any, Any]] | None = None, limit: int = 200
         ) -> list[tuple[str, Any, Any]]:
    """Recursive field-level diff. Returns list of (path, golden, current)."""
    if out is None:
        out = []
    if len(out) >= limit:
        return out

    if isinstance(golden, dict) and isinstance(current, dict):
        for k in sorted(set(golden) | set(current)):
            if k not in golden:
                out.append((f"{path}.{k}", "<missing>", current[k]))
            elif k not in current:
                out.append((f"{path}.{k}", golden[k], "<missing>"))
            else:
                diff(golden[k], current[k], tol=tol, path=f"{path}.{k}", out=out, limit=limit)
    elif isinstance(golden, list) and isinstance(current, list):
        if len(golden) != len(current):
            out.append((f"{path}[len]", len(golden), len(current)))
        for i in range(min(len(golden), len(current))):
            diff(golden[i], current[i], tol=tol, path=f"{path}[{i}]", out=out, limit=limit)
    elif isinstance(golden, (int, float)) and isinstance(current, (int, float)) \
            and not isinstance(golden, bool) and not isinstance(current, bool):
        if not _floats_close(float(golden), float(current), tol):
            out.append((path, golden, current))
    else:
        if golden != current:
            out.append((path, golden, current))
    return out


# =========================================================================== #
# Cache-consistency check (validates the optimisation layer vs ground truth)  #
# =========================================================================== #
def check_caches(model: Any) -> list[str]:
    """Recompute derived facts from PRIMARY state and compare to the cached
    values the model maintains. Returns a list of human-readable problems."""
    problems: list[str] = []

    # Per-owner active/burned counts vs agent.cached_active_tnfts/cached_burned_tnfts.
    active: dict[Any, int] = {}
    burned: dict[Any, int] = {}
    for t in model.tnft_ledger:
        owner = t.get("owner")
        if t.get("status"):
            active[owner] = active.get(owner, 0) + 1
        else:
            burned[owner] = burned.get(owner, 0) + 1
    for a in model.cached_delivery_agents:
        want_a, want_b = active.get(a.unique_id, 0), burned.get(a.unique_id, 0)
        got_a = getattr(a, "cached_active_tnfts", None)
        got_b = getattr(a, "cached_burned_tnfts", None)
        if got_a is not None and got_a != want_a:
            problems.append(f"agent {a.unique_id}: cached_active_tnfts={got_a} but ledger says {want_a}")
        if got_b is not None and got_b != want_b:
            problems.append(f"agent {a.unique_id}: cached_burned_tnfts={got_b} but ledger says {want_b}")

    # Outcome counters vs recomputed-from-outcomes.
    n_succ = sum(1 for o in model.outcomes.values() if o.status == "success")
    n_fail = sum(1 for o in model.outcomes.values() if o.status == "failure")
    n_false = sum(1 for o in model.outcomes.values() if (o.meta or {}).get("review_was_false"))
    if getattr(model, "_n_success", n_succ) != n_succ:
        problems.append(f"_n_success={model._n_success} but outcomes have {n_succ}")
    if getattr(model, "_n_failure", n_fail) != n_fail:
        problems.append(f"_n_failure={model._n_failure} but outcomes have {n_fail}")
    if getattr(model, "_n_false_reviews", n_false) != n_false:
        problems.append(f"_n_false_reviews={model._n_false_reviews} but outcomes have {n_false}")

    # _ledger_by_owner index (if present) must reference the same token objects.
    idx = getattr(model, "_ledger_by_owner", None)
    if idx is not None:
        for t in model.tnft_ledger:
            bucket = idx.get(t.get("owner"), [])
            if not any(x is t for x in bucket):
                problems.append(f"_ledger_by_owner missing token id={t.get('id')} for owner {t.get('owner')}")
                break

    return problems


# =========================================================================== #
# Run orchestration                                                           #
# =========================================================================== #
def run_one(scenario_name: str, seed: int, profile: dict[str, Any],
            *, do_cache_check: bool = False
            ) -> tuple[list[dict[str, Any]], list[str]]:
    """Run a single scenario+seed, returning a snapshot per checkpoint and any
    cache-consistency problems found at the final step."""
    cfg = _resolve_config(scenario_name, profile)
    model = build_model(cfg, seed)
    max_steps = int(cfg["max_steps"])
    every = int(cfg["checkpoint_every"])

    snaps = [snapshot(model, step=0)]
    for step in range(1, max_steps + 1):
        model.step()
        if step % every == 0 or step == max_steps:
            snaps.append(snapshot(model, step=step))

    cache_problems = check_caches(model) if do_cache_check else []
    return snaps, cache_problems


def iter_matrix(profile: dict[str, Any]):
    for scenario_name in SCENARIOS:
        for seed in profile["seeds"]:
            yield scenario_name, seed


def _key(scenario_name: str, seed: int) -> str:
    return f"{scenario_name}__seed{seed}"


# =========================================================================== #
# Commands                                                                    #
# =========================================================================== #
def cmd_capture(args) -> int:
    profile = PROFILES[args.profile]
    out_dir = Path(args.out)
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "format_version": 1,
        "profile": args.profile,
        "round_ndigits": args.round,
        "python": sys.version.split()[0],
        "scenarios": list(SCENARIOS),
        "seeds": profile["seeds"],
        "hashes": {},  # key -> [per-checkpoint hashes]
        "steps": {},   # key -> [checkpoint step numbers]
    }
    try:
        import mesa
        manifest["mesa"] = mesa.__version__
    except Exception:
        manifest["mesa"] = "unknown"

    total = len(SCENARIOS) * len(profile["seeds"])
    done = 0
    for scenario_name, seed in iter_matrix(profile):
        snaps, _ = run_one(scenario_name, seed, profile)
        key = _key(scenario_name, seed)
        (snap_dir / f"{key}.json").write_text(json.dumps(snaps, indent=1))
        manifest["hashes"][key] = [hash_snapshot(s, args.round) for s in snaps]
        manifest["steps"][key] = [s["step"] for s in snaps]
        done += 1
        print(f"  [{done}/{total}] captured {key} "
              f"({len(snaps)} checkpoints, final hash {manifest['hashes'][key][-1][:12]})")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nBaseline written to {out_dir}/ "
          f"({total} runs, profile={args.profile}, rounding={args.round}).")
    print("Commit this directory to version control as your reference.")
    return 0


def cmd_verify(args) -> int:
    out_dir = Path(args.baseline)
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no baseline at {manifest_path}. Run `capture` first.", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())

    profile_name = manifest.get("profile", args.profile)
    profile = PROFILES[profile_name]
    ndigits = manifest.get("round_ndigits", args.round)

    if manifest.get("mesa") not in (None, "unknown"):
        try:
            import mesa
            if mesa.__version__ != manifest["mesa"]:
                print(f"WARNING: mesa version differs from baseline "
                      f"({mesa.__version__} vs {manifest['mesa']}). "
                      f"Float-level differences may be environmental, not behavioural.")
        except Exception:
            pass

    total = len(SCENARIOS) * len(profile["seeds"])
    done = 0
    failures: list[str] = []
    cache_issue_runs: list[str] = []

    for scenario_name, seed in iter_matrix(profile):
        key = _key(scenario_name, seed)
        snaps, cache_problems = run_one(
            scenario_name, seed, profile, do_cache_check=args.check_caches)
        done += 1

        golden_hashes = manifest["hashes"].get(key)
        if golden_hashes is None:
            failures.append(f"{key}: present now but ABSENT from baseline")
            print(f"  [{done}/{total}] {key}: NOT IN BASELINE")
            continue

        cur_hashes = [hash_snapshot(s, ndigits) for s in snaps]
        if cur_hashes == golden_hashes:
            status = "OK"
        else:
            status = "MISMATCH"
            # Find the first diverging checkpoint and diff it in detail.
            first = next((i for i, (g, c) in enumerate(zip(golden_hashes, cur_hashes))
                          if g != c), None)
            if first is None:
                first = min(len(golden_hashes), len(cur_hashes))
            failures.append(f"{key}: first diverges at checkpoint #{first} "
                            f"(step {manifest['steps'][key][first] if first < len(manifest['steps'][key]) else '?'})")
            _report_divergence(out_dir, key, first, snaps, tol=args.tol)

        if cache_problems:
            cache_issue_runs.append(key)

        print(f"  [{done}/{total}] {key}: {status}"
              + (f"  [+{len(cache_problems)} cache issues]" if cache_problems else ""))

    print()
    if cache_issue_runs:
        print(f"CACHE CONSISTENCY problems in {len(cache_issue_runs)} run(s): "
              f"{', '.join(cache_issue_runs)}")
        # cache issues are reported but do not, by themselves, fail behavioural verify

    if failures:
        print("RESULT: BEHAVIOUR CHANGED")
        for f in failures:
            print("  - " + f)
        return 1
    print(f"RESULT: identical to baseline across all {total} runs"
          + (" (caches OK)" if args.check_caches and not cache_issue_runs else "") + ".")
    return 0


def _report_divergence(out_dir: Path, key: str, checkpoint_idx: int,
                       current_snaps: list[dict], *, tol: float) -> None:
    golden_path = out_dir / "snapshots" / f"{key}.json"
    if not golden_path.exists():
        return
    golden_snaps = json.loads(golden_path.read_text())
    if checkpoint_idx >= len(golden_snaps) or checkpoint_idx >= len(current_snaps):
        print(f"      (checkpoint count differs: baseline {len(golden_snaps)}, "
              f"now {len(current_snaps)})")
        return
    g, c = golden_snaps[checkpoint_idx], current_snaps[checkpoint_idx]
    diffs = diff(g, c, tol=tol)
    rng_only = diffs and all(p.startswith(".rng_state") for p, _, _ in diffs)
    print(f"      first {min(len(diffs), 12)} field diff(s) at step {g.get('step')}:")
    for p, gv, cv in diffs[:12]:
        gv_s, cv_s = _short(gv), _short(cv)
        print(f"        {p}:  baseline={gv_s}  now={cv_s}")
    if len(diffs) > 12:
        print(f"        ... and {len(diffs) - 12} more")
    if rng_only:
        print("      NOTE: only the RNG state differs here. The draw order/count "
              "changed; observable state will likely diverge at a later step.")


def _short(v: Any, n: int = 60) -> str:
    s = json.dumps(v) if not isinstance(v, str) else v
    return s if len(s) <= n else s[: n - 3] + "..."


def cmd_selfcheck(args) -> int:
    """Run every matrix cell twice in-process and confirm identical output.
    Proves the model is deterministic for a given seed -- the assumption the
    whole approach rests on. No baseline required."""
    profile = PROFILES[args.profile]
    total = len(SCENARIOS) * len(profile["seeds"])
    done = 0
    bad: list[str] = []
    for scenario_name, seed in iter_matrix(profile):
        a, _ = run_one(scenario_name, seed, profile)
        b, _ = run_one(scenario_name, seed, profile)
        ha = [hash_snapshot(s, args.round) for s in a]
        hb = [hash_snapshot(s, args.round) for s in b]
        done += 1
        ok = ha == hb
        if not ok:
            bad.append(_key(scenario_name, seed))
        print(f"  [{done}/{total}] {_key(scenario_name, seed)}: "
              f"{'deterministic' if ok else 'NON-DETERMINISTIC'}")
    print()
    if bad:
        print("RESULT: model is NON-DETERMINISTIC for: " + ", ".join(bad))
        print("Golden-master testing cannot work until this is resolved.")
        return 1
    print(f"RESULT: deterministic across all {total} runs. Safe to use as a baseline.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Golden-master regression test for BINT.")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("capture", "verify", "selfcheck"):
        sp = sub.add_parser(name)
        sp.add_argument("--profile", choices=list(PROFILES), default="fast")
        sp.add_argument("--round", type=int, default=None,
                        help="round floats to N decimals before hashing (default: exact)")
        if name == "capture":
            sp.add_argument("--out", default="bint_golden_baseline")
        if name == "verify":
            sp.add_argument("--baseline", default="bint_golden_baseline")
            sp.add_argument("--tol", type=float, default=0.0,
                            help="numeric tolerance when reporting diffs")
            sp.add_argument("--check-caches", action="store_true",
                            help="also validate the derived caches against ground truth")
        if name == "selfcheck":
            sp.add_argument("--tol", type=float, default=0.0)

    args = p.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # find agents.py / model.py

    if args.cmd == "capture":
        return cmd_capture(args)
    if args.cmd == "verify":
        return cmd_verify(args)
    if args.cmd == "selfcheck":
        return cmd_selfcheck(args)
    return 2


if __name__ == "__main__":
    _pin_hash_seed()
    raise SystemExit(main())
