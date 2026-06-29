"""Validate bint_golden.py machinery without mesa, using a stand-in model whose
attribute surface matches what snapshot()/check_caches() read from the real model."""
import copy
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bint_golden as bg


class FakeCell:
    def __init__(self, coord): self.coordinate = coord


class FakeAgent:
    def __init__(self, uid, kind, coord):
        self.unique_id = uid
        self.__class__.__name__  # noop
        self._kind = kind
        self.cell = FakeCell(coord)
        self.points = 12.5
        self.delivery_count = 2
        self.state = "DELIVERING"
        self.goal_name = "DropOffLocationAgent_3"
        self.prev_goal_name = "DropOffLocationAgent_1"
        self.target_coordinate = (4, 5)
        self.current_provider_id = "DeliveryAgent_2"
        self.current_interaction_id = "interaction_7"
        self.package = {"destination": "DropOffLocationAgent_3",
                        "max_steps": 20, "min_steps": 10, "steps_taken": 6}
        self.internal_map = {(0, 0): {"type": "floor", "source": "self"},
                             (4, 5): {"type": "drop_off", "source": "system"}}
        self.known_drop_offs = {"DropOffLocationAgent_3": (4, 5)}
        self.candidate_drop_offs = {}
        self.last_decision_type = "accept_map_response"
        self.last_decision_reason = "provider_accepted"
        self.last_decision_peer_id = "DeliveryAgent_2"
        self.last_decision_service_type = "map_data"
        self.last_checked_trust_score = 0.8333333333333334
        self.last_checked_negative_review_rate = 0.0
        self.cached_active_tnfts = 0  # filled by harness builder below
        self.cached_burned_tnfts = 0


# FakeAgent class name needs to look like a real agent type:
class DeliveryAgent(FakeAgent):
    pass


class Rec:
    def __init__(self, **kw): self.__dict__.update(kw)


class FakeModel:
    def __init__(self, seed=1337):
        self.random = random.Random(seed)
        # consume some randomness so getstate() is non-trivial
        for _ in range(50):
            self.random.random()
        self.rng = None
        self.time = 3.0
        self.steps = 3
        self.nft_counter = 4
        self.interaction_counter = 2
        self._n_success = 1
        self._n_failure = 1
        self._n_false_reviews = 0
        self._decision_counts = {"provider_accepted": 5, "requester_accepted": 3}
        self.tnft_ledger = [
            {"id": 1, "issuer": "SYSTEM", "owner": "DeliveryAgent_1", "type": "bootstrap",
             "service_type": "bootstrap", "interaction_id": None, "status": True,
             "timestamp": 0.0, "staked_for": None, "stake_role": None, "stake_service_type": None,
             "metadata": {"bootstrap": True}},
            {"id": 2, "issuer": "SYSTEM", "owner": "DeliveryAgent_2", "type": "bootstrap",
             "service_type": "bootstrap", "interaction_id": None, "status": True,
             "timestamp": 0.0, "staked_for": None, "stake_role": None, "stake_service_type": None,
             "metadata": {"bootstrap": True}},
            {"id": 3, "issuer": "DeliveryAgent_1", "owner": "DeliveryAgent_2", "type": "reward",
             "service_type": "map_data", "interaction_id": "interaction_1", "status": True,
             "timestamp": 1.0, "staked_for": None, "stake_role": None, "stake_service_type": None,
             "metadata": {}},
            {"id": 4, "issuer": "DeliveryAgent_2", "owner": "DeliveryAgent_1", "type": "reward",
             "service_type": "map_data", "interaction_id": "interaction_2", "status": False,
             "timestamp": 2.0, "staked_for": None, "stake_role": None, "stake_service_type": None,
             "burned_by": "DeliveryAgent_2", "burn_reason": "interaction_failure",
             "burn_timestamp": 2.0, "metadata": {}},
        ]
        self.interactions = {
            "interaction_1": Rec(truster_id="DeliveryAgent_2", trustee_id="DeliveryAgent_1",
                                 service_type="map_data", status="completed", timestamp=1.0, meta={}),
            "interaction_2": Rec(truster_id="DeliveryAgent_1", trustee_id="DeliveryAgent_2",
                                 service_type="map_data", status="completed", timestamp=2.0,
                                 meta={"review_was_false": False}),
        }
        self.outcomes = {
            "interaction_1": Rec(status="success", timestamp=1.0, meta={"actual_outcome_status": "success"}),
            "interaction_2": Rec(status="failure", timestamp=2.0, meta={"review_was_false": False}),
        }
        a1 = DeliveryAgent("DeliveryAgent_1", "honest", (1, 1))
        a2 = DeliveryAgent("DeliveryAgent_2", "honest", (4, 5))
        # set cached counts to match ledger truth
        a1.cached_active_tnfts, a1.cached_burned_tnfts = 1, 0  # owns id1 active, id4 burned -> 1 active,1 burned
        a1.cached_burned_tnfts = 1
        a2.cached_active_tnfts, a2.cached_burned_tnfts = 2, 0  # owns id2,id3 active
        self.cached_delivery_agents = [a1, a2]
        self.cached_drop_offs = [
            Rec(unique_id="DropOffLocationAgent_1", cell=FakeCell((9, 9))),
            Rec(unique_id="DropOffLocationAgent_3", cell=FakeCell((4, 5))),
        ]
        # build the _ledger_by_owner index referencing same objects
        self._ledger_by_owner = {}
        for t in self.tnft_ledger:
            self._ledger_by_owner.setdefault(t["owner"], []).append(t)


def main():
    passed, failed = 0, 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name}")

    # 1. snapshot is JSON-serialisable and stable
    m = FakeModel()
    s1 = bg.snapshot(m, step=3)
    import json
    json.dumps(s1)  # raises if not serialisable
    check("snapshot is JSON-serialisable", True)

    # 2. identical models -> identical hash
    m2 = FakeModel()
    h1 = bg.hash_snapshot(bg.snapshot(m, step=3), None)
    h2 = bg.hash_snapshot(bg.snapshot(m2, step=3), None)
    check("identical models hash equal", h1 == h2)

    # 3. a behavioural change is caught with a precise path
    m3 = FakeModel()
    m3.cached_delivery_agents[0].points = 99.0
    s3 = bg.snapshot(m3, step=3)
    check("changed points -> hash differs", bg.hash_snapshot(s3, None) != h1)
    d = bg.diff(s1, s3, tol=0.0)
    paths = [p for p, _, _ in d]
    check("diff pinpoints the changed field",
          any(p.endswith("points") and "DeliveryAgent_1" in p for p in paths))

    # 4. a ledger status flip is caught
    m4 = FakeModel()
    m4.tnft_ledger[2]["status"] = False
    d4 = bg.diff(s1, bg.snapshot(m4, step=3), tol=0.0)
    check("ledger token status flip is caught",
          any("ledger" in p and p.endswith("status") for p, _, _ in d4))

    # 5. RNG divergence is caught even with identical observable state
    m5 = FakeModel()
    m5.random.random()  # one extra draw -> different getstate()
    d5 = bg.diff(s1, bg.snapshot(m5, step=3), tol=0.0)
    check("extra RNG draw is caught via rng_state",
          d5 and all(p.startswith(".rng_state") for p, _, _ in d5))

    # 6. map insertion order does NOT matter (sorted canonicalisation)
    m6 = FakeModel()
    a = m6.cached_delivery_agents[0]
    a.internal_map = dict(reversed(list(a.internal_map.items())))
    check("reordered internal_map hashes the same",
          bg.hash_snapshot(bg.snapshot(m6, step=3), None)
          == bg.hash_snapshot(bg.snapshot(FakeModel(), step=3), None))

    # 7. float rounding knob
    m7 = FakeModel()
    m7.cached_delivery_agents[0].points = 12.5 + 1e-12
    h_exact = bg.hash_snapshot(bg.snapshot(m7, step=3), None)
    h_round = bg.hash_snapshot(bg.snapshot(m7, step=3), 6)
    check("tiny float delta differs under exact hashing", h_exact != h1)
    check("tiny float delta absorbed by rounding=6", h_round == bg.hash_snapshot(s1, 6))

    # 8. cache consistency: clean model reports no problems
    check("check_caches clean on consistent model", bg.check_caches(FakeModel()) == [])

    # 9. cache consistency: injected drift is reported
    m9 = FakeModel()
    m9.cached_delivery_agents[1].cached_active_tnfts = 999
    probs = bg.check_caches(m9)
    check("check_caches catches cached_active drift",
          any("cached_active_tnfts" in p for p in probs))

    # 10. cache consistency: counter drift is reported
    m10 = FakeModel()
    m10._n_success = 5
    check("check_caches catches _n_success drift",
          any("_n_success" in p for p in bg.check_caches(m10)))

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
