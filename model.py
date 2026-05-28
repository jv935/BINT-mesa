import mesa
from mesa.discrete_space import OrthogonalMooreGrid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, Any
from agents import (
    DeliveryAgent,
    DropOffLocationAgent,
    MaliciousDeliveryAgent,
    ENV_DROP_OFF,
    SOURCE_SYSTEM,
)

Coordinate = tuple[int, int]
InteractionStatus = Literal["pending", "completed", "cancelled"]
OutcomeStatus = Literal["success", "failure", "disputed"]

MAP_DATA_SERVICE = "map_data"
SYSTEM_ISSUER_ID = "SYSTEM"

BOOTSTRAP_SERVICE = "bootstrap"

BASE_DELIVERY_POINTS = 10.0
GRACE_WINDOW_RATIO = 0.5
LATE_PENALTY_PER_STEP = 0.5


@dataclass
class InteractionRecord:
    interaction_id: str
    truster_id: str
    trustee_id: str
    service_type: str
    meta: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    status: InteractionStatus = "pending"


@dataclass
class OutcomeRecord:
    interaction_id: str
    status: OutcomeStatus
    meta: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class AgentProfile:
    agent_class: type[DeliveryAgent]
    count: int
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.count = int(self.count)

        if self.count < 0:
            raise ValueError("AgentProfile: count must be greater than or equal to 0.")

        if not issubclass(self.agent_class, DeliveryAgent):
            raise TypeError(
                "AgentProfile: agent_class must be a DeliveryAgent subclass."
            )

        if not isinstance(self.kwargs, dict):
            raise TypeError("AgentProfile: kwargs must be a dictionary.")


class BintWorldModel(mesa.Model):
    """Mesa simulation for BINT."""

    def __init__(
        self,
        num_drop_offs: int = 5,
        agent_profiles: list[AgentProfile] | None = None,
        size: Coordinate | None = None,
        width: int = 100,
        height: int = 100,
        genesis_tokens: int = 1,
        max_steps: int = 1500,
        staking_enabled: bool = True,
        rng: int | str | None = None,
    ) -> None:

        self.rng_seed = self._normalise_rng_seed(rng)
        super().__init__(rng=self.rng_seed)

        self.size = size if size is not None else (width, height)
        self.width, self.height = self.size
        self.max_steps = int(max_steps)

        self.num_drop_offs = int(num_drop_offs)
        self.genesis_tokens = int(genesis_tokens)
        self.staking_enabled = bool(staking_enabled)

        # use defaults if none are provided
        if agent_profiles is None:
            self.agent_profiles = [
                AgentProfile(DeliveryAgent, 7),
                AgentProfile(MaliciousDeliveryAgent, 3),
            ]
        else:
            self.agent_profiles = agent_profiles

        self.total_delivery_agents = sum(
            profile.count for profile in self.agent_profiles
        )

        self._create_grid()
        self._initialise_ledger_and_records()
        self._spawn_agents()
        self._cache_agents()

        self.distribute_initial_knowledge()
        self.dispatch_packages()
        self.seed_genesis_tnfts()

    @staticmethod
    def _normalise_rng_seed(rng: int | str | None) -> int | None:
        if rng is None or rng == "":
            return None
        return int(rng)

    def _create_grid(self) -> None:
        self.grid = OrthogonalMooreGrid(self.size, torus=False, random=self.random)

        all_cells = list(self.grid.all_cells.cells)
        required_cells = self.num_drop_offs + self.total_delivery_agents
        selected_cells = self.random.sample(all_cells, k=required_cells)

        self.drop_off_cells = selected_cells[: self.num_drop_offs]
        self.agent_spawn_cells = selected_cells[self.num_drop_offs :]
        self.all_coordinates = frozenset(cell.coordinate for cell in all_cells)

    def _initialise_ledger_and_records(self) -> None:
        self.tnft_ledger: list[dict[str, Any]] = []
        self.nft_counter = 0
        self.interaction_counter = 0
        self.interactions: dict[str, InteractionRecord] = {}
        self.outcomes: dict[str, OutcomeRecord] = {}
        self.decision_events: list[dict[str, Any]] = []

        # Fix A: per-owner index. Same dict objects as tnft_ledger, so in-place
        # mutations (burn, stake, release) are automatically reflected here.
        # Only needs updating on mint (when a new object is created).
        self._ledger_by_owner: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

        # Fix B: per-interaction stake index, populated in _lock_stake,
        # consumed and popped in release_interaction_stakes / burn_interaction_stakes.
        self._staked_by_interaction: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

        # Fix C: O(1) outcome counters replacing O(n) outcome-dict iteration.
        self._n_success: int = 0
        self._n_failure: int = 0
        self._n_false_reviews: int = 0

        # Perf D: reviewer summary cache — invalidated in record_outcome.
        # get_reviewer_summary is called on every map request (O(interactions)
        # per call); caching reduces it to O(1) for repeated reads.
        self._reviewer_cache: dict[str, dict] = {}

        # Perf E: agent lookup index — O(1) instead of O(n) linear scan.
        # Populated in _cache_agents after all agents are spawned.
        self._agent_index: dict[str, 'DeliveryAgent'] = {}

        # Perf F: decision reason Counter — replaces the full event list
        # for the purpose of reason counting (list kept for optional debug).
        self._decision_counts: dict[str, int] = {}

        # Perf G: trust evidence cache — keyed by (agent_id, service_type).
        # get_trust_evidence is called on every trust evaluation; caching
        # avoids repeated list comprehensions over _ledger_by_owner.
        # Invalidated in mint_tnft (new token) and burn_tnft /
        # burn_interaction_stakes (status change). Lock/release stake does
        # NOT change active/burned status, so no invalidation needed there.
        self._trust_evidence_cache: dict[tuple, dict] = {}

        # Perf H: drop-off coordinate index — O(1) instead of O(n) scan.
        # Populated in _cache_agents after drop-off agents are spawned.
        self._dropoff_index: dict[str, 'DropOffLocationAgent'] = {}

        # Perf J: trust SCORE cache.
        # Key: (evaluator_id, target_id, service_type, filter_flag)
        # This layers on top of Perf G (evidence cache) and additionally
        # avoids repeating _filter_untrusted_trust_evidence + recursive
        # issuer lookups for the same (evaluator, target) pair.
        #
        # Invalidation is TARGETED and uses two monotonic indexes:
        #   _tokens_by_issuer[X]  = agents that hold tokens ISSUED BY X
        #   _tokens_burned_by[X]  = agents that had a token BURNED BY X
        # When X's token state changes, we invalidate:
        #   - all entries where target == X (direct: X's score changed)
        #   - filter=True entries where target in _tokens_by_issuer[X]
        #     (issuer X's score is re-checked in _filter_untrusted_trust_evidence)
        #   - filter=True entries where target in _tokens_burned_by[X]
        #     (burn evidence is filtered by burned_by identity)
        # filter=False entries for targets ≠ X are always safe to keep.
        self._trust_score_cache: dict[tuple, dict] = {}
        self._tokens_by_issuer:  defaultdict[str, set] = defaultdict(set)
        self._tokens_burned_by:  defaultdict[str, set] = defaultdict(set)

    def _spawn_agents(self) -> None:
        spawn_index = 0

        for profile in self.agent_profiles:
            count = int(profile.count)

            if count <= 0:
                continue

            cells = self.agent_spawn_cells[spawn_index : spawn_index + count]

            profile.agent_class.create_agents(self, count, cells, **profile.kwargs)

            spawn_index += count

        DropOffLocationAgent.create_agents(
            self, self.num_drop_offs, self.drop_off_cells
        )

    def _cache_agents(self) -> None:
        self.cached_drop_offs = [
            agent for agent in self.agents if isinstance(agent, DropOffLocationAgent)
        ]
        self.cached_delivery_agents = [
            agent for agent in self.agents if isinstance(agent, DeliveryAgent)
        ]
        # Perf E: build O(1) lookup index after the agent list is finalised.
        self._agent_index = {a.unique_id: a for a in self.cached_delivery_agents}
        # Perf H: drop-off coordinate index.
        self._dropoff_index = {d.unique_id: d for d in self.cached_drop_offs}

    def record_interaction(
        self,
        truster_id: str,
        trustee_id: str,
        service_type: str,
        meta: dict[str, Any] | None = None,
    ) -> str:
        self.interaction_counter += 1
        interaction_id = f"interaction_{self.interaction_counter}"

        self.interactions[interaction_id] = InteractionRecord(
            interaction_id=interaction_id,
            truster_id=truster_id,
            trustee_id=trustee_id,
            service_type=service_type,
            meta=meta or {},
            timestamp=self.time,
            status="pending",
        )
        return interaction_id

    def record_staked_interaction(
        self,
        truster_id: str,
        trustee_id: str,
        service_type: str,
        truster_stake: int,
        trustee_stake: int,
        meta: dict[str, Any] | None = None,
    ) -> str | None:
        """Create an interaction and lock both agents' stake.

        Returns the interaction id if successful.
        Returns None if either side cannot provide the requested stake.
        """
        truster_stake = int(truster_stake)
        trustee_stake = int(trustee_stake)

        if truster_stake < 0 or trustee_stake < 0:
            raise ValueError("Stake amounts must be >= 0.")

        interaction_meta = dict(meta or {})
        interaction_meta.update(
            {
                "staking_enabled": self.staking_enabled,
                "truster_stake_requested": truster_stake,
                "trustee_stake_requested": trustee_stake,
                "truster_stake_ids": [],
                "trustee_stake_ids": [],
            }
        )

        interaction_id = self.record_interaction(
            truster_id=truster_id,
            trustee_id=trustee_id,
            service_type=service_type,
            meta=interaction_meta,
        )

        if not self.staking_enabled:
            return interaction_id

        truster_stake_ids = self._lock_stake(
            agent_id=truster_id,
            interaction_id=interaction_id,
            amount=truster_stake,
            role="truster",
            service_type=service_type,
        )

        if truster_stake_ids is None:
            self.interactions.pop(interaction_id, None)
            return None

        trustee_stake_ids = self._lock_stake(
            agent_id=trustee_id,
            interaction_id=interaction_id,
            amount=trustee_stake,
            role="trustee",
            service_type=service_type,
        )

        if trustee_stake_ids is None:
            self.release_interaction_stakes(interaction_id)
            self.interactions.pop(interaction_id, None)
            return None

        interaction = self.interactions[interaction_id]
        interaction.meta["truster_stake_ids"] = truster_stake_ids
        interaction.meta["trustee_stake_ids"] = trustee_stake_ids
        interaction.meta["truster_stake_locked"] = len(truster_stake_ids)
        interaction.meta["trustee_stake_locked"] = len(trustee_stake_ids)

        return interaction_id

    def get_interaction(self, interaction_id: str) -> InteractionRecord | None:
        return self.interactions.get(interaction_id)

    def record_outcome(
        self,
        interaction_id: str,
        status: OutcomeStatus,
        meta: dict[str, Any] | None = None,
    ) -> OutcomeRecord | None:
        if interaction_id not in self.interactions:
            return None

        outcome = OutcomeRecord(
            interaction_id=interaction_id,
            status=status,
            meta=meta or {},
            timestamp=self.time,
        )

        self.outcomes[interaction_id] = outcome

        # Fix C: maintain O(1)-readable outcome counters.
        if status == "success":
            self._n_success += 1
        elif status == "failure":
            self._n_failure += 1
        if (meta or {}).get("review_was_false"):
            self._n_false_reviews += 1

        # Perf D: invalidate reviewer cache for the truster of this interaction.
        # The truster is the agent who reported the outcome (the reviewer).
        interaction_record = self.interactions.get(interaction_id)
        if interaction_record is not None:
            self._reviewer_cache.pop(interaction_record.truster_id, None)

        return outcome

    def settle_interaction(
        self,
        interaction_id: str,
        evaluator_id: str,
        outcome_status: OutcomeStatus,
        outcome_meta: dict[str, Any] | None = None,
    ) -> OutcomeRecord | None:
        interaction = self.get_interaction(interaction_id)

        if interaction is None or interaction.status != "pending":
            return None

        outcome = self.record_outcome(
            interaction_id=interaction_id,
            status=outcome_status,
            meta=outcome_meta or {},
        )
        if outcome is None:
            return None

        self._apply_outcome_to_interaction(
            interaction=interaction,
            outcome=outcome,
            evaluator_id=evaluator_id,
        )
        return outcome

    def _apply_outcome_to_interaction(
        self, interaction: InteractionRecord, outcome: OutcomeRecord, evaluator_id: str
    ) -> None:
        if outcome.status == "success":
            self.release_interaction_stakes(interaction.interaction_id)
            self._reward_successful_interaction(interaction, outcome, evaluator_id)
            interaction.status = "completed"
            return

        if outcome.status == "failure":
            if self.staking_enabled:
                burned_count = self.burn_interaction_stakes(
                    interaction_id=interaction.interaction_id,
                    burner_id=evaluator_id,
                )
                interaction.status = "completed" if burned_count > 0 else "cancelled"
                return

            burned = self.burn_tnft(
                burner_id=evaluator_id,
                target_id=interaction.trustee_id,
                service_type=interaction.service_type,
            )
            interaction.status = "completed" if burned else "cancelled"
            return

        self.release_interaction_stakes(interaction.interaction_id)
        interaction.status = "cancelled"

    def _reward_successful_interaction(
        self, interaction: InteractionRecord, outcome: OutcomeRecord, evaluator_id: str
    ) -> None:
        self.mint_tnft(
            issuer_id=evaluator_id,
            receiver_id=interaction.trustee_id,
            interaction_type="reward",
            service_type=interaction.service_type,
            interaction_id=interaction.interaction_id,
            meta={
                "interaction_metadata": dict(interaction.meta),
                "outcome_metadata": dict(outcome.meta),
            },
        )

    def seed_genesis_tnfts(self) -> None:
        for agent in self.cached_delivery_agents:
            for _ in range(self.genesis_tokens):
                self.mint_tnft(
                    issuer_id=SYSTEM_ISSUER_ID,
                    receiver_id=agent.unique_id,
                    interaction_type="bootstrap",
                    service_type=BOOTSTRAP_SERVICE,
                    interaction_id=None,
                    meta={"bootstrap": True},
                )

    def mint_tnft(
        self,
        issuer_id: str,
        receiver_id: str,
        interaction_type: str,
        service_type: str,
        interaction_id: str | None,
        meta: dict[str, Any] | None = None,
    ) -> int:
        self.nft_counter += 1

        tnft = {
            "id": self.nft_counter,
            "issuer": issuer_id,
            "owner": receiver_id,
            "type": interaction_type,
            "service_type": service_type,
            "interaction_id": interaction_id,
            "metadata": meta or {},
            "status": True,  # True means active, False means burned
            "timestamp": self.time,
            "staked_for": None,
            "stake_role": None,
            "stake_service_type": None,
        }
        self.tnft_ledger.append(tnft)
        self._ledger_by_owner[receiver_id].append(tnft)   # Fix A
        # Perf J: update issuer index before invalidating.
        self._tokens_by_issuer[issuer_id].add(receiver_id)
        self._invalidate_trust_score_cache(receiver_id)
        # Perf G: invalidate trust evidence cache for receiver — new active token.
        # Pop all keys for this agent (any service_type variant).
        for _k in [k for k in self._trust_evidence_cache if k[0] == receiver_id]:
            del self._trust_evidence_cache[_k]

        # update cache
        target_agent = self._find_delivery_agent(receiver_id)
        if target_agent is not None:
            target_agent.cached_active_tnfts += 1

        return tnft["id"]

    def get_vtp(
        self,
        agent_id: str,
        service_type: str | None = None,
        active_only: bool = True,
        sort: bool = True,
    ) -> list[dict[str, Any]]:
        # Fix A: O(own_tokens) lookup via owner index instead of O(total_ledger) scan.
        tnfts = list(self._ledger_by_owner.get(agent_id, []))

        if active_only:
            tnfts = [nft for nft in tnfts if nft["status"]]

        if service_type is not None:
            tnfts = [nft for nft in tnfts if nft["service_type"] == service_type]

        # Fix F: sort is skipped in hot paths (get_trust_evidence) where order
        # does not affect the result; external callers keep sort=True by default.
        if sort:
            tnfts.sort(key=lambda tnft: tnft["timestamp"], reverse=True)
        return tnfts

    def get_trust_evidence(
        self,
        agent_id: str,
        service_type: str | None = None,
    ) -> dict[str, Any]:
        """Return raw ledger evidence for an agent.

        The model owns the TNFT ledger, so it is responsible for retrieving and
        grouping evidence. Agents are responsible for interpreting this evidence
        into a trust score.
        """
        # Perf G: check cache first. Key is (agent_id, service_type).
        _cache_key = (agent_id, service_type)
        if _cache_key in self._trust_evidence_cache:
            return self._trust_evidence_cache[_cache_key]

        # Single pass over owner tokens — partitions into active and burned
        # without calling get_vtp twice (avoids two separate list comprehensions).
        owner_tokens = self._ledger_by_owner.get(agent_id, [])
        active_tnfts = [t for t in owner_tokens if t["status"]]
        burned_tnfts = [t for t in owner_tokens if not t["status"]]

        if service_type is None:
            context_matching_active_tnfts = active_tnfts
            other_active_tnfts = []
            context_matching_burned_tnfts = burned_tnfts
            other_burned_tnfts = []
        else:
            context_matching_active_tnfts = [
                tnft for tnft in active_tnfts if tnft["service_type"] == service_type
            ]
            other_active_tnfts = [
                tnft for tnft in active_tnfts if tnft["service_type"] != service_type
            ]
            context_matching_burned_tnfts = [
                tnft for tnft in burned_tnfts if tnft["service_type"] == service_type
            ]
            other_burned_tnfts = [
                tnft for tnft in burned_tnfts if tnft["service_type"] != service_type
            ]

        _evidence = {
            "agent_id": agent_id,
            "service_type": service_type,
            "total_active": len(active_tnfts),
            "context_matching_active": len(context_matching_active_tnfts),
            "other_active": len(other_active_tnfts),
            "total_burned": len(burned_tnfts),
            "context_matching_burned": len(context_matching_burned_tnfts),
            "other_burned": len(other_burned_tnfts),
            "active_tnfts": active_tnfts,
            "burned_tnfts": burned_tnfts,
            "context_matching_active_tnfts": context_matching_active_tnfts,
            "other_active_tnfts": other_active_tnfts,
            "context_matching_burned_tnfts": context_matching_burned_tnfts,
            "other_burned_tnfts": other_burned_tnfts,
        }
        self._trust_evidence_cache[_cache_key] = _evidence
        return _evidence

    def get_available_stake_tnfts(
        self,
        agent_id: str,
        service_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return active, unstaked TNFTs that can be used as collateral.

        Staked TNFTs still count as active reputation, but they cannot be reused
        as collateral for another pending interaction.
        """
        # Fix A: use owner index — same filter logic, O(own_tokens) not O(ledger).
        available_tnfts = [
            tnft
            for tnft in self._ledger_by_owner.get(agent_id, [])
            if tnft["status"] and tnft.get("staked_for") is None
        ]

        if service_type is None:
            return sorted(available_tnfts, key=lambda tnft: tnft["id"])

        def stake_priority(tnft: dict[str, Any]) -> tuple[int, int]:
            if tnft["service_type"] == service_type:
                return (0, tnft["id"])
            if tnft["service_type"] == BOOTSTRAP_SERVICE:
                return (1, tnft["id"])
            return (2, tnft["id"])

        return sorted(available_tnfts, key=stake_priority)

    def available_stake_count(
        self,
        agent_id: str,
        service_type: str | None = None,
    ) -> int:
        return len(self.get_available_stake_tnfts(agent_id, service_type))

    def _lock_stake(
        self,
        agent_id: str,
        interaction_id: str,
        amount: int,
        role: str,
        service_type: str,
    ) -> list[int] | None:
        amount = int(amount)

        if amount < 0:
            raise ValueError("Stake amount must be >= 0.")

        if amount == 0:
            return []

        available_tnfts = self.get_available_stake_tnfts(agent_id, service_type)

        if len(available_tnfts) < amount:
            return None

        selected_tnfts = available_tnfts[:amount]

        for tnft in selected_tnfts:
            tnft["staked_for"] = interaction_id
            tnft["stake_role"] = role
            tnft["stake_service_type"] = service_type

        # Fix B: register staked tokens so release/burn can find them in O(staked_count).
        self._staked_by_interaction[interaction_id].extend(selected_tnfts)

        return [tnft["id"] for tnft in selected_tnfts]

    def release_interaction_stakes(self, interaction_id: str) -> None:
        """Release all active TNFTs staked for an interaction."""
        # Fix B: use stake index — O(staked_count) instead of O(total_ledger).
        # pop() also cleans up the index entry to prevent memory accumulation.
        for tnft in self._staked_by_interaction.pop(interaction_id, []):
            if not tnft["status"]:   # guard: burned between stake and release
                continue
            tnft["staked_for"] = None
            tnft["stake_role"] = None
            tnft["stake_service_type"] = None

    def burn_interaction_stakes(
        self,
        interaction_id: str,
        burner_id: str,
    ) -> int:
        """Burn all active TNFTs staked for an interaction."""
        burned_count = 0

        # Fix B: use stake index — O(staked_count) instead of O(total_ledger).
        for tnft in self._staked_by_interaction.pop(interaction_id, []):
            if not tnft["status"]:   # guard: already burned by another path
                continue

            tnft["status"] = False
            tnft["burned_by"] = burner_id
            tnft["burn_timestamp"] = self.time
            tnft["burn_reason"] = "staked_interaction_failure"

            tnft["staked_for"] = None
            tnft["stake_role"] = None
            tnft["stake_service_type"] = None

            # Perf J: update burned-by index and invalidate trust score cache.
            _owner = tnft["owner"]
            self._tokens_burned_by[burner_id].add(_owner)
            self._invalidate_trust_score_cache(_owner)
            # Perf G: invalidate trust evidence cache.
            for _k in [k for k in self._trust_evidence_cache if k[0] == _owner]:
                del self._trust_evidence_cache[_k]

            target_agent = self._find_delivery_agent(tnft["owner"])
            if target_agent is not None:
                target_agent.cached_active_tnfts = max(
                    0,
                    target_agent.cached_active_tnfts - 1,
                )
                target_agent.cached_burned_tnfts += 1

            burned_count += 1

        return burned_count

    def get_vtp_summary(
        self,
        agent_id: str,
        service_type: str | None = None,
        evaluator: DeliveryAgent = None,
    ) -> dict[str, Any]:
        """Return a trust summary from the perspective of evaluator.

        The score is calculated using evaluator's own trust parameters
        (weights, priors, burn multiplier). Always pass an evaluator — the
        module-level constants that the old fallback used are not guaranteed
        to match any specific agent's configuration, so omitting the evaluator
        produces scores that are inconsistent with live agent decisions.
        """
        if evaluator is None:
            raise TypeError(
                "get_vtp_summary() requires an evaluator agent. "
                "Pass the agent whose trust perspective should be used."
            )

        evidence = self.get_trust_evidence(agent_id, service_type)
        return evaluator.calculate_trust_summary_from_evidence(evidence)

    def get_reviewer_summary(self, reviewer_id: str) -> dict[str, Any]:
        # Perf D: return cached result when available.
        # Cache is invalidated in record_outcome whenever a new outcome is
        # recorded for an interaction where this agent was the truster (reviewer).
        if reviewer_id in self._reviewer_cache:
            return self._reviewer_cache[reviewer_id]

        total_reviews = 0
        positive_reviews = 0
        negative_reviews = 0
        disputed_reviews = 0

        for interaction in self.interactions.values():
            if interaction.truster_id != reviewer_id:
                continue

            outcome = self.outcomes.get(interaction.interaction_id)

            if outcome is None:
                continue

            total_reviews += 1

            if outcome.status == "success":
                positive_reviews += 1
            elif outcome.status == "failure":
                negative_reviews += 1
            elif outcome.status == "disputed":
                disputed_reviews += 1

        negative_review_rate = (
            negative_reviews / total_reviews if total_reviews > 0 else 0.0
        )

        result = {
            "reviewer_id": reviewer_id,
            "total_reviews": total_reviews,
            "positive_reviews": positive_reviews,
            "negative_reviews": negative_reviews,
            "disputed_reviews": disputed_reviews,
            "negative_review_rate": negative_review_rate,
        }
        self._reviewer_cache[reviewer_id] = result
        return result

    def query_vtp(self, agent_id: str) -> int:
        """Return the legacy active-token count used by older callers.

        New code should prefer trust evidence or agent-calculated trust summaries.
        """
        return self.get_trust_evidence(agent_id)["total_active"]

    def burn_tnft(
        self, burner_id: str, target_id: str, service_type: str | None = None
    ) -> bool:
        # Fix A: use owner index — same filter, O(own_tokens) not O(total_ledger).
        active_tnfts = [
            tnft
            for tnft in self._ledger_by_owner.get(target_id, [])
            if tnft["status"]
        ]

        if service_type is not None:
            matching_service_tnfts = [
                tnft for tnft in active_tnfts if tnft["service_type"] == service_type
            ]
            bootstrap_tnfts = [
                tnft
                for tnft in active_tnfts
                if tnft["service_type"] == BOOTSTRAP_SERVICE
            ]
            active_tnfts = matching_service_tnfts or bootstrap_tnfts

        if not active_tnfts:
            return False

        active_tnfts.sort(key=lambda tnft: (tnft["type"] != "bootstrap", tnft["id"]))
        tnft_to_burn = active_tnfts[0]

        tnft_to_burn["status"] = False
        tnft_to_burn["burned_by"] = burner_id
        tnft_to_burn["burn_timestamp"] = self.time

        # Perf J: update burned-by index and invalidate trust score cache.
        self._tokens_burned_by[burner_id].add(target_id)
        self._invalidate_trust_score_cache(target_id)
        # Perf G: invalidate trust evidence cache for burned agent.
        for _k in [k for k in self._trust_evidence_cache if k[0] == target_id]:
            del self._trust_evidence_cache[_k]

        target_agent = self._find_delivery_agent(target_id)
        if target_agent is not None:
            target_agent.cached_active_tnfts = max(
                0, target_agent.cached_active_tnfts - 1
            )
            target_agent.cached_burned_tnfts += 1

        return True

    def _invalidate_trust_score_cache(self, changed_agent_id: str) -> None:
        """Remove trust score cache entries affected by a TNFT state change.

        Three sets of entries are removed:
          (a) Any score OF changed_agent_id  (direct: their token count changed).
          (b) filter=True scores where target has tokens ISSUED BY changed_agent_id
              (issuer trust re-evaluated in _filter_untrusted_trust_evidence).
          (c) filter=True scores where target had a token BURNED BY changed_agent_id
              (burned evidence filtered by burned_by identity).
        filter=False entries for targets ≠ changed_agent_id are guaranteed correct
        (they only depend on the target's own token counts, which did not change).
        """
        if not self._trust_score_cache:
            return

        # Targets whose filtered score may have changed via issuer/burner effects
        indirect: set = (
            self._tokens_by_issuer.get(changed_agent_id, frozenset())
            | self._tokens_burned_by.get(changed_agent_id, frozenset())
        )

        to_remove = [
            k for k in self._trust_score_cache
            if (
                k[1] == changed_agent_id          # (a) direct
                or (k[3] and k[1] in indirect)    # (b)/(c) indirect, filter=True only
            )
        ]
        for k in to_remove:
            del self._trust_score_cache[k]

    def _find_delivery_agent(self, agent_id: str) -> DeliveryAgent | None:
        # Perf E: O(1) dict lookup — index built in _cache_agents.
        return self._agent_index.get(agent_id)

    def get_drop_off_coordinate(self, drop_off_name: str | None) -> Coordinate | None:
        if drop_off_name is None:
            return None
        # Perf H: O(1) dict lookup — index built in _cache_agents.
        drop_off = self._dropoff_index.get(drop_off_name)
        return drop_off.cell.coordinate if drop_off is not None else None

    def request_map_data(
        self, requester: DeliveryAgent, target_name: str
    ) -> list[dict[str, Any]]:
        responses = []

        for agent in self.cached_delivery_agents:
            if agent == requester:
                continue

            agent_resp = agent.share_map(requester, target_name)

            if agent_resp is None:
                continue

            provider_distance = self.chebyshev_distance(
                requester.cell.coordinate, agent.cell.coordinate
            )

            responses.append(
                {
                    "agent": agent.unique_id,
                    "dist": provider_distance,
                    "coord": agent_resp["coord"],
                    "provider_stake_limit": agent_resp["provider_stake_limit"],
                    "requester_stake_required": agent_resp["requester_stake_required"],
                    "provider_stake_limit_meta": agent_resp[
                        "provider_stake_limit_meta"
                    ],
                    "requester_stake_required_meta": agent_resp[
                        "requester_stake_required_meta"
                    ],
                }
            )

        responses = sorted(responses, key=lambda response: response["dist"])

        return responses

    def distribute_initial_knowledge(self) -> None:
        """
        Evenly distribute drop-off location coordinates among the delivery agents.
        Makes sure each delivery agent knows at least one drop-off location before giving a second coordinate.
        If there are less drop-offs than agents then some agents will not start with any initial knowledge.
        """
        if not self.cached_delivery_agents or not self.cached_drop_offs:
            return

        delivery_agents = list(self.cached_delivery_agents)
        self.random.shuffle(delivery_agents)

        for i, drop_off in enumerate(self.cached_drop_offs):
            receiving_agent = delivery_agents[i % len(delivery_agents)]
            receiving_agent.update_internal_map(
                coordinate=drop_off.cell.coordinate,
                env_type=ENV_DROP_OFF,
                info_source=SOURCE_SYSTEM,
                drop_off_name=drop_off.unique_id,
            )

    def dispatch_packages(self) -> None:
        """
        Randomly assign new delivery goals for each agent.
        Will not assign the same goal as lsat time.
        """
        if not self.cached_delivery_agents or not self.cached_drop_offs:
            return

        # Perf I: build the without-packages list once and short-circuit if
        # all agents are busy — avoids the O(n) comprehension when not needed.
        _idle_agents = self._agents_without_packages()
        if not _idle_agents:
            return

        for agent in _idle_agents:
            possible_destinations = [
                d for d in self.cached_drop_offs if d.unique_id != agent.prev_goal_name
            ]
            if not possible_destinations:
                continue

            new_destination = self.random.choice(possible_destinations)
            min_steps_to_destination = self.chebyshev_distance(
                agent.cell.coordinate, new_destination.cell.coordinate
            )
            max_steps_to_destination = int(
                # min_steps_to_destination * (self.random.betavariate(5, 5) + 1) + 1
                min_steps_to_destination * 2
            )

            package = {
                "destination": new_destination.unique_id,
                "max_steps": max_steps_to_destination,
                "min_steps": min_steps_to_destination,
                "steps_taken": 0,
            }
            agent.receive_package(package)

    def _agents_without_packages(self) -> list[DeliveryAgent]:
        return [
            agent for agent in self.cached_delivery_agents if agent.goal_name is None
        ]

    def verify_delivery(self, agent: DeliveryAgent, package: dict[str, Any]) -> bool:
        agents_on_cell = [a.unique_id for a in agent.cell.agents]

        if agent.goal_name not in agents_on_cell:
            return False

        min_steps = package["min_steps"]
        max_steps = package["max_steps"]
        steps_taken = package["steps_taken"]

        window = max_steps - min_steps
        grace_steps = min_steps + int(window * GRACE_WINDOW_RATIO)

        if steps_taken <= grace_steps:
            points_awarded = BASE_DELIVERY_POINTS

        elif steps_taken <= max_steps:
            if max_steps == grace_steps:
                points_awarded = 0.0
            else:
                decay_ratio = (max_steps - steps_taken) / (max_steps - grace_steps)
                points_awarded = float(BASE_DELIVERY_POINTS * decay_ratio)

        else:
            lateness = steps_taken - max_steps
            points_awarded = max(
                (-BASE_DELIVERY_POINTS * 2), (-lateness * LATE_PENALTY_PER_STEP)
            )

        agent.points += points_awarded
        return True

    @staticmethod
    def chebyshev_distance(a: tuple, b: tuple) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def step(self) -> None:
        self.agents.shuffle_do("step")
        self.dispatch_packages()
