import mesa
from mesa.discrete_space import CellAgent, FixedAgent
from typing_extensions import override
from typing import Any, Literal, TypedDict
from math import ceil

Coordinate = tuple[int, int]
AgentState = Literal["IDLE", "EXPLORING", "DELIVERING"]
StakeRole = Literal["requester", "provider"]
ReportedOutcomeStatus = Literal["success", "failure"]

ENV_FLOOR = "floor"
ENV_DROP_OFF = "drop_off"

SOURCE_SELF = "self"
SOURCE_SYSTEM = "system"

VERIFIED_MAP_SOURCES = {SOURCE_SELF, SOURCE_SYSTEM}

MAP_DATA_SERVICE = "map_data"
TRUSTED_SYSTEM_ISSUERS = {"SYSTEM"}


class Package(TypedDict):
    destination: str
    max_steps: int
    min_steps: int
    steps_taken: int


class MapRecord(TypedDict):
    type: str
    source: str


class MapShareResponse(TypedDict):
    coord: Coordinate
    provider_stake_limit: int
    requester_stake_required: int
    provider_stake_limit_meta: dict[str, Any]
    requester_stake_required_meta: dict[str, Any]


class DeliveryAgent(CellAgent):
    def __init__(
        self,
        model: mesa.Model,
        cell: mesa.discrete_space.Cell,
        vision_radius: int = 1,
        trust_reject_threshold: float = 0.3,
        trust_accept_threshold: float = 0.8,
        max_negative_review_rate: float = 0.6,
        min_reviews_before_reviewer_check: int = 10,
        context_match_weight: float = 1.0,
        other_context_weight: float = 0.25,
        trust_prior_active: float = 1.0,
        trust_prior_burned: float = 1.0,
        burned_weight_multiplier: float = 1.0,
        filter_untrusted_evidence: bool = True,
        staking_min_fraction: float = 0.25,
        staking_max_fraction: float = 1.0,
        provider_stake_vtp_weight: float = 0.8,
        provider_stake_reviewer_weight: float = 0.2,
        requester_stake_vtp_weight: float = 0.4,
        requester_stake_reviewer_weight: float = 0.6,
    ) -> None:
        """
        An agent that delivers packages to drop-off locations. Can share map data with other agents.

        :param model: The mesa model.
        :param cell: The cell on which the agent spawns.
        :param vision_radius: The vision radius.
        :param trust_reject_threshold: Score at or below which this agent always rejects trust.
        :param trust_accept_threshold: Score at or above which this agent always accepts trust.
        """

        super().__init__(model)
        self.cell = cell
        self.internal_map: dict[Coordinate, MapRecord] = {}

        # verified drop-off coordinates, safe to share.
        self.known_drop_offs: dict[str, Coordinate] = {}

        # unverified drop-off coordinates, can be used by this agent
        # but should not be shared until verified
        self.candidate_drop_offs: dict[str, Coordinate] = {}

        self.goal_name: str | None = None
        self.prev_goal_name: str | None = None
        self.state: AgentState = "IDLE"
        self.target_coordinate: Coordinate | None = None

        self.package: Package | None = None
        self.current_provider_id: str | None = None
        self.current_interaction_id: str | None = None

        self.vision_radius = vision_radius
        self.trust_reject_threshold = trust_reject_threshold
        self.trust_accept_threshold = trust_accept_threshold

        if not 0.0 <= self.trust_reject_threshold < self.trust_accept_threshold <= 1.0:
            raise ValueError(
                "Trust thresholds must satisfy "
                "0.0 <= reject_threshold < accept_threshold <= 1.0."
            )

        self.max_negative_review_rate = self._validate_probability(
            max_negative_review_rate,
            "max_negative_review_rate",
        )
        self.min_reviews_before_reviewer_check = int(min_reviews_before_reviewer_check)

        if self.min_reviews_before_reviewer_check < 0:
            raise ValueError("min_reviews_before_reviewer_check must be >= 0.")

        self.context_match_weight = float(context_match_weight)
        self.other_context_weight = float(other_context_weight)
        self.trust_prior_active = float(trust_prior_active)
        self.trust_prior_burned = float(trust_prior_burned)
        self.burned_weight_multiplier = float(burned_weight_multiplier)
        self.filter_untrusted_evidence = bool(filter_untrusted_evidence)

        if self.context_match_weight < 0:
            raise ValueError("context_match_weight must be >= 0.")

        if self.other_context_weight < 0:
            raise ValueError("other_context_weight must be >= 0.")

        if self.trust_prior_active <= 0:
            raise ValueError("trust_prior_active must be > 0.")

        if self.trust_prior_burned <= 0:
            raise ValueError("trust_prior_burned must be > 0.")

        if self.burned_weight_multiplier < 0:
            raise ValueError("burned_weight_multiplier must be >= 0.")

        self.staking_min_fraction = self._validate_probability(
            staking_min_fraction,
            "staking_min_fraction",
        )
        self.staking_max_fraction = self._validate_probability(
            staking_max_fraction,
            "staking_max_fraction",
        )

        if self.staking_max_fraction < self.staking_min_fraction:
            raise ValueError("staking_max_fraction must be >= staking_min_fraction.")

        self.provider_stake_vtp_weight = float(provider_stake_vtp_weight)
        self.provider_stake_reviewer_weight = float(provider_stake_reviewer_weight)
        self.requester_stake_vtp_weight = float(requester_stake_vtp_weight)
        self.requester_stake_reviewer_weight = float(requester_stake_reviewer_weight)

        for name, value in (
            ("provider_stake_vtp_weight", self.provider_stake_vtp_weight),
            ("provider_stake_reviewer_weight", self.provider_stake_reviewer_weight),
            ("requester_stake_vtp_weight", self.requester_stake_vtp_weight),
            ("requester_stake_reviewer_weight", self.requester_stake_reviewer_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")

        if self.provider_stake_vtp_weight + self.provider_stake_reviewer_weight <= 0:
            raise ValueError("Provider stake weights must have a positive total.")

        if self.requester_stake_vtp_weight + self.requester_stake_reviewer_weight <= 0:
            raise ValueError("Requester stake weights must have a positive total.")

        self.points = 0.0
        self.delivery_count = 0
        self._all_possible_coords = model.all_coordinates
        self.cached_active_tnfts = 0
        self.cached_burned_tnfts = 0

        self.last_decision_type: str = "none"
        self.last_decision_reason: str = "none"
        self.last_decision_peer_id: str | None = None
        self.last_decision_service_type: str | None = None
        self.last_checked_trust_score: float | None = None
        self.last_checked_negative_review_rate: float | None = None

    @property
    def map_size(self) -> int:
        return len(self.internal_map)

    @property
    def known_drop_offs_count(self) -> int:
        return len(self.known_drop_offs)

    @property
    def steps_on_package(self) -> int:
        return self.package["steps_taken"] if self.package else 0

    def verify_vtp(self, target_id: str, service_type: str = MAP_DATA_SERVICE) -> bool:
        summary = self.calculate_trust_summary(target_id, service_type)
        return self._accepts_trust_score(summary["score"])

    def verify_credibility(
        self,
        reviewer_id: str,
    ) -> bool:
        summary = self.model.get_reviewer_summary(reviewer_id)

        if summary["total_reviews"] < self.min_reviews_before_reviewer_check:
            return True

        return summary["negative_review_rate"] <= self.max_negative_review_rate

    @staticmethod
    def _validate_probability(value: float, name: str) -> float:
        value = float(value)

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} probability must be between 0.0 and 1.0")

        return value

    def record_decision(
        self,
        decision_type: str,
        reason: str,
        peer_id: str | None = None,
        service_type: str | None = None,
        trust_score: float | None = None,
        negative_review_rate: float | None = None,
    ) -> None:
        self.last_decision_type = decision_type
        self.last_decision_reason = reason
        self.last_decision_peer_id = peer_id
        self.last_decision_service_type = service_type
        self.last_checked_trust_score = trust_score
        self.last_checked_negative_review_rate = negative_review_rate

        # Perf F: increment the O(1) reason counter on the model.
        # This replaces the need to iterate decision_events for counting.
        if hasattr(self.model, "_decision_counts"):
            counts = self.model._decision_counts
            counts[reason] = counts.get(reason, 0) + 1

        # Keep the full event list only when DEBUG_DECISIONS is set.
        # In production runs this list is left empty, saving ~2M dict
        # allocations per run and significant GC pressure.
        if getattr(self.model, "DEBUG_DECISIONS", False) and hasattr(self.model, "decision_events"):
            self.model.decision_events.append(
                {
                    "timestamp": getattr(self.model, "time", None),
                    "step": getattr(self.model, "steps", None),
                    "agent_id": self.unique_id,
                    "agent_type": type(self).__name__,
                    "decision_type": decision_type,
                    "reason": reason,
                    "peer_id": peer_id,
                    "service_type": service_type,
                    "trust_score": trust_score,
                    "negative_review_rate": negative_review_rate,
                }
            )

    def calculate_trust_summary(
        self,
        target_id: str,
        service_type: str | None = MAP_DATA_SERVICE,
        filter_untrusted_evidence: bool | None = None,
    ) -> dict[str, Any]:
        """Calculate this agent's trust summary for another agent.

        The model supplies raw ledger evidence. This agent decides how that
        evidence should be filtered, weighted, and converted into a trust score.

        Perf J: results are cached on the model keyed by
        (evaluator_id, target_id, service_type, filter_flag). The cache is
        invalidated precisely in _invalidate_trust_score_cache whenever
        a TNFT state change could affect this score.
        """
        # Resolve filter flag early — it is part of the cache key.
        _use_filter: bool = (
            self.filter_untrusted_evidence
            if filter_untrusted_evidence is None
            else bool(filter_untrusted_evidence)
        )

        # Perf J: check trust score cache.
        _tsc = getattr(self.model, "_trust_score_cache", None)
        if _tsc is not None:
            _key = (self.unique_id, target_id, service_type, _use_filter)
            _cached = _tsc.get(_key)
            if _cached is not None:
                return _cached

        evidence = self.model.get_trust_evidence(target_id, service_type)
        result = self.calculate_trust_summary_from_evidence(
            evidence=evidence,
            filter_untrusted_evidence=_use_filter,
        )

        # Perf J: store result.
        if _tsc is not None:
            _tsc[(self.unique_id, target_id, service_type, _use_filter)] = result

        return result

    def calculate_trust_summary_from_evidence(
        self,
        evidence: dict[str, Any],
        filter_untrusted_evidence: bool | None = None,
    ) -> dict[str, Any]:
        if filter_untrusted_evidence is None:
            filter_untrusted_evidence = self.filter_untrusted_evidence

        unfiltered_total_active = evidence["total_active"]
        unfiltered_total_burned = evidence["total_burned"]

        if filter_untrusted_evidence:
            evidence = self._filter_untrusted_trust_evidence(evidence)

        weighted_active = self._calculate_weighted_trust_evidence(
            context_matching_count=evidence["context_matching_active"],
            other_context_count=evidence["other_active"],
        )
        weighted_burned = self._calculate_weighted_trust_evidence(
            context_matching_count=evidence["context_matching_burned"],
            other_context_count=evidence["other_burned"],
        )

        weighted_burned *= self.burned_weight_multiplier

        score = self._calculate_trust_score_from_weights(
            weighted_active=weighted_active,
            weighted_burned=weighted_burned,
        )

        return {
            **evidence,
            "unfiltered_total_active": unfiltered_total_active,
            "unfiltered_total_burned": unfiltered_total_burned,
            "filtered_out_active": unfiltered_total_active - evidence["total_active"],
            "filtered_out_burned": unfiltered_total_burned - evidence["total_burned"],
            "filter_untrusted_evidence": filter_untrusted_evidence,
            "weighted_active": weighted_active,
            "weighted_burned": weighted_burned,
            "score": score,
            "trust_calculator": type(self).__name__,
            "context_match_weight": self.context_match_weight,
            "other_context_weight": self.other_context_weight,
            "trust_prior_active": self.trust_prior_active,
            "trust_prior_burned": self.trust_prior_burned,
            "burned_weight_multiplier": self.burned_weight_multiplier,
        }

    def _filter_untrusted_trust_evidence(
        self,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        service_type = evidence["service_type"]

        trusted_active_tnfts = [
            tnft
            for tnft in evidence["active_tnfts"]
            if self._trusts_evidence_source(tnft.get("issuer"), service_type)
        ]

        trusted_burned_tnfts = [
            tnft
            for tnft in evidence["burned_tnfts"]
            if self._trusts_evidence_source(tnft.get("burned_by"), service_type)
        ]

        return self._build_trust_evidence_from_tnfts(
            original_evidence=evidence,
            active_tnfts=trusted_active_tnfts,
            burned_tnfts=trusted_burned_tnfts,
        )

    def _trusts_evidence_source(
        self,
        source_id: Any,
        service_type: str | None,
    ) -> bool:
        """Return whether this agent accepts evidence created by source_id.

        This uses a hard reject-threshold rule, not the probabilistic trust
        acceptance rule, because evidence filtering should be deterministic.
        """
        if source_id is None:
            return False

        if source_id in TRUSTED_SYSTEM_ISSUERS:
            return True

        if source_id == self.unique_id:
            return True

        source_summary = self.calculate_trust_summary(
            target_id=source_id,
            service_type=service_type,
            filter_untrusted_evidence=False,
        )

        return source_summary["score"] > self.trust_reject_threshold

    def _build_trust_evidence_from_tnfts(
        self,
        original_evidence: dict[str, Any],
        active_tnfts: list[dict[str, Any]],
        burned_tnfts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        service_type = original_evidence["service_type"]

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

        return {
            **original_evidence,
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

    def _calculate_weighted_trust_evidence(
        self,
        context_matching_count: int,
        other_context_count: int,
    ) -> float:
        return (
            self.context_match_weight * context_matching_count
            + self.other_context_weight * other_context_count
        )

    def _calculate_trust_score_from_weights(
        self,
        weighted_active: float,
        weighted_burned: float,
    ) -> float:
        return (self.trust_prior_active + weighted_active) / (
            self.trust_prior_active
            + self.trust_prior_burned
            + weighted_active
            + weighted_burned
        )

    @staticmethod
    def _clamp_probability(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _trust_score_to_risk(self, trust_score: float) -> float:
        """Convert a trust score into a bounded risk value.

        High trust means low risk.
        Low trust means high risk.
        Scores between the reject and accept thresholds scale linearly.
        """
        if trust_score >= self.trust_accept_threshold:
            return 0.0

        if trust_score <= self.trust_reject_threshold:
            return 1.0

        risk = (self.trust_accept_threshold - trust_score) / (
            self.trust_accept_threshold - self.trust_reject_threshold
        )
        return self._clamp_probability(risk)

    def _reviewer_summary_to_risk(self, reviewer_summary: dict[str, Any]) -> float:
        """Convert reviewer history into a bounded risk value."""
        if reviewer_summary["total_reviews"] < self.min_reviews_before_reviewer_check:
            return 0.0

        return self._clamp_probability(reviewer_summary["negative_review_rate"])

    def _stake_weights_for_role(self, role: StakeRole) -> tuple[float, float]:
        if role == "provider":
            return self.provider_stake_vtp_weight, self.provider_stake_reviewer_weight

        if role == "requester":
            return self.requester_stake_vtp_weight, self.requester_stake_reviewer_weight

        raise ValueError(f"Unknown stake role: {role}")

    def calculate_interaction_risk(
        self,
        agent_id: str,
        role: StakeRole,
        service_type: str = MAP_DATA_SERVICE,
    ) -> dict[str, Any]:
        """Calculate how risky another agent looks in a specific interaction role."""
        trust_summary = self.calculate_trust_summary(agent_id, service_type)
        reviewer_summary = self.model.get_reviewer_summary(agent_id)

        trust_score = trust_summary["score"]
        vtp_risk = self._trust_score_to_risk(trust_score)
        reviewer_risk = self._reviewer_summary_to_risk(reviewer_summary)

        vtp_weight, reviewer_weight = self._stake_weights_for_role(role)
        total_weight = vtp_weight + reviewer_weight

        combined_risk = (
            (vtp_weight * vtp_risk) + (reviewer_weight * reviewer_risk)
        ) / total_weight

        return {
            "agent_id": agent_id,
            "role": role,
            "service_type": service_type,
            "trust_score": trust_score,
            "negative_review_rate": reviewer_summary["negative_review_rate"],
            "total_reviews": reviewer_summary["total_reviews"],
            "vtp_risk": vtp_risk,
            "reviewer_risk": reviewer_risk,
            "combined_risk": self._clamp_probability(combined_risk),
            "vtp_weight": vtp_weight,
            "reviewer_weight": reviewer_weight,
        }

    def _stake_fraction_from_risk(self, risk: float) -> float:
        """Convert bounded risk into a stake fraction."""
        if not getattr(self.model, "staking_enabled", False):
            return 0.0

        risk = self._clamp_probability(risk)

        return self.staking_min_fraction + risk * (
            self.staking_max_fraction - self.staking_min_fraction
        )

    def _stake_amount_from_risk(self, risk: float, available_stake: int) -> int:
        """Convert bounded risk into an available-TNFT-relative stake amount."""
        if not getattr(self.model, "staking_enabled", False):
            return 0

        available_stake = int(available_stake)

        if available_stake <= 0:
            return 0

        stake_fraction = self._stake_fraction_from_risk(risk)
        stake_amount = ceil(available_stake * stake_fraction)

        return max(1, min(available_stake, stake_amount))

    def calculate_required_stake(
        self,
        agent_id: str,
        role: StakeRole,
        service_type: str = MAP_DATA_SERVICE,
    ) -> dict[str, Any]:
        """Calculate how much stake this agent requires from another agent."""
        risk_summary = self.calculate_interaction_risk(
            agent_id=agent_id,
            role=role,
            service_type=service_type,
        )

        available_stake = self.model.available_stake_count(
            agent_id,
            service_type,
        )
        stake_fraction = self._stake_fraction_from_risk(risk_summary["combined_risk"])
        required_stake = self._stake_amount_from_risk(
            risk=risk_summary["combined_risk"],
            available_stake=available_stake,
        )

        return {
            **risk_summary,
            "available_stake": available_stake,
            "stake_fraction": stake_fraction,
            "required_stake": required_stake,
        }

    def _stake_limit_fraction_from_risk(self, risk: float) -> float:
        """Convert peer risk into a maximum stake fraction.

        Higher peer risk means this agent is willing to risk less of its own stake.
        """
        if not getattr(self.model, "staking_enabled", False):
            return 0.0

        risk = self._clamp_probability(risk)

        return self.staking_min_fraction + (1.0 - risk) * (
            self.staking_max_fraction - self.staking_min_fraction
        )

    def _stake_amount_from_fraction(
        self,
        stake_fraction: float,
        available_stake: int,
    ) -> int:
        if not getattr(self.model, "staking_enabled", False):
            return 0

        available_stake = int(available_stake)

        if available_stake <= 0:
            return 0

        stake_fraction = self._clamp_probability(stake_fraction)
        stake_amount = ceil(available_stake * stake_fraction)

        return max(1, min(available_stake, stake_amount))

    def calculate_stake_limit(
        self,
        peer_id: str,
        peer_role: StakeRole,
        service_type: str = MAP_DATA_SERVICE,
    ) -> dict[str, Any]:
        """Calculate the maximum stake this agent is willing to risk with a peer.

        If the peer looks risky, this agent is willing to risk less.
        If the peer looks trustworthy, this agent is willing to risk more.
        """
        risk_summary = self.calculate_interaction_risk(
            agent_id=peer_id,
            role=peer_role,
            service_type=service_type,
        )

        available_stake = self.model.available_stake_count(
            self.unique_id,
            service_type,
        )
        stake_limit_fraction = self._stake_limit_fraction_from_risk(
            risk_summary["combined_risk"]
        )
        stake_limit = self._stake_amount_from_fraction(
            stake_fraction=stake_limit_fraction,
            available_stake=available_stake,
        )

        return {
            **risk_summary,
            "available_stake": available_stake,
            "stake_limit_fraction": stake_limit_fraction,
            "stake_limit": stake_limit,
        }

    def accepts_stake_offer(
        self,
        agent_id: str,
        role: StakeRole,
        offered_stake: int,
        service_type: str = MAP_DATA_SERVICE,
    ) -> bool:
        """Return whether another agent's stake offer is enough for this agent."""
        if not getattr(self.model, "staking_enabled", False):
            return True

        stake_requirement = self.calculate_required_stake(
            agent_id=agent_id,
            role=role,
            service_type=service_type,
        )

        return int(offered_stake) >= stake_requirement["required_stake"]

    def build_map_share_response(
        self,
        requester: CellAgent,
        coordinate: Coordinate,
        service_type: str = MAP_DATA_SERVICE,
    ) -> MapShareResponse:
        provider_stake_limit = self.calculate_stake_limit(
            peer_id=requester.unique_id,
            peer_role="requester",
            service_type=service_type,
        )
        requester_stake_required = self.calculate_required_stake(
            agent_id=requester.unique_id,
            role="requester",
            service_type=service_type,
        )

        return {
            "coord": coordinate,
            "provider_stake_limit": provider_stake_limit["stake_limit"],
            "requester_stake_required": requester_stake_required["required_stake"],
            "provider_stake_limit_meta": provider_stake_limit,
            "requester_stake_required_meta": requester_stake_required,
        }

    def _accepts_trust_score(self, score: float) -> bool:
        """Decide whether this agent accepts another agent's ledger-derived score."""
        if score <= self.trust_reject_threshold:
            return False

        if score >= self.trust_accept_threshold:
            return True

        acceptance_probability = (score - self.trust_reject_threshold) / (
            self.trust_accept_threshold - self.trust_reject_threshold
        )

        return self.random.random() <= acceptance_probability

    def accepts_requester_for_service(
        self,
        requester: CellAgent,
        service_type: str = MAP_DATA_SERVICE,
    ) -> bool:
        # doing it like this just for logging purposes
        trust_summary = self.calculate_trust_summary(requester.unique_id, service_type)
        reviewer_summary = self.model.get_reviewer_summary(requester.unique_id)

        trust_score = trust_summary["score"]
        negative_review_rate = reviewer_summary["negative_review_rate"]

        if not self._accepts_trust_score(trust_score):
            self.record_decision(
                decision_type="share_map",
                reason="requester_rejected_low_vtp",
                peer_id=requester.unique_id,
                service_type=service_type,
                trust_score=trust_score,
                negative_review_rate=negative_review_rate,
            )
            return False

        # TODO: maybe we should always have this check, even when the other agent isn't leaving a review?
        if not self.verify_credibility(requester.unique_id):
            self.record_decision(
                decision_type="share_map",
                reason="requester_rejected_low_reviewer_credibility",
                peer_id=requester.unique_id,
                service_type=service_type,
                trust_score=trust_score,
                negative_review_rate=negative_review_rate,
            )
            return False

        self.record_decision(
            decision_type="share_map",
            reason="requester_accepted",
            peer_id=requester.unique_id,
            service_type=service_type,
            trust_score=trust_score,
            negative_review_rate=negative_review_rate,
        )
        return True

    def build_outcome_meta(self) -> dict[str, Any]:
        return {
            "goal_name": self.goal_name,
            "provider_id": self.current_provider_id,
            "interaction_id": self.current_interaction_id,
        }

    def move(self) -> bool:
        """
        Move one step toward the current target coordinate.
        Returns True if the agent moved, False otherwise.
        """
        if self.target_coordinate is None:
            return False

        moved = self._move_one_step_towards_target()

        if moved and self.package is not None:
            self.package["steps_taken"] += 1

        if self.cell.coordinate == self.target_coordinate:
            self._handle_target_reached()

        return moved

    def _move_one_step_towards_target(self) -> bool:
        if self.target_coordinate is None:
            return False

        current_x, current_y = self.cell.coordinate
        target_x, target_y = self.target_coordinate

        dx = 1 if current_x < target_x else -1 if current_x > target_x else 0
        dy = 1 if current_y < target_y else -1 if current_y > target_y else 0

        if dx == 0 and dy == 0:
            return False

        self.move_relative((dx, dy))
        return True

    def _handle_target_reached(self) -> None:
        if self.state == "DELIVERING":
            if self._goal_drop_off_is_on_current_cell():
                self._complete_delivery()
            else:
                self._handle_bad_delivery_target()

        self._clear_current_target()

    def _goal_drop_off_is_on_current_cell(self) -> bool:
        if self.goal_name is None:
            return False

        return any(agent.unique_id == self.goal_name for agent in self.cell.agents)

    def _complete_delivery(self) -> None:
        if self.package is None:
            return

        previous_points = self.points
        success = self.model.verify_delivery(self, self.package)

        if not success:
            return

        points_delta = self.points - previous_points
        outcome_meta = self.build_outcome_meta()
        outcome_meta["points_delta"] = points_delta

        self.delivery_count += 1

        if self.goal_name is not None:
            self.update_internal_map(
                coordinate=self.cell.coordinate,
                env_type=ENV_DROP_OFF,
                info_source=SOURCE_SELF,
                drop_off_name=self.goal_name,
            )

        self._settle_current_interaction(
            outcome_status="success",
            points_delta=points_delta,
            outcome_meta=outcome_meta,
        )

        self.prev_goal_name = self.goal_name
        self.goal_name = None
        self.package = None

    def _handle_bad_delivery_target(self) -> None:
        bad_goal_name = self.goal_name
        bad_coordinate = self.target_coordinate
        bad_provider_id = self.current_provider_id

        outcome_meta = self.build_outcome_meta()
        outcome_meta["points_delta"] = 0.0

        self._settle_current_interaction(
            outcome_status="failure",
            points_delta=0.0,
            outcome_meta=outcome_meta,
        )

        # The provider's coordinate failed for the current goal.
        # Remove only the current goal mapping.
        self._remove_drop_off_name(bad_goal_name)

        if bad_coordinate is None:
            return

        record = self.internal_map.get(bad_coordinate)

        # If perception already proved what this coordinate is, keep that
        # self/system-verified knowledge. Only remove stale provider-created
        # coordinate records.
        coordinate_still_has_name = any(
            coord == bad_coordinate for coord in self.known_drop_offs.values()
        ) or any(coord == bad_coordinate for coord in self.candidate_drop_offs.values())

        if (
            record is not None
            and record["source"] == bad_provider_id
            and not coordinate_still_has_name
        ):
            self.internal_map.pop(bad_coordinate, None)

    def decide_reported_outcome(
        self, actual_outcome_status: ReportedOutcomeStatus, outcome_meta: dict[str, Any]
    ) -> tuple[ReportedOutcomeStatus, dict[str, Any]]:
        review_meta = dict(outcome_meta)
        review_meta.update(
            {
                "actual_outcome_status": actual_outcome_status,
                "reported_outcome_status": actual_outcome_status,
                "review_was_false": False,
                "review_mode": "honest_review",
                "reviewer_id": self.unique_id,
                "reviewer_type": type(self).__name__,
            }
        )

        return actual_outcome_status, review_meta

    def _settle_current_interaction(
        self,
        outcome_status: ReportedOutcomeStatus,
        points_delta: float,
        outcome_meta: dict | None = None,
    ) -> None:
        interaction_id = self.current_interaction_id

        if interaction_id is None:
            return

        # Clear the reference before notifying the model so the method is
        # self-contained — callers do not need to follow this with
        # _clear_current_target to avoid a double-settle.
        self.current_interaction_id = None

        outcome_meta = (
            dict(outcome_meta)
            if outcome_meta is not None
            else self.build_outcome_meta()
        )
        outcome_meta["points_delta"] = points_delta

        reported_outcome_status, reported_outcome_meta = self.decide_reported_outcome(
            actual_outcome_status=outcome_status, outcome_meta=outcome_meta
        )

        self.model.settle_interaction(
            interaction_id=interaction_id,
            evaluator_id=self.unique_id,
            outcome_status=reported_outcome_status,
            outcome_meta=reported_outcome_meta,
        )

    def _clear_current_target(self) -> None:
        self.state = "IDLE"
        self.target_coordinate = None
        self.current_provider_id = None
        self.current_interaction_id = None

    @staticmethod
    def _is_verified_map_source(info_source: str) -> bool:
        return info_source in VERIFIED_MAP_SOURCES

    def _get_goal_coordinate(self, drop_off_name: str | None) -> Coordinate | None:
        if drop_off_name is None:
            return None

        if drop_off_name in self.known_drop_offs:
            return self.known_drop_offs[drop_off_name]

        return self.candidate_drop_offs.get(drop_off_name)

    def _remove_drop_off_name(self, drop_off_name: str | None) -> None:
        if drop_off_name is None:
            return

        self.known_drop_offs.pop(drop_off_name, None)
        self.candidate_drop_offs.pop(drop_off_name, None)

    def _remove_drop_off_coordinate(self, coordinate: Coordinate) -> None:
        for drop_offs in (self.known_drop_offs, self.candidate_drop_offs):
            for drop_off_name, stored_coordinate in list(drop_offs.items()):
                if stored_coordinate == coordinate:
                    drop_offs.pop(drop_off_name, None)

    def update_internal_map(
        self,
        coordinate: Coordinate,
        env_type: str,
        info_source: str = SOURCE_SELF,
        drop_off_name: str | None = None,
    ) -> None:
        """
        Update the agent's internal map.

        Self/system drop-off knowledge is treated as verified.
        Peer-provided drop-off knowledge is treated as candidate knowledge until verified.

        Direct self-observation is allowed to correct previous candidate knowledge.
        """
        if coordinate in self.internal_map:
            existing_source = self.internal_map[coordinate]["source"]

            if self._is_verified_map_source(
                existing_source
            ) and not self._is_verified_map_source(info_source):
                return

        self.internal_map[coordinate] = {
            "type": env_type,
            "source": info_source,
        }

        # Directly observing floor means no known/candidate drop-off should point here.
        if info_source == SOURCE_SELF and env_type != ENV_DROP_OFF:
            self._remove_drop_off_coordinate(coordinate)
            return

        # Directly observing a drop-off means this coordinate belongs to that
        # observed drop-off, not to any other candidate drop-off name.
        if (
            info_source == SOURCE_SELF
            and env_type == ENV_DROP_OFF
            and drop_off_name is not None
        ):
            for drop_offs in (self.known_drop_offs, self.candidate_drop_offs):
                for stored_name, stored_coordinate in list(drop_offs.items()):
                    if stored_coordinate == coordinate and stored_name != drop_off_name:
                        drop_offs.pop(stored_name, None)

        if drop_off_name is None:
            return

        if self._is_verified_map_source(info_source):
            self.known_drop_offs[drop_off_name] = coordinate
            self.candidate_drop_offs.pop(drop_off_name, None)
        elif drop_off_name not in self.known_drop_offs:
            self.candidate_drop_offs[drop_off_name] = coordinate

    def share_map(self, requester: CellAgent, target: str) -> MapShareResponse | None:
        if target not in self.known_drop_offs:
            self.record_decision(
                decision_type="share_map",
                reason="rejected_unknown_target",
                peer_id=requester.unique_id,
                service_type=MAP_DATA_SERVICE,
            )
            return None

        if not self.accepts_requester_for_service(requester, MAP_DATA_SERVICE):
            return None

        coordinate = self.known_drop_offs[target]

        return self.build_map_share_response(
            requester=requester,
            coordinate=coordinate,
            service_type=MAP_DATA_SERVICE,
        )

    def perceive_env(self) -> None:
        """
        Check area visible in vision range and update internal map.
        """

        visible_area = self.cell.get_neighborhood(
            include_center=True, radius=self.vision_radius
        ).cells

        for cell in visible_area:
            drop_off = next(
                (
                    agent
                    for agent in cell.agents
                    if isinstance(agent, DropOffLocationAgent)
                ),
                None,
            )

            if drop_off is not None:
                self.update_internal_map(
                    cell.coordinate, ENV_DROP_OFF, drop_off_name=drop_off.unique_id
                )
            else:
                self.update_internal_map(cell.coordinate, ENV_FLOOR)

    def receive_package(self, package: Package) -> None:
        """
        Set new goal location.

        :param package: The new goal location and the amount of time before expiration.
        """
        self.package = package
        self.goal_name = self.package["destination"]
        self.package["steps_taken"] = 0

    def select_unexplored_coordinate(self) -> Coordinate | None:
        """
        Randomly select an unexplored coordinate.
        If there are no unexplored coordinates, return None.

        :return: None or coordinate
        """

        # all_possible_coordinates = set((x,y) for x in range(self.model.grid.width) for y in range(self.model.grid.height))
        explored_coordinates = set(self.internal_map.keys())
        unexplored_coordinates = tuple(self._all_possible_coords - explored_coordinates)

        if not unexplored_coordinates:
            return None
        return self.random.choice(unexplored_coordinates)

    def step(self) -> None:
        self.perceive_env()

        if self._current_delivery_target_invalidated():
            self._handle_bad_delivery_target()
            self._clear_current_target()
            return

        if not self._has_active_package():
            return

        if self._knows_goal_location() and self.state != "DELIVERING":
            self._start_delivering_to_known_goal()

        elif self._should_search_for_goal_location():
            found_goal_location = self._try_get_goal_location_from_other_agent()

            if found_goal_location:
                self._start_delivering_to_known_goal()
            else:
                self._start_exploring()

        if self.target_coordinate is not None:
            self.move()

    def _has_active_package(self) -> bool:
        return self.goal_name is not None and self.package is not None

    def _knows_goal_location(self) -> bool:
        return self._get_goal_coordinate(self.goal_name) is not None

    def _start_delivering_to_known_goal(self) -> None:
        coordinate = self._get_goal_coordinate(self.goal_name)

        if coordinate is None:
            return

        self.target_coordinate = coordinate
        self.state = "DELIVERING"

    def _should_search_for_goal_location(self) -> bool:
        if self.state == "IDLE":
            return True

        if self.state != "EXPLORING":
            return False

        return (
            self.target_coordinate is None
            or self.target_coordinate in self.internal_map
        )

    def _current_delivery_target_invalidated(self) -> bool:
        """Return True if perception has disproved the current provider map.

        This only applies to deliveries based on an accepted interaction.
        If the current target came from a provider and direct perception removed
        or changed the goal mapping, the accepted map has failed.
        """
        if self.state != "DELIVERING":
            return False

        if self.current_interaction_id is None:
            return False

        if self.goal_name is None:
            return False

        if self.target_coordinate is None:
            return False

        current_goal_coordinate = self._get_goal_coordinate(self.goal_name)

        return current_goal_coordinate != self.target_coordinate

    def _try_get_goal_location_from_other_agent(self) -> bool:
        if self.goal_name is None:
            return False

        responses = self.model.request_map_data(self, self.goal_name)

        for response in responses:
            if self._accept_map_response(response):
                return True

        return False

    def evaluate_map_response_stakes(
        self,
        provider_id: str,
        response: dict[str, Any],
        service_type: str = MAP_DATA_SERVICE,
    ) -> dict[str, Any]:
        """Evaluate whether the staking terms in a provider response are acceptable."""
        staking_enabled = getattr(self.model, "staking_enabled", False)

        provider_stake_limit = int(response.get("provider_stake_limit", 0))
        requester_stake_required = int(response.get("requester_stake_required", 0))

        provider_stake_requirement = self.calculate_required_stake(
            agent_id=provider_id,
            role="provider",
            service_type=service_type,
        )
        requester_stake_limit = self.calculate_stake_limit(
            peer_id=provider_id,
            peer_role="provider",
            service_type=service_type,
        )

        provider_required = provider_stake_requirement["required_stake"]
        requester_limit = requester_stake_limit["stake_limit"]

        provider_available = provider_stake_requirement["available_stake"]
        requester_available = requester_stake_limit["available_stake"]

        provider_has_available_stake = not staking_enabled or provider_available > 0
        requester_has_available_stake = not staking_enabled or requester_available > 0

        provider_can_afford_required_stake = (
            not staking_enabled or provider_available >= provider_required
        )
        requester_can_afford_required_stake = (
            not staking_enabled or requester_available >= requester_stake_required
        )

        provider_willing_to_stake = (
            not staking_enabled or provider_stake_limit >= provider_required
        )
        requester_willing_to_stake = (
            not staking_enabled or requester_limit >= requester_stake_required
        )

        provider_stake_ok = (
            provider_has_available_stake
            and provider_can_afford_required_stake
            and provider_willing_to_stake
        )
        requester_stake_ok = (
            requester_has_available_stake
            and requester_can_afford_required_stake
            and requester_willing_to_stake
        )

        accepted = provider_stake_ok and requester_stake_ok

        rejection_reason = None
        if not provider_has_available_stake:
            rejection_reason = "provider_rejected_no_available_stake"
        elif not provider_can_afford_required_stake:
            rejection_reason = "provider_rejected_insufficient_stake"
        elif not provider_willing_to_stake:
            rejection_reason = "provider_rejected_stake_limit"
        elif not requester_has_available_stake:
            rejection_reason = "requester_rejected_no_available_stake"
        elif not requester_can_afford_required_stake:
            rejection_reason = "requester_rejected_insufficient_stake"
        elif not requester_willing_to_stake:
            rejection_reason = "requester_rejected_stake_limit"

        return {
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "provider_stake_limit": provider_stake_limit,
            "provider_stake_required": provider_required,
            "requester_stake_limit": requester_limit,
            "requester_stake_required": requester_stake_required,
            # Lock exactly what's required, not the full offered ceiling.
            # provider_willing_to_stake already guarantees the offer covers
            # provider_required, so locking the exact requirement is sufficient.
            "provider_stake_to_lock": provider_required if staking_enabled else 0,
            "requester_stake_to_lock": (
                requester_stake_required if staking_enabled else 0
            ),
            "provider_available_stake": provider_available,
            "requester_available_stake": requester_available,
            "provider_has_available_stake": provider_has_available_stake,
            "requester_has_available_stake": requester_has_available_stake,
            "provider_can_afford_required_stake": provider_can_afford_required_stake,
            "requester_can_afford_required_stake": requester_can_afford_required_stake,
            "provider_willing_to_stake": provider_willing_to_stake,
            "requester_willing_to_stake": requester_willing_to_stake,
            "provider_stake_requirement_meta": provider_stake_requirement,
            "requester_stake_limit_meta": requester_stake_limit,
            "provider_response_stake_limit_meta": response.get(
                "provider_stake_limit_meta", {}
            ),
            "provider_response_requester_requirement_meta": response.get(
                "requester_stake_required_meta", {}
            ),
        }

    def _accept_map_response(self, response: dict[str, Any]) -> bool:
        if self.goal_name is None:
            return False

        provider_id = response["agent"]

        provider_summary = self.calculate_trust_summary(provider_id, MAP_DATA_SERVICE)
        provider_reviewer_summary = self.model.get_reviewer_summary(provider_id)

        provider_trust_score = provider_summary["score"]
        provider_negative_review_rate = provider_reviewer_summary[
            "negative_review_rate"
        ]

        if not self._accepts_trust_score(provider_trust_score):
            self.record_decision(
                decision_type="accept_map_response",
                reason="provider_rejected_low_vtp",
                peer_id=provider_id,
                service_type=MAP_DATA_SERVICE,
                trust_score=provider_trust_score,
                negative_review_rate=provider_negative_review_rate,
            )
            return False

        if not self.verify_credibility(provider_id):
            self.record_decision(
                decision_type="accept_map_response",
                reason="provider_rejected_low_reviewer_credibility",
                peer_id=provider_id,
                service_type=MAP_DATA_SERVICE,
                trust_score=provider_trust_score,
                negative_review_rate=provider_negative_review_rate,
            )
            return False

        stake_terms = self.evaluate_map_response_stakes(
            provider_id=provider_id,
            response=response,
            service_type=MAP_DATA_SERVICE,
        )

        if not stake_terms["accepted"]:
            self.record_decision(
                decision_type="accept_map_response",
                reason=stake_terms["rejection_reason"],
                peer_id=provider_id,
                service_type=MAP_DATA_SERVICE,
                trust_score=provider_trust_score,
                negative_review_rate=provider_negative_review_rate,
            )
            return False

        interaction_meta = {
            "goal_name": self.goal_name,
            "shared_coordinate": response["coord"],
            "provider_distance": response["dist"],
            "provider_stake_limit": stake_terms["provider_stake_limit"],
            "provider_stake_required": stake_terms["provider_stake_required"],
            "requester_stake_limit": stake_terms["requester_stake_limit"],
            "requester_stake_required": stake_terms["requester_stake_required"],
            "provider_available_stake": stake_terms["provider_available_stake"],
            "requester_available_stake": stake_terms["requester_available_stake"],
            "provider_willing_to_stake": stake_terms["provider_willing_to_stake"],
            "requester_willing_to_stake": stake_terms["requester_willing_to_stake"],
            "provider_stake_requirement_meta": stake_terms[
                "provider_stake_requirement_meta"
            ],
            "requester_stake_limit_meta": stake_terms["requester_stake_limit_meta"],
            "provider_response_stake_limit_meta": stake_terms[
                "provider_response_stake_limit_meta"
            ],
            "provider_response_requester_requirement_meta": stake_terms[
                "provider_response_requester_requirement_meta"
            ],
        }

        interaction_id = self.model.record_staked_interaction(
            truster_id=self.unique_id,
            trustee_id=provider_id,
            service_type=MAP_DATA_SERVICE,
            truster_stake=stake_terms["requester_stake_to_lock"],
            trustee_stake=stake_terms["provider_stake_to_lock"],
            meta=interaction_meta,
        )

        if interaction_id is None:
            self.record_decision(
                decision_type="accept_map_response",
                reason="stake_lock_failed",
                peer_id=provider_id,
                service_type=MAP_DATA_SERVICE,
                trust_score=provider_trust_score,
                negative_review_rate=provider_negative_review_rate,
            )
            return False

        self.current_provider_id = provider_id
        self.current_interaction_id = interaction_id

        self.update_internal_map(
            coordinate=response["coord"],
            env_type=ENV_DROP_OFF,
            info_source=provider_id,
            drop_off_name=self.goal_name,
        )

        if self._get_goal_coordinate(self.goal_name) is None:
            # The agent already has verified knowledge that this coordinate is
            # not a drop-off, so the provider's info is provably wrong.
            # Treat this identically to arriving at the wrong destination.
            outcome_meta = self.build_outcome_meta()
            outcome_meta["points_delta"] = 0.0
            outcome_meta["failure_reason"] = "provider_coordinate_immediately_disproved"
            self._settle_current_interaction(
                outcome_status="failure",
                points_delta=0.0,
                outcome_meta=outcome_meta,
            )
            self._remove_drop_off_name(self.goal_name)
            self.current_provider_id = None
            return False

        self.record_decision(
            decision_type="accept_map_response",
            reason="provider_accepted",
            peer_id=provider_id,
            service_type=MAP_DATA_SERVICE,
            trust_score=provider_trust_score,
            negative_review_rate=provider_negative_review_rate,
        )

        return True

    def _start_exploring(self) -> None:
        self.target_coordinate = self.select_unexplored_coordinate()
        self.state = "EXPLORING" if self.target_coordinate is not None else "IDLE"


class DropOffLocationAgent(FixedAgent):
    def __init__(self, model: mesa.Model, cell: mesa.discrete_space.Cell) -> None:
        """
        A fixed agent that represents a drop-off location.

        :param model: The mesa model.
        :param cell: The cell on which the agent spawns.
        """

        super().__init__(model)
        self.cell = cell

        self.points = None
        self.delivery_count = None
        self.global_rep = None


class MaliciousDeliveryAgent(DeliveryAgent):
    def __init__(
        self,
        model: mesa.Model,
        cell: mesa.discrete_space.Cell,
        vision_radius: int = 1,
        trust_reject_threshold: float = 0.3,
        trust_accept_threshold: float = 0.8,
        max_negative_review_rate: float = 0.6,
        min_reviews_before_reviewer_check: int = 10,
        context_match_weight: float = 1.0,
        other_context_weight: float = 0.25,
        trust_prior_active: float = 1.0,
        trust_prior_burned: float = 1.0,
        burned_weight_multiplier: float = 1.0,
        filter_untrusted_evidence: bool = True,
        staking_min_fraction: float = 0.25,
        staking_max_fraction: float = 1.0,
        provider_stake_vtp_weight: float = 0.8,
        provider_stake_reviewer_weight: float = 0.2,
        requester_stake_vtp_weight: float = 0.4,
        requester_stake_reviewer_weight: float = 0.6,
        false_map_probability: float = 0.5,
        false_negative_review_probability: float = 0.5,
        false_positive_review_probability: float = 0.5,
    ) -> None:
        super().__init__(
            model=model,
            cell=cell,
            vision_radius=vision_radius,
            trust_reject_threshold=trust_reject_threshold,
            trust_accept_threshold=trust_accept_threshold,
            max_negative_review_rate=max_negative_review_rate,
            min_reviews_before_reviewer_check=min_reviews_before_reviewer_check,
            context_match_weight=context_match_weight,
            other_context_weight=other_context_weight,
            trust_prior_active=trust_prior_active,
            trust_prior_burned=trust_prior_burned,
            burned_weight_multiplier=burned_weight_multiplier,
            filter_untrusted_evidence=filter_untrusted_evidence,
            staking_min_fraction=staking_min_fraction,
            staking_max_fraction=staking_max_fraction,
            provider_stake_vtp_weight=provider_stake_vtp_weight,
            provider_stake_reviewer_weight=provider_stake_reviewer_weight,
            requester_stake_vtp_weight=requester_stake_vtp_weight,
            requester_stake_reviewer_weight=requester_stake_reviewer_weight,
        )

        self.false_map_probability = self._validate_probability(
            false_map_probability,
            "false_map_probability",
        )
        self.false_negative_review_probability = self._validate_probability(
            false_negative_review_probability,
            "false_negative_review_probability",
        )
        self.false_positive_review_probability = self._validate_probability(
            false_positive_review_probability,
            "false_positive_review_probability",
        )

    @override
    def share_map(self, requester: CellAgent, target: str) -> MapShareResponse | None:
        target_is_known = target in self.known_drop_offs

        if not self.accepts_requester_for_service(requester, MAP_DATA_SERVICE):
            return None

        # false_map_probability governs ALL fabrication — both for known and
        # unknown targets.  This ensures false_map_probability=0.0 is a clean
        # "no map attack" baseline: even when the target is unknown, a
        # non-attacking malicious agent declines honestly (returns None),
        # exactly as an honest agent would.  false_map_probability=1.0 means
        # always fabricate, regardless of whether the target is known.
        if not self._draw_probability(self.false_map_probability):
            # Not lying this interaction: behave like an honest agent.
            if not target_is_known:
                self.record_decision(
                    decision_type="share_map",
                    reason="rejected_unknown_target",
                    peer_id=requester.unique_id,
                    service_type=MAP_DATA_SERVICE,
                )
                return None
            return self.build_map_share_response(
                requester=requester,
                coordinate=self.known_drop_offs[target],
                service_type=MAP_DATA_SERVICE,
            )

        # Lying this interaction: fabricate a random coordinate.
        coordinate = (
            self.random.randint(0, self.model.grid.width - 1),
            self.random.randint(0, self.model.grid.height - 1),
        )
        return self.build_map_share_response(
            requester=requester,
            coordinate=coordinate,
            service_type=MAP_DATA_SERVICE,
        )

    @override
    def decide_reported_outcome(
        self, actual_outcome_status: ReportedOutcomeStatus, outcome_meta: dict[str, Any]
    ) -> tuple[ReportedOutcomeStatus, dict[str, Any]]:
        reported_outcome_status = actual_outcome_status
        review_was_false = False
        review_mode = "honest_review"

        if actual_outcome_status == "success" and self._draw_probability(
            self.false_negative_review_probability
        ):
            reported_outcome_status = "failure"
            review_was_false = True
            review_mode = "false_negative_review"

        elif actual_outcome_status == "failure" and self._draw_probability(
            self.false_positive_review_probability
        ):
            reported_outcome_status = "success"
            review_was_false = True
            review_mode = "false_positive_review"

        review_meta = dict(outcome_meta)
        review_meta.update(
            {
                "actual_outcome_status": actual_outcome_status,
                "reported_outcome_status": reported_outcome_status,
                "review_was_false": review_was_false,
                "review_mode": review_mode,
                "reviewer_id": self.unique_id,
                "reviewer_type": type(self).__name__,
                "false_negative_review_probability": self.false_negative_review_probability,
                "false_positive_review_probability": self.false_positive_review_probability,
            }
        )

        return reported_outcome_status, review_meta

    def _draw_probability(self, probability: float) -> bool:
        return self.random.random() < probability
