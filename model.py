import mesa
from mesa.discrete_space import OrthogonalMooreGrid
from dataclasses import dataclass, field
from typing import Literal, Any
from agents import (
    DeliveryAgent,
    DropOffLocationAgent,
    MaliciousDeliveryAgent,
)

Coordinate = tuple[int, int]
InteractionStatus = Literal["pending", "completed", "cancelled"]
OutcomeStatus = Literal["success", "failure", "disputed"]

MAP_DATA_SERVICE = "map_data"
SYSTEM_ISSUER_ID = "SYSTEM"

# Trust uses a bounded Beta-style score:
# active TNFTs are positive evidence, burned TNFTs are negative evidence.
# Context-matching evidence counts more than evidence from other services.
CONTEXT_MATCH_WEIGHT = 1.0
OTHER_CONTEXT_WEIGHT = 0.25
TRUST_PRIOR_ACTIVE = 1.0
TRUST_PRIOR_BURNED = 1.0
BOOTSTRAP_SERVICE = "bootstrap"

BASE_DELIVERY_POINTS = 10.0
GRACE_WINDOW_RATIO = 0.5
LATE_PENALTY_PER_STEP = 1.0


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
        rng: int | str | None = None,
    ) -> None:

        self.rng_seed = self._normalise_rng_seed(rng)
        super().__init__(rng=self.rng_seed)

        self.size = size if size is not None else (width, height)
        self.width, self.height = self.size
        self.max_steps = int(max_steps)

        self.num_drop_offs = int(num_drop_offs)
        self.genesis_tokens = int(genesis_tokens)

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
            self._reward_successful_interaction(interaction, outcome, evaluator_id)
            interaction.status = "completed"
            return

        if outcome.status == "failure":
            burned = self.burn_tnft(
                burner_id=evaluator_id,
                target_id=interaction.trustee_id,
                service_type=interaction.service_type,
            )
            interaction.status = "completed" if burned else "cancelled"
            return

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
        }
        self.tnft_ledger.append(tnft)

        # update cache
        target_agent = self._find_delivery_agent(receiver_id)
        if target_agent is not None:
            target_agent.cached_active_tnfts += 1

        return tnft["id"]

    def get_vtp(
        self, agent_id: str, service_type: str | None = None, active_only: bool = True
    ) -> list[dict[str, Any]]:
        tnfts = [t for t in self.tnft_ledger if t["owner"] == agent_id]

        if active_only:
            tnfts = [nft for nft in tnfts if nft["status"]]

        if service_type is not None:
            tnfts = [nft for nft in tnfts if nft["service_type"] == service_type]

        return sorted(tnfts, key=lambda tnft: tnft["timestamp"], reverse=True)

    def get_vtp_summary(self, agent_id: str, service_type: str | None = None) -> dict:
        active_tnfts = self.get_vtp(agent_id, service_type=None, active_only=True)
        burned_tnfts = [
            tnft
            for tnft in self.get_vtp(agent_id, service_type=None, active_only=False)
            if not tnft["status"]
        ]

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

        weighted_active = self._calculate_weighted_tnfts(
            context_matching_tnfts=context_matching_active_tnfts,
            other_tnfts=other_active_tnfts,
        )
        weighted_burned = self._calculate_weighted_tnfts(
            context_matching_tnfts=context_matching_burned_tnfts,
            other_tnfts=other_burned_tnfts,
        )
        score = self._calculate_trust_score(
            weighted_active=weighted_active,
            weighted_burned=weighted_burned,
        )

        return {
            "agent_id": agent_id,
            "service_type": service_type,
            "total_active": len(active_tnfts),
            "context_matching_active": len(context_matching_active_tnfts),
            "other_active": len(other_active_tnfts),
            "total_burned": len(burned_tnfts),
            "context_matching_burned": len(context_matching_burned_tnfts),
            "other_burned": len(other_burned_tnfts),
            "weighted_active": weighted_active,
            "weighted_burned": weighted_burned,
            "score": score,
            "tnfts": active_tnfts,
            "burned_tnfts": burned_tnfts,
            "context_matching": context_matching_active_tnfts,
            "other_tnfts": other_active_tnfts,
            "context_matching_burned_tnfts": context_matching_burned_tnfts,
            "other_burned_tnfts": other_burned_tnfts,
        }

    def get_reviewer_summary(self, reviewer_id: str) -> dict[str, Any]:
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

        return {
            "reviewer_id": reviewer_id,
            "total_reviews": total_reviews,
            "positive_reviews": positive_reviews,
            "negative_reviews": negative_reviews,
            "disputed_reviews": disputed_reviews,
            "negative_review_rate": negative_review_rate,
        }

    @staticmethod
    def _tnft_weight(tnft: dict[str, Any]) -> float:
        """Return the reputation weight of a TNFT.

        This currently returns 1.0 for every token. Keeping it as a helper makes
        it easy to add time decay later without changing the trust-score logic.
        """
        return 1.0

    def _calculate_weighted_tnfts(
        self,
        context_matching_tnfts: list[dict[str, Any]],
        other_tnfts: list[dict[str, Any]],
    ) -> float:
        context_weight = sum(self._tnft_weight(tnft) for tnft in context_matching_tnfts)
        other_weight = sum(self._tnft_weight(tnft) for tnft in other_tnfts)

        return (
            CONTEXT_MATCH_WEIGHT * context_weight + OTHER_CONTEXT_WEIGHT * other_weight
        )

    @staticmethod
    def _calculate_trust_score(weighted_active: float, weighted_burned: float) -> float:
        return (TRUST_PRIOR_ACTIVE + weighted_active) / (
            TRUST_PRIOR_ACTIVE + TRUST_PRIOR_BURNED + weighted_active + weighted_burned
        )

    def query_vtp(self, agent_id: str) -> int:
        """Return the legacy active-token count used by older callers.

        New code should prefer `get_vtp_summary()`, which returns a bounded trust
        score and separates active and burned evidence.
        """
        return self.get_vtp_summary(agent_id)["total_active"]

    def burn_tnft(
        self, burner_id: str, target_id: str, service_type: str | None = None
    ) -> bool:
        active_tnfts = [
            tnft
            for tnft in self.tnft_ledger
            if tnft["owner"] == target_id and tnft["status"]
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

        target_agent = self._find_delivery_agent(target_id)
        if target_agent is not None:
            target_agent.cached_active_tnfts = max(
                0, target_agent.cached_active_tnfts - 1
            )
            target_agent.cached_burned_tnfts += 1

        return True

    def _find_delivery_agent(self, agent_id: str) -> DeliveryAgent | None:
        return next(
            (
                agent
                for agent in self.cached_delivery_agents
                if agent.unique_id == agent_id
            ),
            None,
        )

    def get_drop_off_coordinate(self, drop_off_name: str | None) -> Coordinate | None:
        if drop_off_name is None:
            return None

        drop_off = next(
            (
                agent
                for agent in self.cached_drop_offs
                if agent.unique_id == drop_off_name
            ),
            None,
        )
        return drop_off.cell.coordinate if drop_off is not None else None

    def request_map_data(
        self, requester: DeliveryAgent, target_name: str
    ) -> list[dict[str, Any]]:
        responses = []

        for agent in self.cached_delivery_agents:
            if agent == requester:
                continue

            if target_name in agent.known_drop_offs:
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
                        "coord": agent_resp,
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
                env_type="drop_off",
                info_source="system",
                drop_off_name=drop_off.unique_id,
            )

    def dispatch_packages(self) -> None:
        """
        Randomly assign new delivery goals for each agent.
        Will not assign the same goal as lsat time.
        """
        if not self.cached_delivery_agents or not self.cached_drop_offs:
            return

        # get the names of each drop off location

        for agent in self._agents_without_packages():
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
                min_steps_to_destination * (self.random.betavariate(5, 5) + 1) + 1
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
            points_awarded = max((-max_steps * 2), (-lateness * LATE_PENALTY_PER_STEP))

        agent.points += points_awarded
        return True

    @staticmethod
    def chebyshev_distance(a: tuple, b: tuple) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def step(self) -> None:
        self.agents.shuffle_do("step")
        self.dispatch_packages()
