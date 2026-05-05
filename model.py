import mesa
from mesa.discrete_space import OrthogonalMooreGrid, CellAgent
from dataclasses import dataclass, field, asdict
from typing import Literal, Any
from agents import DeliveryAgent, DropOffLocationAgent, MaliciousMapDeliveryAgent
import os
import json
import uuid


Coordinate = tuple[int, int]
InteractionStatus = Literal["pending", "completed", "cancelled"]
OutcomeStatus = Literal["success", "failure", "disputed"]

MAP_DATA_SERVICE = "map_data"
SYSTEM_ISSUER_ID = "SYSTEM"
EXPORT_DIR = "exports"

CONTEXT_MATCH_WEIGHT = 1.0
OTHER_CONTEXT_WEIGHT = 0.25

BASE_DELIVERY_POINTS = 10.0
GRACE_WINDOW_RATIO = 0.3
LATE_PENALTY_PER_STEP = 1.0


def get_agent_type(agent: CellAgent) -> str:
    return type(agent).__name__


def get_ledger_size(model: mesa.Model) -> int:
    return len(model.tnft_ledger)



def get_active_ledger_size(model: mesa.Model) -> int:
    return sum(1 for tnft in model.tnft_ledger if tnft["status"])


def get_burned_ledger_size(model: mesa.Model) -> int:
    return sum(1 for tnft in model.tnft_ledger if not tnft["status"])


def get_total_points(model: mesa.Model) -> float:
    return float(sum(agent.points for agent in model.cached_delivery_agents))


def get_total_deliveries(model: mesa.Model) -> int:
    return sum(agent.delivery_count for agent in model.cached_delivery_agents)


def get_pending_interactions(model: mesa.Model) -> int:
    return sum(1 for record in model.interactions.values() if record.status == "pending")


def get_completed_interactions(model: mesa.Model) -> int:
    return sum(1 for record in model.interactions.values() if record.status == "completed")


def get_cancelled_interactions(model: mesa.Model) -> int:
    return sum(1 for record in model.interactions.values() if record.status == "cancelled")


def get_success_outcomes(model: mesa.Model) -> int:
    return sum(1 for record in model.outcomes.values() if record.status == "success")


def get_failure_outcomes(model: mesa.Model) -> int:
    return sum(1 for record in model.outcomes.values() if record.status == "failure")


def get_disputed_outcomes(model: mesa.Model) -> int:
    return sum(1 for record in model.outcomes.values() if record.status == "disputed")


def get_delivery_events(model: mesa.Model) -> int:
    return len(model.delivery_events)


def get_successful_delivery_events(model: mesa.Model) -> int:
    return sum(1 for event in model.delivery_events if event["outcome_status"] == "success")


def get_failed_delivery_events(model: mesa.Model) -> int:
    return sum(1 for event in model.delivery_events if event["outcome_status"] != "success")


def get_agent_map_size(agent: CellAgent) -> int:
    return getattr(agent, "map_size", 0)


def get_agent_current_provider_id(agent: CellAgent) -> str:
    return getattr(agent, "current_provider_id", None) or ""


def get_agent_goal_name(agent: CellAgent) -> str:
    return getattr(agent, "goal_name", None) or ""


def get_agent_cell_x(agent: CellAgent) -> int | None:
    return agent.cell.coordinate[0] if getattr(agent, "cell", None) is not None else None


def get_agent_cell_y(agent: CellAgent) -> int | None:
    return agent.cell.coordinate[1] if getattr(agent, "cell", None) is not None else None


def get_agent_target_x(agent: CellAgent) -> int | None:
    target = getattr(agent, "target_coordinate", None)
    return target[0] if target is not None else None


def get_agent_target_y(agent: CellAgent) -> int | None:
    target = getattr(agent, "target_coordinate", None)
    return target[1] if target is not None else None


def get_agent_package_min_steps(agent: CellAgent) -> int | None:
    package = getattr(agent, "package", None)
    return package["min_steps"] if package is not None else None


def get_agent_package_max_steps(agent: CellAgent) -> int | None:
    package = getattr(agent, "package", None)
    return package["max_steps"] if package is not None else None


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


class BintWorldModel(mesa.Model):
    """Mesa simulation for BINT."""

    def __init__(
            self,
            num_drop_offs: int=5,
            agent_counts: dict[type,int]|None=None,
            num_delivery: int=5,
            num_map_malicious: int=2,
            size: Coordinate|None=None,
            width: int=100,
            height: int=100,
            agent_vision_radius: int=2,
            trust_threshold: float=0.5,
            genesis_tokens: int=3,
            maliciousness_prob: float=0.5,
            max_steps: int=1000,
            scenario_name: str="default",
            scenario_family: str="default",
            export_dir: str=EXPORT_DIR,
            rng: int|str|None=None
    ) -> None:

        self.rng_seed = self._normalise_rng_seed(rng)
        super().__init__(rng=self.rng_seed)

        self.size = size if size is not None else (width, height)
        self.width, self.height = self.size
        self.max_steps = int(max_steps)
        self.scenario_name = str(scenario_name)
        self.scenario_family = str(scenario_family)
        self.export_dir = str(export_dir)

        self.num_drop_offs = int(num_drop_offs)
        self.agent_vision_radius = int(agent_vision_radius)
        self.trust_threshold = float(trust_threshold)
        self.genesis_tokens = int(genesis_tokens)
        self.maliciousness_prob = float(maliciousness_prob)

        self.agent_counts = self._normalise_agent_counts(
            agent_counts=agent_counts,
            num_delivery=num_delivery,
            num_map_malicious=num_map_malicious,
        )
        self.num_delivery = self.agent_counts.get(DeliveryAgent, 0)
        self.num_map_malicious = self.agent_counts.get(MaliciousMapDeliveryAgent, 0)
        self.total_delivery_agents = sum(self.agent_counts.values())

        self._validate_configuration()
        self._create_grid()
        self._initialise_ledger_and_records()
        self._spawn_agents()
        self._cache_agents()

        self.distribute_initial_knowledge()
        self.dispatch_packages()
        self.seed_genesis_tnfts()

        self._initialise_data_collection()


    @staticmethod
    def _normalise_rng_seed(rng: int|str|None) -> int|None:
        if rng is None or rng == "":
            return None
        return int(rng)


    @staticmethod
    def _normalise_agent_counts(
            agent_counts: dict[type, int] | None,
            num_delivery: int,
            num_map_malicious: int,
    ) -> dict[type, int]:
        if agent_counts is None:
            agent_counts = {
                DeliveryAgent: num_delivery,
                MaliciousMapDeliveryAgent: num_map_malicious,
            }

        return {agent_class: int(count) for agent_class, count in agent_counts.items()}


    def _validate_configuration(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Grid width and height must be positive.")
        if self.num_drop_offs < 1:
            raise ValueError("num_drop_offs must be >= 1.")
        if self.agent_vision_radius < 0:
            raise ValueError("agent_vision_radius must be >= 0.")
        if any(count < 0 for count in self.agent_counts.values()):
            raise ValueError("Agent counts must be non-negative.")

        total_cells = self.width * self.height
        required_cells = self.num_drop_offs + self.total_delivery_agents

        if required_cells > total_cells:
            raise ValueError(
                f"Configuration needs {required_cells} distinct cells, but grid only has {total_cells}."
            )


    def _create_grid(self) -> None:
        self.grid = OrthogonalMooreGrid(self.size, torus=False, random=self.random)

        all_cells = list(self.grid.all_cells.cells)
        required_cells = self.num_drop_offs + self.total_delivery_agents
        selected_cells = self.random.sample(all_cells, k=required_cells)

        self.drop_off_cells = selected_cells[:self.num_drop_offs]
        self.agent_spawn_cells = selected_cells[self.num_drop_offs:]
        self.all_coordinates = frozenset(cell.coordinate for cell in all_cells)


    def _initialise_ledger_and_records(self) -> None:
        self.tnft_ledger: list[dict[str, Any]] = []
        self.nft_counter = 0
        self.interaction_counter = 0
        self.interactions: dict[str, InteractionRecord] = {}
        self.outcomes: dict[str, OutcomeRecord] = {}
        self.delivery_events: list[dict[str, Any]] = []


    def _spawn_agents(self) -> None:
        spawn_index = 0

        for agent_class,count in self.agent_counts.items():
            if count <= 0:
                continue

            cells = self.agent_spawn_cells[spawn_index : spawn_index + count]
            common_args = [self, count, cells, self.agent_vision_radius, self.trust_threshold]

            if agent_class is MaliciousMapDeliveryAgent:
                agent_class.create_agents(*common_args, self.maliciousness_prob)
            else:
                agent_class.create_agents(*common_args)

            spawn_index += count

        DropOffLocationAgent.create_agents(self, self.num_drop_offs, self.drop_off_cells)


    def _cache_agents(self) -> None:
        self.cached_drop_offs = [agent for agent in self.agents if isinstance(agent, DropOffLocationAgent)]
        self.cached_delivery_agents = [agent for agent in self.agents if isinstance(agent, DeliveryAgent)]


    def _initialise_data_collection(self) -> None:
        tracking_parameters = {
            "agent_type": get_agent_type,
            "state": "state",
            "points": "points",
            "delivery_count": "delivery_count",
            "active_tnfts": "cached_active_tnfts",
            "burned_tnfts": "cached_burned_tnfts",
            "known_drop_offs": "known_drop_offs_count",
            "map_size": get_agent_map_size,
            "steps_on_package": "steps_on_package",
            "package_min_steps": get_agent_package_min_steps,
            "package_max_steps": get_agent_package_max_steps,
            "goal_name": get_agent_goal_name,
            "current_provider_id": get_agent_current_provider_id,
            "cell_x": get_agent_cell_x,
            "cell_y": get_agent_cell_y,
            "target_x": get_agent_target_x,
            "target_y": get_agent_target_y,
        }

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "ledger_size": get_ledger_size,
                "active_ledger_size": get_active_ledger_size,
                "burned_ledger_size": get_burned_ledger_size,
                "total_points": get_total_points,
                "total_deliveries": get_total_deliveries,
                "pending_interactions": get_pending_interactions,
                "completed_interactions": get_completed_interactions,
                "cancelled_interactions": get_cancelled_interactions,
                "success_outcomes": get_success_outcomes,
                "failure_outcomes": get_failure_outcomes,
                "disputed_outcomes": get_disputed_outcomes,
                "delivery_events": get_delivery_events,
                "successful_delivery_events": get_successful_delivery_events,
                "failed_delivery_events": get_failed_delivery_events,
            },
            agenttype_reporters={
                DeliveryAgent: tracking_parameters,
                MaliciousMapDeliveryAgent: tracking_parameters,
            }
        )
        self.datacollector.collect(self)


    def record_interaction(self, truster_id: str, trustee_id: str, service_type: str, meta: dict[str,Any]|None = None) -> str:
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


    def get_interaction(self, interaction_id: str) -> InteractionRecord|None:
        return self.interactions.get(interaction_id)


    def record_outcome(self, interaction_id: str, status: OutcomeStatus, meta: dict[str, Any]|None = None) -> OutcomeRecord|None:
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

    def settle_interaction(self, interaction_id: str, evaluator_id: str, outcome_status: OutcomeStatus, outcome_meta: dict[str,Any]|None = None) -> OutcomeRecord|None:
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


    def _apply_outcome_to_interaction(self, interaction: InteractionRecord, outcome: OutcomeRecord, evaluator_id: str) -> None:
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


    def _reward_successful_interaction(self, interaction: InteractionRecord, outcome: OutcomeRecord, evaluator_id: str) -> None:
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
                    service_type="bootstrap",
                    interaction_id=None,
                    meta={"bootstrap": True},
                )


    def mint_tnft(self, issuer_id: str, receiver_id: str, interaction_type: str, service_type: str, interaction_id: str|None, meta: dict[str,Any]|None=None) -> int:
        self.nft_counter += 1

        tnft = {
            "id": self.nft_counter,
            "issuer": issuer_id,
            "owner": receiver_id,
            "type": interaction_type,
            "service_type": service_type,
            "interaction_id": interaction_id,
            "metadata": meta or {},
            "status": True, # True means active, False means burned
            "timestamp": self.time
        }
        self.tnft_ledger.append(tnft)

        # update cache
        target_agent = self._find_delivery_agent(receiver_id)
        if target_agent is not None:
            target_agent.cached_active_tnfts += 1

        return tnft["id"]


    def get_vtp(self, agent_id: str,service_type: str|None = None, active_only: bool = True) -> list[dict[str,Any]]:
        tnfts = [t for t in self.tnft_ledger if t["owner"] == agent_id]

        if active_only:
            tnfts = [nft for nft in tnfts if nft["status"]]

        if service_type is not None:
            tnfts = [nft for nft in tnfts if nft["service_type"] == service_type]

        return sorted(tnfts, key=lambda tnft: tnft["timestamp"], reverse=True)


    def get_vtp_summary(self, agent_id: str, service_type: str|None=None) -> dict:
        active_tnfts = self.get_vtp(agent_id, service_type=None, active_only=True)

        if service_type is None:
            context_matching_active_tnfts = active_tnfts
            other_active_tnfts = []
        else:
            context_matching_active_tnfts = [tnft for tnft in active_tnfts if tnft["service_type"] == service_type]
            other_active_tnfts = [tnft for tnft in active_tnfts if tnft["service_type"] != service_type]

        return {
            "agent_id": agent_id,
            "service_type": service_type,
            "total_active": len(active_tnfts),
            "context_matching_active": len(context_matching_active_tnfts),
            "other_active": len(other_active_tnfts),
            "score": self._calculate_trust_score(
                context_matching_tnfts=context_matching_active_tnfts,
                other_tnfts=other_active_tnfts
            ),
            "tnfts": active_tnfts,
            "context_matching": context_matching_active_tnfts,
            "other_tnfts": other_active_tnfts
        }


    @staticmethod
    def _calculate_trust_score(context_matching_tnfts: list[dict[str, Any]], other_tnfts: list[dict[str, Any]]) -> float:
        return CONTEXT_MATCH_WEIGHT * len(context_matching_tnfts) + OTHER_CONTEXT_WEIGHT * len(other_tnfts)


    def query_vtp(self, agent_id: str) -> int:
        """
        Keeping it for backwards compatibility.
        """
        return self.get_vtp_summary(agent_id)["total_active"]


    def burn_tnft(self, burner_id: str, target_id: str, service_type: str|None = None) -> bool:
        active_tnfts = [t for t in self.tnft_ledger if t["owner"] == target_id and t["status"]]

        if service_type is not None:
            active_tnfts = [t for t in active_tnfts if t["service_type"] == service_type]

        if not active_tnfts:
            return False

        active_tnfts.sort(key=lambda t: (t["type"] != "bootstrap", t["id"]))
        tnft_to_burn = active_tnfts[0]

        tnft_to_burn["status"] = False
        tnft_to_burn["burned_by"] = burner_id
        tnft_to_burn["burn_timestamp"] = self.time

        target_agent = self._find_delivery_agent(target_id)
        if target_agent is not None:
            target_agent.cached_active_tnfts = max(0, target_agent.cached_active_tnfts - 1)
            target_agent.cached_burned_tnfts += 1

        return  True


    def _find_delivery_agent(self, agent_id: str) -> DeliveryAgent | None:
        return next((agent for agent in self.cached_delivery_agents if agent.unique_id == agent_id), None)


    def get_drop_off_coordinate(self, drop_off_name: str | None) -> Coordinate | None:
        if drop_off_name is None:
            return None

        drop_off = next((agent for agent in self.cached_drop_offs if agent.unique_id == drop_off_name), None)
        return drop_off.cell.coordinate if drop_off is not None else None


    def is_true_drop_off_coordinate(self, drop_off_name: str | None, coordinate: Coordinate | None) -> bool:
        true_coordinate = self.get_drop_off_coordinate(drop_off_name)

        if true_coordinate is None or coordinate is None:
            return False

        return tuple(coordinate) == tuple(true_coordinate)


    def _agent_snapshot(self, agent: DeliveryAgent) -> dict[str, Any]:
        return {
            "agent_id": agent.unique_id,
            "agent_type": type(agent).__name__,
            "is_malicious_type": isinstance(agent, MaliciousMapDeliveryAgent),
            "maliciousness": getattr(agent, "maliciousness", 0.0),
            "cell": agent.cell.coordinate,
            "final_state": agent.state,
            "final_points": agent.points,
            "total_deliveries": agent.delivery_count,
            "active_tnfts": agent.cached_active_tnfts,
            "burned_tnfts": agent.cached_burned_tnfts,
            "known_drop_offs": agent.known_drop_offs_count,
            "map_size": agent.map_size,
        }


    def record_delivery_event(self, agent_id: str, outcome_status: OutcomeStatus, meta: dict[str, Any] | None = None) -> dict[str, Any] | None:
        agent = self._find_delivery_agent(agent_id)

        if agent is None:
            return None

        event_id = f"delivery_{len(self.delivery_events) + 1}"
        event = {
            "delivery_event_id": event_id,
            "timestamp": self.time,
            "agent_id": agent_id,
            "agent_type": type(agent).__name__,
            "outcome_status": outcome_status,
            "provider_id": getattr(agent, "current_provider_id", None),
            "interaction_id": getattr(agent, "current_interaction_id", None),
            "meta": meta or {},
        }
        self.delivery_events.append(event)
        return event


    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


    def request_map_data(self, requester: DeliveryAgent, target_name: str) -> list[dict[str,Any]]:
        responses = []
        true_coordinate = self.get_drop_off_coordinate(target_name)

        for agent in self.cached_delivery_agents:
            if agent == requester:
                continue

            if target_name in agent.known_drop_offs:
                agent_resp = agent.share_map(requester, target_name)

                if agent_resp is None:
                    continue

                provider_summary = self.get_vtp_summary(agent.unique_id, MAP_DATA_SERVICE)
                requester_summary = self.get_vtp_summary(requester.unique_id, MAP_DATA_SERVICE)
                provider_distance = self.chebyshev_distance(requester.cell.coordinate, agent.cell.coordinate)

                responses.append(
                    {
                        # Keep these legacy keys because DeliveryAgent currently consumes them.
                        "agent": agent.unique_id,
                        "dist": provider_distance,
                        "coord": agent_resp,

                        # Rich evaluation metadata.
                        "requester_id": requester.unique_id,
                        "requester_type": type(requester).__name__,
                        "provider_id": agent.unique_id,
                        "provider_type": type(agent).__name__,
                        "target_name": target_name,
                        "shared_coordinate": agent_resp,
                        "shared_coord_x": agent_resp[0],
                        "shared_coord_y": agent_resp[1],
                        "true_coordinate": true_coordinate,
                        "true_coord_x": true_coordinate[0] if true_coordinate is not None else None,
                        "true_coord_y": true_coordinate[1] if true_coordinate is not None else None,
                        "was_shared_coordinate_true": self.is_true_drop_off_coordinate(target_name, agent_resp),
                        "provider_distance": provider_distance,
                        "provider_share_mode": getattr(agent, "last_share_mode", ""),
                        "provider_share_was_malicious": bool(getattr(agent, "last_share_was_malicious", False)),
                        "provider_trust_score_at_request": provider_summary["score"],
                        "provider_total_active_at_request": provider_summary["total_active"],
                        "provider_context_matching_active_at_request": provider_summary["context_matching_active"],
                        "provider_other_active_at_request": provider_summary["other_active"],
                        "requester_trust_score_at_request": requester_summary["score"],
                        "requester_total_active_at_request": requester_summary["total_active"],
                    }
                )

        responses = sorted(responses, key=lambda response: response["dist"])

        for rank, response in enumerate(responses, start=1):
            response["response_rank_by_distance"] = rank
            response["num_candidate_responses"] = len(responses)

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
            possible_destinations = [d for d in self.cached_drop_offs if d.unique_id != agent.prev_goal_name]
            if not possible_destinations:
                continue

            new_destination = self.random.choice(possible_destinations)
            min_steps_to_destination = self.chebyshev_distance(agent.cell.coordinate, new_destination.cell.coordinate)
            max_steps_to_destination = int(min_steps_to_destination * (self.random.betavariate(5, 5) + 1) + 1)

            package = {
                "destination": new_destination.unique_id,
                "max_steps": max_steps_to_destination,
                "min_steps": min_steps_to_destination,
                "steps_taken": 0,
            }
            agent.receive_package(package)


    def _agents_without_packages(self) -> list[DeliveryAgent]:
        return [agent for agent in self.cached_delivery_agents if agent.goal_name is None]


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
        return max(abs(a[0]-b[0]), abs(a[1] - b[1]))


    def step(self) -> None:
        self.agents.shuffle_do("step")
        self.dispatch_packages()
        self.datacollector.collect(self)

        if self.steps == self.max_steps:
            self.export_end_of_run_data()


    def export_end_of_run_data(self) -> None:
        os.makedirs(self.export_dir, exist_ok=True)
        run_uuid = uuid.uuid4().hex[:8]
        safe_scenario = self._safe_name(self.scenario_name)

        export_payload = {
            "run_id": run_uuid,
            "rng_seed": self.rng_seed,
            "scenario": {
                "scenario_name": self.scenario_name,
                "scenario_family": self.scenario_family,
            },
            "config": {
                "size": self.size,
                "width": self.width,
                "height": self.height,
                "num_drop_offs": self.num_drop_offs,
                "num_delivery": self.num_delivery,
                "num_map_malicious": self.num_map_malicious,
                "total_delivery_agents": self.total_delivery_agents,
                "agent_vision_radius": self.agent_vision_radius,
                "trust_threshold": self.trust_threshold,
                "genesis_tokens": self.genesis_tokens,
                "maliciousness_prob": self.maliciousness_prob,
                "max_steps": self.max_steps,
            },
            "total_steps": self.steps,
            "summary": {
                "ledger_size": get_ledger_size(self),
                "active_ledger_size": get_active_ledger_size(self),
                "burned_ledger_size": get_burned_ledger_size(self),
                "total_points": get_total_points(self),
                "total_deliveries": get_total_deliveries(self),
                "pending_interactions": get_pending_interactions(self),
                "completed_interactions": get_completed_interactions(self),
                "cancelled_interactions": get_cancelled_interactions(self),
                "success_outcomes": get_success_outcomes(self),
                "failure_outcomes": get_failure_outcomes(self),
                "delivery_events": get_delivery_events(self),
                "successful_delivery_events": get_successful_delivery_events(self),
                "failed_delivery_events": get_failed_delivery_events(self),
            },

            "drop_offs": [
                {
                    "id": d.unique_id,
                    "coord": d.cell.coordinate,
                    "coord_x": d.cell.coordinate[0],
                    "coord_y": d.cell.coordinate[1],
                } for d in self.cached_drop_offs
            ],

            "agent_snapshots": [
                self._agent_snapshot(agent) for agent in self.cached_delivery_agents
            ],

            "interactions": [asdict(record) for record in self.interactions.values()],
            "outcomes": [asdict(record) for record in self.outcomes.values()],
            "delivery_events": self.delivery_events,
            "tnft_ledger": self.tnft_ledger,
        }

        filename = os.path.join(
            self.export_dir,
            f"{safe_scenario}_seed_{self.rng_seed}_{run_uuid}.json",
        )
        with open(filename, "w") as f:
            json.dump(export_payload, f, indent=4)