import mesa
from mesa.discrete_space import OrthogonalMooreGrid
from dataclasses import dataclass, field, asdict
from typing import Literal
from agents import DeliveryAgent, DropOffLocationAgent, MaliciousMapDeliveryAgent


def get_agent_type(agent):
    return type(agent).__name__

def get_ledger_size(model):
    return len(model.tnft_ledger)


@dataclass
class InteractionRecord:
    interaction_id: str
    truster_id: str
    trustee_id: str
    service_type: str
    meta: dict = field(default_factory=dict)
    timestamp: float = 0.0
    status: Literal["pending", "completed", "cancelled"] = "pending"


@dataclass
class OutcomeRecord:
    interaction_id: str
    status: Literal["success", "failure"]
    meta: dict = field(default_factory=dict)
    timestamp: float = 0.0


class BintWorldModel(mesa.Model):
    def __init__(
            self,
            num_drop_offs: int=5,
            agent_counts: dict=None,
            num_delivery: int=5,
            num_map_malicious: int=2,
            size: tuple[int, int]=None,
            width: int=100,
            height: int=100,
            agent_vision_radius: int=2,
            bootstrap_tnfts_per_agent: int=0,
            trust_threshold: float=0.5,
            trust_mode: str="bint", # bint or none
            maliciousness: float=0.5,
            rng: int|str=None,
    ) -> None:
        """
        A model for the implementation of BINT.

        :param agent_counts: A dictionary storing the type of agent along with the amount.
        :param size: The width and height of the grid.
        :param num_drop_offs: The number of drop-off locations.
        :param agent_vision_radius: The vision radius of the delivery agents.
        :param rng: Random generation seed.
        """

        super().__init__(rng=(int(rng) if rng is not None and rng != "" else None))

        self.size = size if size is not None else (width, height)
        self.width, self.height = self.size
        self.bootstrap_tnfts_per_agent = bootstrap_tnfts_per_agent
        self.trust_threshold = trust_threshold
        self.trust_mode = trust_mode
        self.maliciousness = maliciousness

        if self.bootstrap_tnfts_per_agent < 0:
            raise ValueError("bootstrap_tnfts_per_agent must be >= 0.")
        if not (0.0 <= self.trust_threshold):
            raise ValueError("trust_threshold must be >= 0.")
        if self.trust_mode not in {"bint", "none"}:
            raise ValueError("trust_mode must be 'bint' or 'none'.")
        if not (0.0 <= self.maliciousness <= 1.0):
            raise ValueError("maliciousness must be between 0 and 1.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Grid width and height must both be > 0.")
        if num_drop_offs < 1:
            raise ValueError("num_drop_offs must be at least 1.")
        if agent_vision_radius < 0:
            raise ValueError("agent_vision_radius must be >= 0.")

        if agent_counts is None:
            agent_counts = {
                DeliveryAgent:num_delivery,
                MaliciousMapDeliveryAgent: num_map_malicious,
            }

        self.agent_counts = {cls: int(count) for cls, count in agent_counts.items()}
        if any(count < 0 for count in self.agent_counts.values()):
            raise ValueError("Agent counts cannot be negative.")

        self.num_drop_offs = int(num_drop_offs)
        self.agent_vision_radius = int(agent_vision_radius)
        self.total_delivery_agents = sum(self.agent_counts.values())

        total_cells = self.width * self.height
        required_distinct_cells = self.num_drop_offs + self.total_delivery_agents
        if required_distinct_cells > total_cells:
            raise ValueError(
                f"Configuration needs {required_distinct_cells} distinct cells, "
                f"but grid only has {total_cells}."
            )

        self.grid = OrthogonalMooreGrid(self.size, torus=False, random=self.random)

        all_cells = list(self.grid.all_cells.cells)
        selected_cells = self.random.sample(all_cells, k=required_distinct_cells)

        self.drop_off_cells = selected_cells[: self.num_drop_offs]
        self.agent_spawn_cells = selected_cells[self.num_drop_offs :]
        self.all_coordinates = frozenset(cell.coordinate for cell in all_cells)

        self.tnft_ledger: list[dict] = []
        self.nft_counter = 0
        self.interaction_counter = 0
        self.interactions: dict[str, InteractionRecord] = {}
        self.outcomes: dict[str, OutcomeRecord] = {}

        spawn_idx = 0
        for AgentClass, count in self.agent_counts.items():
            if count <= 0:
                continue
            cells_for_current_class = self.agent_spawn_cells[spawn_idx : spawn_idx + count]
            AgentClass.create_agents(self, count, cells_for_current_class, self.agent_vision_radius)
            spawn_idx += count

        DropOffLocationAgent.create_agents(self, self.num_drop_offs, self.drop_off_cells)

        # DeliveryAgent.create_agents(self, self.num_agents, self.agent_spawn_cells[:num_agents], self.agent_vision_radius)
        # MaliciousMapDeliveryAgent.create_agents(self, self.num_agents, self.agent_spawn_cells[num_agents:], self.agent_vision_radius)
        # DropOffLocationAgent.create_agents(self, self.num_drop_offs, self.drop_off_cells)

        # cache the agent lists since they never die or spawn mid-simulation
        self.cached_drop_offs = [a for a in self.agents if isinstance(a, DropOffLocationAgent)]
        self.cached_delivery_agents = [a for a in self.agents if isinstance(a, DeliveryAgent)]

        for agent in self.cached_delivery_agents:
            if isinstance(agent, MaliciousMapDeliveryAgent):
                agent.maliciousness = self.maliciousness

        self.distribute_initial_knowledge()
        self.dispatch_packages()
        self.seed_genesis_tnfts()

        tracking_parameters = {
            "Agent Type": get_agent_type,
            "State": "state",
            "Points": "points",
            "Deliveries": "delivery_count",
            "Active TNFTs": "cached_active_tnfts",
            "Map Size": "map_size",
            "Known Drop-Offs": "known_drop_offs_count",
            "Steps on Package": "steps_on_package"
        }

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Ledger Size": get_ledger_size,
            },
            agenttype_reporters={
                DeliveryAgent: tracking_parameters,
                MaliciousMapDeliveryAgent: tracking_parameters
            }
        )

        self.datacollector.collect(self)


    def record_interaction(self, truster_id: str, trustee_id: str, service_type: str, meta: dict|None = None) -> str:
        self.interaction_counter += 1
        interaction_id = f"interaction_{self.interaction_counter}"

        record = InteractionRecord(
            interaction_id=interaction_id,
            truster_id=truster_id,
            trustee_id=trustee_id,
            service_type=service_type,
            meta=meta or {},
            timestamp=self.time,
            status="pending",
        )

        self.interactions[interaction_id] = record
        return interaction_id


    def get_interaction(self, interaction_id: str) -> InteractionRecord|None:
        return self.interactions.get(interaction_id)


    def record_outcome(self, interaction_id: str, status: Literal["success", "failure"], meta: dict|None = None) -> OutcomeRecord|None:
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

    def settle_interaction(self, interaction_id: str, evaluator_id: str, outcome_status: Literal["success", "failure"], outcome_meta: dict|None = None) -> OutcomeRecord|None:
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

        if outcome.status == "success":
            self.mint_tnft(
                issuer_id=evaluator_id,
                receiver_id=interaction.trustee_id,
                interaction_type="reward",
                service_type=interaction.service_type,
                interaction_id=interaction_id,
                meta={
                    "interaction_metadata": dict(interaction.meta),
                    "outcome_metadata": dict(outcome.meta),
                },
            )
            interaction.status = "completed"

        elif outcome.status == "failure":
            burned = self.burn_tnft(
                burner_id=evaluator_id,
                target_id=interaction.trustee_id,
                service_type=interaction.service_type,
            )
            interaction.status = "completed" if burned else "cancelled"

        else:  # disputed
            interaction.status = "cancelled"

        return outcome


    def seed_genesis_tnfts(self) -> None:
        if not self.cached_delivery_agents or self.bootstrap_tnfts_per_agent <= 0:
            return

        for agent in self.cached_delivery_agents:
            for _ in range(self.bootstrap_tnfts_per_agent):
                self.mint_tnft(
                    issuer_id="SYSTEM",
                    receiver_id=agent.unique_id,
                    interaction_type="bootstrap",
                    service_type="map_data",
                    interaction_id=None,
                    meta={"bootstrap": True},
                )


    def mint_tnft(self, issuer_id: str, receiver_id: str, interaction_type: str, service_type: str, interaction_id: str|None, meta: dict|None=None) -> int:
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
        target_agent = next((a for a in self.cached_delivery_agents if a.unique_id == receiver_id), None)
        if target_agent is not None:
            target_agent.cached_active_tnfts += 1

        return tnft["id"]


    def get_vtp(self, agent_id: str,service_type: str|None = None, active_only: bool = True) -> list[dict]:
        tnfts = [t for t in self.tnft_ledger if t["owner"] == agent_id]

        if active_only:
            tnfts = [nft for nft in tnfts if nft["status"]]

        if service_type is not None:
            tnfts = [nft for nft in tnfts if nft["service_type"] == service_type]

        tnfts.sort(key=lambda t: t["timestamp"], reverse=True)
        return tnfts


    def get_vtp_summary(self, agent_id: str, service_type: str|None=None) -> dict:
        tnfts = self.get_vtp(agent_id, service_type=service_type, active_only=True)

        earned_tnfts = [t for t in tnfts if t["type"] != "bootstrap"]
        bootstrap_tnfts = [t for t in tnfts if t["type"] == "bootstrap"]

        # simple scoring for now
        score = (1.0 * len(earned_tnfts)) + (0.25 * len(bootstrap_tnfts))

        return {
            "agent_id": agent_id,
            "service_type": service_type,
            "total_active": len(tnfts),
            "earned_active": len(earned_tnfts),
            "bootstrap_active": len(bootstrap_tnfts),
            "score": score,
            "tnfts": tnfts,
        }


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

        target_agent = next((a for a in self.cached_delivery_agents if a.unique_id == target_id), None)
        if target_agent is not None:
            target_agent.cached_active_tnfts = max(0, target_agent.cached_active_tnfts - 1)
            target_agent.cached_burned_tnfts += 1

        return  True


    def request_map_data(self, requester: DeliveryAgent, target_name: str) -> list:
        responses = []

        for agent in self.cached_delivery_agents:
            if agent == requester:
                continue

            if target_name in agent.known_drop_offs:
                agent_resp = agent.share_map(requester, target_name)

                if agent_resp is None:
                    continue

                dist = self.chebyshev_distance(requester.cell.coordinate, agent.cell.coordinate)
                responses.append({"agent": agent.unique_id, "dist": dist, "coord": agent_resp})

        responses.sort(key=lambda x: x["dist"])

        return responses

        # for response in responses:
        #     # Always accept for now
        #     requester.update_internal_map(response["coord"], "drop_off", response["agent"], target_name)
        #     return True
        #
        # return False


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

        for agent in self.cached_delivery_agents:
            if agent.goal_name is not None:
                continue

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


    def verify_delivery(self, agent: DeliveryAgent, package: dict):
        base_points = 1.0
        agents_on_cell = [a.unique_id for a in agent.cell.agents]

        if agent.goal_name in agents_on_cell:
            min_steps = package["min_steps"]
            max_steps = package["max_steps"]
            steps_taken = package["steps_taken"]

            window = max_steps - min_steps
            grace_steps = min_steps + int(window * 0.3)

            if steps_taken <= grace_steps:
                points_awarded = base_points

            elif steps_taken <= max_steps:
                if max_steps > grace_steps:
                    decay_ratio = (max_steps - steps_taken) / (max_steps - grace_steps)
                    points_awarded = float(base_points * decay_ratio)
                else:
                    points_awarded = 0.0

            else:
                lateness = steps_taken - max_steps
                points_awarded = max(-base_points * 2, float(-lateness * 0.5))

            agent.points += points_awarded

            return True

        return False


    @staticmethod
    def chebyshev_distance(a: tuple, b: tuple) -> int:
        return max(abs(a[0]-b[0]), abs(a[1] - b[1]))


    def step(self) -> None:
        self.agents.shuffle_do("step")
        self.dispatch_packages()
        self.datacollector.collect(self)
