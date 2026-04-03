import mesa
from mesa.discrete_space import OrthogonalMooreGrid
from agents import DeliveryAgent, DropOffLocationAgent, MaliciousMapDeliveryAgent


class BintWorldModel(mesa.Model):
    def __init__(self, num_drop_offs: int=5, agent_counts: dict=None, num_delivery: int=-1, num_map_malicious: int=-1, width: int=50, height: int=50, agent_vision_radius: int=2, rng: int|str=None) -> None:
        """
        A model for the implementation of BINT.

        :param agent_counts: A dictionary storing the type of agent along with the amount.
        :param width: The width of the grid.
        :param height: The height of the grid.
        :param num_drop_offs: The number of drop-off locations.
        :param agent_vision_radius: The vision radius of the delivery agents.
        :param rng: Random generation seed.
        """

        super().__init__(rng=int(rng))

        if agent_counts is None:
            agent_counts = {
                DeliveryAgent: (7 if num_delivery == -1 else num_delivery),
                MaliciousMapDeliveryAgent: (3 if num_map_malicious == -1 else num_map_malicious),
            }

        self.agent_counts = agent_counts
        self.num_drop_offs = num_drop_offs
        self.agent_vision_radius = agent_vision_radius
        self.total_delivery_agents = sum(self.agent_counts.values())

        self.grid = OrthogonalMooreGrid((width, height), torus=False, random=self.random)

        self.drop_off_cells = self.random.sample(self.grid.all_cells.cells, k=self.num_drop_offs)
        self.agent_spawn_cells = self.random.sample(self.grid.all_cells.cells, k=self.total_delivery_agents)

        self.tnft_ledger = []

        spawn_idx = 0
        for AgentClass, count in agent_counts.items():
            if count > 0:
                cells_for_current_class = self.agent_spawn_cells[spawn_idx:spawn_idx+count]
                AgentClass.create_agents(self, count, cells_for_current_class, self.agent_vision_radius)
                spawn_idx += count

        DropOffLocationAgent.create_agents(self, self.num_drop_offs, self.drop_off_cells)

        # DeliveryAgent.create_agents(self, self.num_agents, self.agent_spawn_cells[:num_agents], self.agent_vision_radius)
        # MaliciousMapDeliveryAgent.create_agents(self, self.num_agents, self.agent_spawn_cells[num_agents:], self.agent_vision_radius)
        # DropOffLocationAgent.create_agents(self, self.num_drop_offs, self.drop_off_cells)

        self.distribute_initial_knowledge()
        self.dispatch_packages()
        self.seed_genesis_tnfts()

        tracking_parameters = {
            "Points": "points",
            "Deliveries": "delivery_count",
            "Global Trust": lambda a: self.calc_global_trust(a.unique_id)
        }

        self.datacollector = mesa.DataCollector(
            agenttype_reporters={
                DeliveryAgent: tracking_parameters,
                MaliciousMapDeliveryAgent: tracking_parameters
            }
        )

        self.datacollector.collect(self)


    def seed_genesis_tnfts(self) -> None:
        for agent in self.agents.select(agent_type=DeliveryAgent):
            self.mint_tnft(
                issuer_id="SYSTEM",
                receiver_id=agent.unique_id,
                interaction_type="genesis",
                is_positive=True
            )


    def calc_global_trust(self, agent_id: str) -> float:
        agent_tnfts = [nft for nft in self.tnft_ledger if nft["receiver"] == agent_id]

        if not agent_tnfts:
            return 0.0

        pos_tnfts = sum(1 for nft in agent_tnfts if nft["positive"])
        total_count = max(5, len(agent_tnfts))

        return float(pos_tnfts/total_count)


    def mint_tnft(self, issuer_id: str, receiver_id: str, interaction_type: str, is_positive: bool) -> None:
        tnft = {
            "issuer": issuer_id,
            "receiver": receiver_id,
            "type": interaction_type,
            "positive": is_positive,
            "timestamp": self.time
        }
        self.tnft_ledger.append(tnft)
        # else:
        #     agent_tnfts = [nft for nft in self.tnft_ledger if nft["receiver"] == receiver_id]
        #
        #     if agent_tnfts:
        #         self.random.choice(agent_tnfts)["receiver"] = "null"


    def request_map_data(self, requester: DeliveryAgent, target_name: str) -> list:
        responses = []

        for agent in self.agents.select(agent_type=DeliveryAgent):
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

        drop_offs = self.agents.select(agent_type=DropOffLocationAgent).to_list()
        delivery_agents = self.agents.select(agent_type=DeliveryAgent).to_list()

        self.random.shuffle(delivery_agents)

        for i, drop_off in enumerate(drop_offs):
            receiving_agent = delivery_agents[i % self.total_delivery_agents]

            receiving_agent.update_internal_map(
                coordinate=drop_off.cell.coordinate,
                env_type="drop_off",
                info_source="system",
                drop_off_name=drop_off.unique_id
            )


    def dispatch_packages(self) -> None:
        """
        Randomly assign new delivery goals for each agent.
        Will not assign the same goal as lsat time.
        """

        # get the names of each drop off location
        all_drop_offs = [d for d in self.agents.select(agent_type=DropOffLocationAgent)]

        for agent in self.agents.select(agent_type=DeliveryAgent):
            if agent.goal_name is None:
                # do not use previous drop off again
                possible_destinations = [d for d in all_drop_offs if d.unique_id != agent.prev_goal_name]

                if possible_destinations:
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
                    #self.packages_to_be_delivered[agent.unique_id] = package


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
