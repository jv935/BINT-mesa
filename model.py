import mesa
from mesa.discrete_space import OrthogonalMooreGrid
from agents import DeliveryAgent, DropOffLocationAgent


class BintWorldModel(mesa.Model):
    def __init__(self, num_agents: int=5, width: int=25, height: int=25, num_drop_offs: int=5, agent_vision_radius: int=2, rng: int=None) -> None:
        """
        A model for the implementation of BINT.

        :param num_agents: The number of delivery agents.
        :param width: The width of the grid.
        :param height: The height of the grid.
        :param num_drop_offs: The number of drop-off locations.
        :param agent_vision_radius: The vision radius of the delivery agents.
        :param rng: Random generation seed.
        """

        super().__init__(rng=rng)
        self.num_agents = num_agents
        self.num_drop_offs = num_drop_offs
        self.agent_vision_radius = agent_vision_radius
        self.grid = OrthogonalMooreGrid((width, height), torus=False, random=self.random)

        self.drop_off_cells = self.random.sample(self.grid.all_cells.cells, k=self.num_drop_offs)
        self.agent_spawn_cells = self.random.sample(self.grid.all_cells.cells, k=self.num_agents)

        DeliveryAgent.create_agents(self, self.num_agents, self.agent_spawn_cells, self.agent_vision_radius)
        DropOffLocationAgent.create_agents(self, self.num_drop_offs, self.drop_off_cells)

        self.distribute_initial_knowledge()
        self.dispatch_packages()

        self.datacollector = mesa.DataCollector(
            model_reporters={"Number of Agents": "num_agents"},
            agenttype_reporters={DeliveryAgent: {"Points": "points"}}
        )

        self.datacollector.collect(self)


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
            receiving_agent = delivery_agents[i % self.num_agents]

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
        all_drop_offs = [d.unique_id for d in self.agents.select(agent_type=DropOffLocationAgent)]

        for agent in self.agents.select(agent_type=DeliveryAgent):
            if agent.goal_name is None:
                # do not use previous drop off again
                possible_destinations = [d for d in all_drop_offs if d != agent.prev_goal_name]

                if possible_destinations:
                    new_destination = self.random.choice(possible_destinations)
                    agent.receive_package(new_destination)

    def step(self) -> None:
        self.agents.shuffle_do("step")
        self.dispatch_packages()
        self.datacollector.collect(self)
