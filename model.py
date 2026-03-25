import mesa
from mesa.discrete_space import OrthogonalMooreGrid
from agents import DeliveryAgent, DropOffLocationAgent


class BintWorldModel(mesa.Model):
    def __init__(self, num_agents: int=5, width: int=25, height: int=25, num_drop_offs: int=5, agent_vision_radius: int=2, rng=None):
        super().__init__(rng=rng)
        self.num_agents = num_agents
        self.num_drop_offs = num_drop_offs
        self.agent_vision_radius = agent_vision_radius
        self.grid = OrthogonalMooreGrid((width, height), torus=False, random=self.random)

        self.drop_off_cells = self.random.sample(self.grid.all_cells.cells, k=self.num_drop_offs)
        self.agent_spawn_cells = self.random.sample(self.grid.all_cells.cells, k=self.num_agents)

        DeliveryAgent.create_agents(self, self.num_agents, self.agent_spawn_cells, self.agent_vision_radius)
        DropOffLocationAgent.create_agents(self, self.num_drop_offs, self.drop_off_cells)

        self.dispatch_packages()

        #self.datacollector = DataCollector()

    def dispatch_packages(self):
        all_drop_offs = [d.unique_id for d in self.agents.select(agent_type=DropOffLocationAgent)]

        for agent in self.agents.select(agent_type=DeliveryAgent):
            if agent.goal_name is None:
                possible_destinations = [d for d in all_drop_offs if d != agent.prev_goal_name]

                if possible_destinations:
                    new_destination = self.random.choice(possible_destinations)
                    agent.receive_package(new_destination)

    def step(self):
        self.agents.shuffle_do("step")
        self.dispatch_packages()
