import mesa
from mesa.discrete_space import OrthogonalMooreGrid
from agents import DeliveryAgent, DropOffLocationAgent


class BintWorldModel(mesa.Model):
    def __init__(self, num_agents: int=5, width: int=25, height: int=25, num_drop_offs: int=5, rng=None):
        super().__init__(rng=rng)
        self.num_agents = num_agents
        self.num_drop_offs = num_drop_offs
        self.grid = OrthogonalMooreGrid((width, height), torus=False, random=self.random)

        self.drop_off_locations = self.random.sample(self.grid.all_cells.cells, k=self.num_drop_offs)
        # self.grid.create_property_layer("drop_off_locations")
        #
        # for cell in self.drop_off_locations:
        #     self.grid.drop_off_locations.data[cell.coordinate] = 1

        DeliveryAgent.create_agents(self, self.num_agents, self.random.sample(self.grid.all_cells.cells, k=self.num_agents))
        DropOffLocationAgent.create_agents(self, self.num_drop_offs, self.random.sample(self.grid.all_cells.cells, k=self.num_drop_offs))

        #self.datacollector = DataCollector()

    def step(self):
        self.agents.shuffle_do("step")
