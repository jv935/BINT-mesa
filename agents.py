import mesa
from mesa.discrete_space import CellAgent, FixedAgent

class DeliveryAgent(CellAgent):
    def __init__(self, model: mesa.Model, cell: mesa.discrete_space.Cell):
        super().__init__(model)
        self.cell = cell
        self.goal = None
        self.prev_goal = None
        self.points = 0

    def move(self, target):
        # self.cell = self.cell.neighborhood.select_random_cell()
        current_x, current_y = self.cell.coordinate
        target_x, target_y = target.cell.coordinate

        dx = 0
        if current_x < target_x:
            dx = 1
        elif current_x > target_x:
            dx = -1

        dy = 0
        if current_y < target_y:
            dy = 1
        elif current_y > target_y:
            dy = -1

        #print(f"Current location: {self.cell.coordinate}, Target location: {target.cell.coordinate}, dx: {dx}, dy: {dy}")
        #new_x, new_y = (current_x + dx, current_y + dy)
        if dx != 0 or dy != 0:
            self.move_relative((dx, dy))

        if self.cell.coordinate == target.cell.coordinate:
            self.points += 1
            self.goal = None

    def step(self):
        if self.goal is None:
            while True:
                self.goal = self.random.choice(self.model.agents.select(agent_type=DropOffLocationAgent))
                if self.goal is not self.prev_goal:
                    break

        self.move(self.goal)



class DropOffLocationAgent(FixedAgent):
    def __init__(self, model: mesa.Model, cell: mesa.discrete_space.Cell):
        super().__init__(model)
        self.cell = cell
    #     self.storage = []
    #     self.capacity = 3
    #
    # def get_packages(self):
    #     self.storage.append(
    #         self.random.sample(
    #             self.model.grid.all_cells.cells,
    #             k=self.capacity
    #         )
    #     )
    #
    # def step(self):
    #     if len(self.storage) == 0:
    #         self.get_packages()