import mesa
from mesa.discrete_space import CellAgent, FixedAgent

class DeliveryAgent(CellAgent):
    def __init__(self, model: mesa.Model, cell: mesa.discrete_space.Cell, vision_radius: int):
        super().__init__(model)
        self.cell = cell

        self.internal_map = {}
        self.known_drop_offs = {}

        self.goal_name = None
        self.prev_goal_name = None
        self.state = None
        self.target_coordinate = None

        self.vision_radius = vision_radius
        self.points = 0

    def move(self):
        current_x, current_y = self.cell.coordinate
        target_x, target_y = self.target_coordinate

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

        if self.cell.coordinate == self.target_coordinate:
            if self.state == "MOVING TO TARGET":
                self.points += 1
                self.prev_goal_name = self.goal_name
                self.goal_name = None

            self.state = None
            self.target_coordinate = None

    def perceive_env(self):
        visible_area = self.cell.get_neighborhood(
            include_center=True,
            radius=self.vision_radius,
        ).cells

        for cell in visible_area:
            if cell.is_empty:
                self.internal_map[cell.coordinate] = "floor"

            for agent in cell.agents:
                if isinstance(agent, DropOffLocationAgent):
                    self.internal_map[cell.coordinate] = "drop_off"
                    self.known_drop_offs[agent.unique_id] = cell.coordinate
                    break
                else:
                    self.internal_map[cell.coordinate] = "floor"

    def receive_package(self, new_goal):
        self.goal_name = new_goal

    def select_unexplored_coordinate(self):
        all_possible_coordinates = set((x,y) for x in range(self.model.grid.width) for y in range(self.model.grid.height))
        explored_coordinates = set(self.internal_map.keys())

        unexplored_coordinates = all_possible_coordinates - explored_coordinates

        if unexplored_coordinates:
            return self.random.choice(list(unexplored_coordinates))
        else:
            return None

    def step(self):
        self.perceive_env()

        if self.goal_name is None:
            return

        if self.goal_name in self.known_drop_offs and self.state != "MOVING TO TARGET":
            self.target_coordinate = self.known_drop_offs[self.goal_name]
            self.state = "MOVING TO TARGET"

        elif (self.state == "EXPLORING" and self.target_coordinate in self.internal_map) or self.state is None:
            self.target_coordinate = self.select_unexplored_coordinate()
            self.state = "EXPLORING"

        if self.target_coordinate:
            self.move()



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