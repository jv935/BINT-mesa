import mesa
from mesa.discrete_space import CellAgent, FixedAgent

class DeliveryAgent(CellAgent):
    def __init__(self, model: mesa.Model, cell: mesa.discrete_space.Cell, vision_radius: int) -> None:
        """
        An agent that delivers packages to drop-off locations. Can share map data with other agents.

        :param model: The mesa model.
        :param cell: The cell on which the agent spawns.
        :param vision_radius: The vision radius.
        """

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


    def move(self) -> None:
        """
        Move towards the internal target coordinate.
        Can move 1 cell in one of 8 directions at a time.
        """

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

        if dx != 0 or dy != 0:
            self.move_relative((dx, dy))

        if self.cell.coordinate == self.target_coordinate:
            agents_on_cell = [a.unique_id for a in self.cell.agents]
            # if the delivery location is here
            if self.state == "DELIVERING" and self.goal_name in agents_on_cell:
                self.points += 1
                self.prev_goal_name = self.goal_name
                self.goal_name = None

            self.state = None
            self.target_coordinate = None


    def update_internal_map(self, coordinate: tuple[int], env_type: str, info_source: str="self", drop_off_name: str|None=None) -> None:
        """
        Updates the internal map of the agent.

        :param coordinate: The coordinate to store.
        :param env_type: Can be 'floor' or 'drop_off'.
        :param info_source: Either 'self' or the id of the agent that provided the information.
        :param drop_off_name: The name of the drop-off location.
        """

        # If coordinate already has a record
        if coordinate in self.internal_map:
            existing_source = self.internal_map[coordinate]["source"]

            # Do not overwrite direct map info with indirect info
            if existing_source == "self" and info_source != "self":
                return

        self.internal_map[coordinate] = {
            "type": env_type,
            "source": info_source,
        }

        # If it's a drop-off add it to the drop-off index
        if drop_off_name is not None:
            self.known_drop_offs[drop_off_name] = coordinate


    def perceive_env(self) -> None:
        """
        Check area visible in vision range and update internal map.
        """

        visible_area = self.cell.get_neighborhood(
            include_center=True,
            radius=self.vision_radius,
        ).cells

        for cell in visible_area:
            if cell.is_empty:
                self.update_internal_map(cell.coordinate, "floor")

            for agent in cell.agents:
                if isinstance(agent, DropOffLocationAgent):
                    self.update_internal_map(cell.coordinate, "drop_off", drop_off_name=agent.unique_id)
                    break
                else:
                    self.update_internal_map(cell.coordinate, "floor")


    def receive_package(self, package: dict) -> None:
        """
        Set new goal location.

        :param package: The new goal location and the amount of time before expiration.
        """
        self.goal_name = package["destination"]


    def select_unexplored_coordinate(self) -> None|tuple[int, int]:
        """
        Randomly select an unexplored coordinate.
        If there are no unexplored coordinates, return None.

        :return: None or coordinate
        """

        all_possible_coordinates = set((x,y) for x in range(self.model.grid.width) for y in range(self.model.grid.height))
        explored_coordinates = set(self.internal_map.keys())

        unexplored_coordinates = all_possible_coordinates - explored_coordinates

        if unexplored_coordinates:
            return self.random.choice(list(unexplored_coordinates))
        else:
            return None


    def step(self) -> None:
        self.perceive_env()

        if self.goal_name is None:
            return

        if self.goal_name in self.known_drop_offs.keys() and self.state != "DELIVERING":
            self.target_coordinate = self.known_drop_offs[self.goal_name]
            self.state = "DELIVERING"
        elif self.state is None or (self.state == "EXPLORING" and self.target_coordinate in self.internal_map):
            success = self.model.request_map_data(self, self.goal_name)

            if success:
                self.target_coordinate = self.known_drop_offs[self.goal_name]
                self.state = "DELIVERING"
            else:
                self.target_coordinate = self.select_unexplored_coordinate()
                self.state = "EXPLORING"

        if self.target_coordinate:
            self.move()



class DropOffLocationAgent(FixedAgent):
    def __init__(self, model: mesa.Model, cell: mesa.discrete_space.Cell) -> None:
        """
        A fixed agent that represents a drop-off location.

        :param model: The mesa model.
        :param cell: The cell on which the agent spawns.
        """

        super().__init__(model)
        self.cell = cell