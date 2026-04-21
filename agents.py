import mesa
from mesa.discrete_space import CellAgent, FixedAgent
from typing_extensions import override


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
        self.internal_map: dict[tuple[int, int], dict] = {}
        self.known_drop_offs: dict[str, tuple[int, int]] = {}
        self.goal_name = None
        self.prev_goal_name = None
        self.state = "IDLE"
        self.target_coordinate = None
        self.vision_radius = vision_radius
        self.points = 0.0
        self.package = None
        self.current_provider_id = None
        self.delivery_count = 0
        self._all_possible_coords = model.all_coordinates
        self.current_interaction_id = None
        self.cached_active_tnfts = 0
        self.cached_burned_tnfts = 0


    @property
    def map_size(self) -> int:
        return len(self.internal_map)

    @property
    def known_drop_offs_count(self) -> int:
        return len(self.known_drop_offs)

    @property
    def steps_on_package(self) -> int:
        return self.package["steps_taken"] if self.package else 0


    def verify_vtp(self, target_id: str, service_type: str="map_data") -> bool:
        summary = self.model.get_vtp_summary(target_id, service_type)
        return summary["score"] >= 0.5



    # def calculate_trust(self, target_agent_id: str) -> float:
    #     global_rep = self.model.calc_global_trust(target_agent_id)
    #     all_target_tnfts = [nft for nft in self.model.tnft_ledger if nft["receiver"] == target_agent_id]
    #     direct_experiences = [nft for nft in all_target_tnfts if nft["issuer"] == self.unique_id]
    #
    #     if not direct_experiences:
    #         return global_rep
    #     else:
    #         pos_direct = sum(1 for nft in direct_experiences if nft["positive"])
    #         local_trust = float(pos_direct/max(3, len(direct_experiences)))
    #
    #         blended_trust = (0.7 * local_trust) + (0.3 * global_rep)
    #         return blended_trust


    def move(self) -> bool:
        """
        Move towards the internal target coordinate.
        Can move 1 cell in one of 8 directions at a time.
        """
        if self.target_coordinate is None:
            return False

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

        moved = dx != 0 or dy != 0
        if moved:
            self.move_relative((dx, dy))

        # could maybe check if the agent is on cell from the get-go?
        # would need to change a bunch of stuff tho

        if self.cell.coordinate == self.target_coordinate:
            agents_on_cell = [a.unique_id for a in self.cell.agents]

            # if the delivery location is here
            if self.state == "DELIVERING":
                if self.goal_name in agents_on_cell:
                    success = self.model.verify_delivery(self, self.package)

                    if success:
                        self.delivery_count += 1

                        if self.current_interaction_id is not None:
                            self.model.settle_interaction(
                                interaction_id=self.current_interaction_id,
                                evaluator_id=self.unique_id,
                                outcome_status="success",
                                outcome_meta={"goal_name": self.goal_name}
                            )

                        self.prev_goal_name = self.goal_name
                        self.goal_name = None
                        self.package = None

                else:
                    if self.current_interaction_id is not None:
                        self.model.settle_interaction(
                            interaction_id=self.current_interaction_id,
                            evaluator_id=self.unique_id,
                            outcome_status="failure",
                            outcome_meta={"goal_name": self.goal_name},
                        )

                    if self.goal_name in self.known_drop_offs:
                        del self.known_drop_offs[self.goal_name]

                    if self.target_coordinate in self.internal_map:
                        del self.internal_map[self.target_coordinate]

            self.state = "IDLE"
            self.target_coordinate = None
            self.current_provider_id = None
            self.current_interaction_id = None

        return moved


    def update_internal_map(self, coordinate: tuple[int, int], env_type: str, info_source: str="self", drop_off_name: str|None=None) -> None:
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


    def share_map(self, requester: CellAgent, target: str) -> None | tuple[int, int]:
        if self.verify_vtp(requester.unique_id) and target in self.known_drop_offs:
            return self.known_drop_offs[target]

        return None


    def perceive_env(self) -> None:
        """
        Check area visible in vision range and update internal map.
        """

        visible_area = self.cell.get_neighborhood(
            include_center=True,
            radius=self.vision_radius,
        ).cells

        for cell in visible_area:
            drop_off = next(
                (agent for agent in cell.agents if isinstance(agent, DropOffLocationAgent)),
                None
            )

            if drop_off is not None:
                self.update_internal_map(cell.coordinate, "drop_off", drop_off_name=drop_off.unique_id)
            else:
                self.update_internal_map(cell.coordinate, "floor")


    def receive_package(self, package: dict) -> None:
        """
        Set new goal location.

        :param package: The new goal location and the amount of time before expiration.
        """
        self.package = package
        self.goal_name = self.package["destination"]
        self.package["steps_taken"] = 0


    def select_unexplored_coordinate(self) -> None|tuple[int, int]:
        """
        Randomly select an unexplored coordinate.
        If there are no unexplored coordinates, return None.

        :return: None or coordinate
        """

        # all_possible_coordinates = set((x,y) for x in range(self.model.grid.width) for y in range(self.model.grid.height))
        explored_coordinates = set(self.internal_map.keys())
        unexplored_coordinates = tuple(self._all_possible_coords - explored_coordinates)

        if not unexplored_coordinates:
            return None
        return self.random.choice(unexplored_coordinates)


    def step(self) -> None:
        self.perceive_env()

        if self.goal_name is None or self.package is None:
            return

        if self.goal_name in self.known_drop_offs and self.state != "DELIVERING":
            self.target_coordinate = self.known_drop_offs[self.goal_name]
            self.state = "DELIVERING"

        elif self.state == "IDLE" or (self.state == "EXPLORING" and (self.target_coordinate is None or self.target_coordinate in self.internal_map)):
            responses = self.model.request_map_data(self, self.goal_name)
            success = False

            for response in responses:
                provider_id = response["agent"]

                if self.verify_vtp(provider_id):
                    self.update_internal_map(response["coord"], "drop_off", response["agent"], self.goal_name)

                    if self.goal_name in self.known_drop_offs:
                        self.current_provider_id = provider_id
                        self.current_interaction_id = self.model.record_interaction(
                            truster_id=self.unique_id,
                            trustee_id=provider_id,
                            service_type="map_data",
                            meta={"goal_name": self.goal_name, "shared_coordinate": response["coord"]},
                        )
                        success = True
                        break

            if success:
                self.target_coordinate = self.known_drop_offs[self.goal_name]
                self.state = "DELIVERING"
            else:
                self.target_coordinate = self.select_unexplored_coordinate()
                self.state = "EXPLORING" if self.target_coordinate is not None else "IDLE"

        if self.target_coordinate is not None:
            moved = self.move()
            if moved and self.package is not None:
                self.package["steps_taken"] += 1


class DropOffLocationAgent(FixedAgent):
    def __init__(self, model: mesa.Model, cell: mesa.discrete_space.Cell) -> None:
        """
        A fixed agent that represents a drop-off location.

        :param model: The mesa model.
        :param cell: The cell on which the agent spawns.
        """

        super().__init__(model)
        self.cell = cell

        self.points = None
        self.delivery_count = None
        self.global_rep = None



class MaliciousMapDeliveryAgent(DeliveryAgent):
    def __init__(self, model: mesa.Model, cell: mesa.discrete_space.Cell, vision_radius: int, maliciousness: float=0.5) -> None:
        super().__init__(model, cell, vision_radius)
        self.maliciousness = maliciousness

    @override
    def share_map(self, requester: CellAgent, target: str) -> None | tuple[int, int]:
        # if self.model.calc_global_trust(self.unique_id) >= 0.5 and self.random.random() <= self.maliciousness:
        if self.verify_vtp(requester.unique_id):
            if self.random.random() <= self.maliciousness:
                return self.random.randint(0, self.model.grid.width-1), self.random.randint(0, self.model.grid.height-1)

        return super().share_map(requester, target)
