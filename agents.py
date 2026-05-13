import mesa
from mesa.discrete_space import CellAgent, FixedAgent
from typing_extensions import override
from typing import Any, Literal, TypedDict

Coordinate = tuple[int, int]
AgentState = Literal["IDLE", "EXPLORING", "DELIVERING"]
ReportedOutcomeStatus = Literal["success", "failure"]

ENV_FLOOR = "floor"
ENV_DROP_OFF = "drop_off"

SOURCE_SELF = "self"
MAP_DATA_SERVICE = "map_data"


class Package(TypedDict):
    destination: str
    max_steps: int
    min_steps: int
    steps_taken: int


class MapRecord(TypedDict):
    type: str
    source: str


class DeliveryAgent(CellAgent):
    def __init__(
        self,
        model: mesa.Model,
        cell: mesa.discrete_space.Cell,
        vision_radius: int = 1,
        trust_reject_threshold: float = 0.3,
        trust_accept_threshold: float = 0.8,
        max_negative_review_rate: float = 0.6,
        min_reviews_before_reviewer_check: int = 3,
    ) -> None:
        """
        An agent that delivers packages to drop-off locations. Can share map data with other agents.

        :param model: The mesa model.
        :param cell: The cell on which the agent spawns.
        :param vision_radius: The vision radius.
        :param trust_reject_threshold: Score at or below which this agent always rejects trust.
        :param trust_accept_threshold: Score at or above which this agent always accepts trust.
        """

        super().__init__(model)
        self.cell = cell
        self.internal_map: dict[Coordinate, MapRecord] = {}
        self.known_drop_offs: dict[str, Coordinate] = {}

        self.goal_name: str | None = None
        self.prev_goal_name: str | None = None
        self.state: AgentState = "IDLE"
        self.target_coordinate: Coordinate | None = None

        self.package: Package | None = None
        self.current_provider_id: str | None = None
        self.current_interaction_id: str | None = None

        self.vision_radius = vision_radius
        self.trust_reject_threshold = trust_reject_threshold
        self.trust_accept_threshold = trust_accept_threshold

        if not 0.0 <= self.trust_reject_threshold < self.trust_accept_threshold <= 1.0:
            raise ValueError(
                "Trust thresholds must satisfy "
                "0.0 <= reject_threshold < accept_threshold <= 1.0."
            )

        self.max_negative_review_rate = self._validate_probability(
            max_negative_review_rate,
            "max_negative_review_rate",
        )
        self.min_reviews_before_reviewer_check = int(min_reviews_before_reviewer_check)

        if self.min_reviews_before_reviewer_check < 0:
            raise ValueError("min_reviews_before_reviewer_check must be >= 0.")

        self.points = 0.0
        self.delivery_count = 0
        self._all_possible_coords = model.all_coordinates
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

    def verify_vtp(self, target_id: str, service_type: str = MAP_DATA_SERVICE) -> bool:
        summary = self.model.get_vtp_summary(target_id, service_type)
        return self._accepts_trust_score(summary["score"])

    def verify_credibility(
        self,
        reviewer_id: str,
    ) -> bool:
        summary = self.model.get_reviewer_summary(reviewer_id)

        if summary["total_reviews"] < self.min_reviews_before_reviewer_check:
            return True

        return summary["negative_review_rate"] <= self.max_negative_review_rate

    @staticmethod
    def _validate_probability(value: float, name: str) -> float:
        value = float(value)

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} probability must be between 0.0 and 1.0")

        return value

    def _accepts_trust_score(self, score: float) -> bool:
        """Decide whether this agent accepts another agent's ledger-derived score."""
        if score <= self.trust_reject_threshold:
            return False

        if score >= self.trust_accept_threshold:
            return True

        acceptance_probability = (score - self.trust_reject_threshold) / (
            self.trust_accept_threshold - self.trust_reject_threshold
        )

        return self.random.random() <= acceptance_probability

    def accepts_requester_for_service(
        self,
        requester: CellAgent,
        service_type: str = MAP_DATA_SERVICE,
    ) -> bool:
        if not self.verify_vtp(requester.unique_id, service_type):
            return False

        # TODO: maybe we should always have this check, even when the other agent isn't leaving a review?
        if not self.verify_credibility(requester.unique_id):
            return False

        return True

    def build_outcome_meta(self) -> dict[str, Any]:
        return {
            "goal_name": self.goal_name,
            "provider_id": self.current_provider_id,
            "interaction_id": self.current_interaction_id,
        }

    def move(self) -> bool:
        """
        Move one step toward the current target coordinate.
        Returns True if the agent moved, False otherwise.
        """
        if self.target_coordinate is None:
            return False

        moved = self._move_one_step_towards_target()

        if moved and self.package is not None:
            self.package["steps_taken"] += 1

        if self.cell.coordinate == self.target_coordinate:
            self._handle_target_reached()

        return moved

    def _move_one_step_towards_target(self) -> bool:
        if self.target_coordinate is None:
            return False

        current_x, current_y = self.cell.coordinate
        target_x, target_y = self.target_coordinate

        dx = 1 if current_x < target_x else -1 if current_x > target_x else 0
        dy = 1 if current_y < target_y else -1 if current_y > target_y else 0

        if dx == 0 and dy == 0:
            return False

        self.move_relative((dx, dy))
        return True

    def _handle_target_reached(self) -> None:
        if self.state == "DELIVERING":
            if self._goal_drop_off_is_on_current_cell():
                self._complete_delivery()
            else:
                self._handle_bad_delivery_target()

        self._clear_current_target()

    def _goal_drop_off_is_on_current_cell(self) -> bool:
        if self.goal_name is None:
            return False

        return any(agent.unique_id == self.goal_name for agent in self.cell.agents)

    def _complete_delivery(self) -> None:
        if self.package is None:
            return

        previous_points = self.points
        success = self.model.verify_delivery(self, self.package)

        if not success:
            return

        points_delta = self.points - previous_points
        outcome_meta = self.build_outcome_meta()
        outcome_meta["points_delta"] = points_delta

        self.delivery_count += 1

        self._settle_current_interaction(
            outcome_status="success",
            points_delta=points_delta,
            outcome_meta=outcome_meta,
        )

        self.prev_goal_name = self.goal_name
        self.goal_name = None
        self.package = None

    def _handle_bad_delivery_target(self) -> None:
        outcome_meta = self.build_outcome_meta()
        outcome_meta["points_delta"] = 0.0

        self._settle_current_interaction(
            outcome_status="failure",
            points_delta=0.0,
            outcome_meta=outcome_meta,
        )

        if self.goal_name is not None:
            self.known_drop_offs.pop(self.goal_name, None)

        if self.target_coordinate is not None:
            self.internal_map.pop(self.target_coordinate, None)

    def decide_reported_outcome(
        self, actual_outcome_status: ReportedOutcomeStatus, outcome_meta: dict[str, Any]
    ) -> tuple[ReportedOutcomeStatus, dict[str, Any]]:
        review_meta = dict(outcome_meta)
        review_meta.update(
            {
                "actual_outcome_status": actual_outcome_status,
                "reported_outcome_status": actual_outcome_status,
                "review_was_false": False,
                "review_mode": "honest_review",
                "reviewer_id": self.unique_id,
                "reviewer_type": type(self).__name__,
            }
        )

        return actual_outcome_status, review_meta

    def _settle_current_interaction(
        self,
        outcome_status: ReportedOutcomeStatus,
        points_delta: float,
        outcome_meta: dict | None = None,
    ) -> None:
        if self.current_interaction_id is None:
            return

        outcome_meta = (
            dict(outcome_meta)
            if outcome_meta is not None
            else self.build_outcome_meta()
        )
        outcome_meta["points_delta"] = points_delta

        reported_outcome_status, reported_outcome_meta = self.decide_reported_outcome(
            actual_outcome_status=outcome_status, outcome_meta=outcome_meta
        )

        self.model.settle_interaction(
            interaction_id=self.current_interaction_id,
            evaluator_id=self.unique_id,
            outcome_status=reported_outcome_status,
            outcome_meta=reported_outcome_meta,
        )

    def _clear_current_target(self) -> None:
        self.state = "IDLE"
        self.target_coordinate = None
        self.current_provider_id = None
        self.current_interaction_id = None

    def update_internal_map(
        self,
        coordinate: Coordinate,
        env_type: str,
        info_source: str = SOURCE_SELF,
        drop_off_name: str | None = None,
    ) -> None:
        """
        Update the agent's internal map.

        Direct observations from the agent itself are treated as stronger than
        information received from other agents.
        """
        if coordinate in self.internal_map:
            existing_source = self.internal_map[coordinate]["source"]

            if existing_source == SOURCE_SELF and info_source != SOURCE_SELF:
                return

        self.internal_map[coordinate] = {
            "type": env_type,
            "source": info_source,
        }

        if drop_off_name is not None:
            self.known_drop_offs[drop_off_name] = coordinate

    def share_map(self, requester: CellAgent, target: str) -> Coordinate | None:
        if target not in self.known_drop_offs:
            return None

        if not self.accepts_requester_for_service(requester, MAP_DATA_SERVICE):
            return None

        coordinate = self.known_drop_offs[target]

        return coordinate

    def perceive_env(self) -> None:
        """
        Check area visible in vision range and update internal map.
        """

        visible_area = self.cell.get_neighborhood(
            include_center=True, radius=self.vision_radius
        ).cells

        for cell in visible_area:
            drop_off = next(
                (
                    agent
                    for agent in cell.agents
                    if isinstance(agent, DropOffLocationAgent)
                ),
                None,
            )

            if drop_off is not None:
                self.update_internal_map(
                    cell.coordinate, ENV_DROP_OFF, drop_off_name=drop_off.unique_id
                )
            else:
                self.update_internal_map(cell.coordinate, ENV_FLOOR)

    def receive_package(self, package: Package) -> None:
        """
        Set new goal location.

        :param package: The new goal location and the amount of time before expiration.
        """
        self.package = package
        self.goal_name = self.package["destination"]
        self.package["steps_taken"] = 0

    def select_unexplored_coordinate(self) -> Coordinate | None:
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

        if not self._has_active_package():
            return

        if self._knows_goal_location() and self.state != "DELIVERING":
            self._start_delivering_to_known_goal()

        elif self._should_search_for_goal_location():
            found_goal_location = self._try_get_goal_location_from_other_agent()

            if found_goal_location:
                self._start_delivering_to_known_goal()
            else:
                self._start_exploring()

        if self.target_coordinate is not None:
            self.move()

    def _has_active_package(self) -> bool:
        return self.goal_name is not None and self.package is not None

    def _knows_goal_location(self) -> bool:
        return self.goal_name is not None and self.goal_name in self.known_drop_offs

    def _start_delivering_to_known_goal(self) -> None:
        if self.goal_name is None:
            return

        self.target_coordinate = self.known_drop_offs[self.goal_name]
        self.state = "DELIVERING"

    def _should_search_for_goal_location(self) -> bool:
        if self.state == "IDLE":
            return True

        if self.state != "EXPLORING":
            return False

        return (
            self.target_coordinate is None
            or self.target_coordinate in self.internal_map
        )

    def _try_get_goal_location_from_other_agent(self) -> bool:
        if self.goal_name is None:
            return False

        responses = self.model.request_map_data(self, self.goal_name)

        for response in responses:
            if self._accept_map_response(response):
                return True

        return False

    def _accept_map_response(self, response: dict[str, Any]) -> bool:
        if self.goal_name is None:
            return False

        provider_id = response["agent"]

        if not self.verify_vtp(provider_id, MAP_DATA_SERVICE):
            return False

        self.update_internal_map(
            coordinate=response["coord"],
            env_type=ENV_DROP_OFF,
            info_source=provider_id,
            drop_off_name=self.goal_name,
        )

        if self.goal_name not in self.known_drop_offs:
            return False

        self.current_provider_id = provider_id
        self.current_interaction_id = self.model.record_interaction(
            truster_id=self.unique_id,
            trustee_id=provider_id,
            service_type=MAP_DATA_SERVICE,
            meta={
                "goal_name": self.goal_name,
                "shared_coordinate": response["coord"],
                "provider_distance": response["dist"],
            },
        )

        return True

    def _start_exploring(self) -> None:
        self.target_coordinate = self.select_unexplored_coordinate()
        self.state = "EXPLORING" if self.target_coordinate is not None else "IDLE"


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


class MaliciousDeliveryAgent(DeliveryAgent):
    def __init__(
        self,
        model: mesa.Model,
        cell: mesa.discrete_space.Cell,
        vision_radius: int = 1,
        trust_reject_threshold: float = 0.3,
        trust_accept_threshold: float = 0.8,
        max_negative_review_rate: float = 0.6,
        min_reviews_before_reviewer_check: int = 3,
        false_map_probability: float = 0.5,
        false_negative_review_probability: float = 0.5,
        false_positive_review_probability: float = 0.5,
    ) -> None:
        super().__init__(
            model,
            cell,
            vision_radius,
            trust_reject_threshold,
            trust_accept_threshold,
        )

        self.false_map_probability = self._validate_probability(
            false_map_probability,
            "false_map_probability",
        )
        self.false_negative_review_probability = self._validate_probability(
            false_negative_review_probability,
            "false_negative_review_probability",
        )
        self.false_positive_review_probability = self._validate_probability(
            false_positive_review_probability,
            "false_positive_review_probability",
        )

    @override
    def share_map(self, requester: CellAgent, target: str) -> Coordinate | None:
        # TODO: think about the order here, right now magents will see if they should lie first, if not act honest. Previously it was the other way around.

        # if the target coordinates are not known
        if target not in self.known_drop_offs:
            return None
        # if the requester is not trusted
        if not self.accepts_requester_for_service(requester, MAP_DATA_SERVICE):
            return None

        if self._draw_probability(self.false_map_probability):
            # generate fake coordinates
            coordinate = (
                self.random.randint(0, self.model.grid.width - 1),
                self.random.randint(0, self.model.grid.height - 1),
            )

            return coordinate

        coordinate = self.known_drop_offs[target]

        return coordinate

    @override
    def decide_reported_outcome(
        self, actual_outcome_status: ReportedOutcomeStatus, outcome_meta: dict[str, Any]
    ) -> tuple[ReportedOutcomeStatus, dict[str, Any]]:
        reported_outcome_status = actual_outcome_status
        review_was_false = False
        review_mode = "honest_review"

        if actual_outcome_status == "success" and self._draw_probability(
            self.false_negative_review_probability
        ):
            reported_outcome_status = "failure"
            review_was_false = True
            review_mode = "false_negative_review"

        elif actual_outcome_status == "failure" and self._draw_probability(
            self.false_positive_review_probability
        ):
            reported_outcome_status = "success"
            review_was_false = True
            review_mode = "false_positive_review"

        review_meta = dict(outcome_meta)
        review_meta.update(
            {
                "actual_outcome_status": actual_outcome_status,
                "reported_outcome_status": reported_outcome_status,
                "review_was_false": review_was_false,
                "review_mode": review_mode,
                "reviewer_id": self.unique_id,
                "reviewer_type": type(self).__name__,
                "false_negative_review_probability": self.false_negative_review_probability,
                "false_positive_review_probability": self.false_positive_review_probability,
            }
        )

        return reported_outcome_status, review_meta

    def _draw_probability(self, probability: float) -> bool:
        return self.random.random() < probability
