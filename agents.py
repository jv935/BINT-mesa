import mesa
from mesa.discrete_space import CellAgent, FixedAgent
from typing_extensions import override
from typing import Any, Literal, TypedDict

Coordinate = tuple[int, int]
AgentState = Literal["IDLE", "EXPLORING", "DELIVERING"]
ReportedOutcomeStatus = Literal["success", "failure"]

STATE_IDLE = "IDLE"
STATE_EXPLORING = "EXPLORING"
STATE_DELIVERING = "DELIVERING"

ENV_FLOOR = "floor"
ENV_DROP_OFF = "drop_off"

SOURCE_SELF = "self"
MAP_DATA_SERVICE = "map_data"

DEFAULT_TRUST_REJECT_THRESHOLD = 0.30
DEFAULT_TRUST_ACCEPT_THRESHOLD = 0.80


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
        vision_radius: int,
        trust_reject_threshold: float = DEFAULT_TRUST_REJECT_THRESHOLD,
        trust_accept_threshold: float = DEFAULT_TRUST_ACCEPT_THRESHOLD,
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
        self.state: AgentState = STATE_IDLE
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
        self.points = 0.0
        self.delivery_count = 0
        self._all_possible_coords = model.all_coordinates
        self.cached_active_tnfts = 0
        self.cached_burned_tnfts = 0

        # Evaluation trace state for the most recent map share response.
        self.last_share_mode = "none"
        self.last_share_was_malicious = False
        self.last_share_coordinate: Coordinate | None = None
        self.last_share_target: str | None = None

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

    def build_outcome_meta(self) -> dict:
        true_goal_coordinate = self.model.get_drop_off_coordinate(self.goal_name)
        ended_on_coordinate = self.cell.coordinate
        target_coordinate = self.target_coordinate

        base_meta = {
            "goal_name": self.goal_name,
            "target_coordinate": target_coordinate,
            "target_x": target_coordinate[0] if target_coordinate is not None else None,
            "target_y": target_coordinate[1] if target_coordinate is not None else None,
            "true_goal_coordinate": true_goal_coordinate,
            "true_goal_x": (
                true_goal_coordinate[0] if true_goal_coordinate is not None else None
            ),
            "true_goal_y": (
                true_goal_coordinate[1] if true_goal_coordinate is not None else None
            ),
            "target_is_true_goal": target_coordinate == true_goal_coordinate,
            "ended_on_coordinate": ended_on_coordinate,
            "ended_x": ended_on_coordinate[0],
            "ended_y": ended_on_coordinate[1],
            "ended_at_true_goal": ended_on_coordinate == true_goal_coordinate,
            "provider_id": self.current_provider_id,
            "interaction_id": self.current_interaction_id,
        }

        if self.package is None:
            return {
                **base_meta,
                "steps_taken": None,
                "min_steps": None,
                "max_steps": None,
                "extra_steps": None,
                "lateness": None,
                "normalized_extra_steps": None,
                "delivery_efficiency": None,
            }

        steps_taken = self.package["steps_taken"]
        min_steps = self.package["min_steps"]
        max_steps = self.package["max_steps"]
        extra_steps = max(0, steps_taken - min_steps)
        lateness = max(0, steps_taken - max_steps)

        return {
            **base_meta,
            "steps_taken": steps_taken,
            "min_steps": min_steps,
            "max_steps": max_steps,
            "extra_steps": extra_steps,
            "lateness": lateness,
            "normalized_extra_steps": extra_steps / max(min_steps, 1),
            "delivery_efficiency": min_steps / max(steps_taken, 1),
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
        if self.state == STATE_DELIVERING:
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
        outcome_meta["delivery_success"] = True
        outcome_meta["failure_reason"] = None

        self.delivery_count += 1
        self.model.record_delivery_event(
            agent_id=self.unique_id,
            outcome_status="success",
            meta=outcome_meta,
        )
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
        outcome_meta["delivery_success"] = False
        outcome_meta["failure_reason"] = "bad_delivery_target"

        self.model.record_delivery_event(
            agent_id=self.unique_id,
            outcome_status="failure",
            meta=outcome_meta,
        )
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
        self.state = STATE_IDLE
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
        self.last_share_target = target
        self.last_share_coordinate = None
        self.last_share_was_malicious = False
        self.last_share_mode = "none"

        if target not in self.known_drop_offs:
            self.last_share_mode = "blocked_unknown_target"
            return None

        if not self.verify_vtp(requester.unique_id, MAP_DATA_SERVICE):
            self.last_share_mode = "blocked_untrusted_requester"
            return None

        coordinate = self.known_drop_offs[target]
        self.last_share_coordinate = coordinate
        self.last_share_mode = "honest_known_coordinate"
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

        if self._knows_goal_location() and self.state != STATE_DELIVERING:
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
        self.state = STATE_DELIVERING

    def _should_search_for_goal_location(self) -> bool:
        if self.state == STATE_IDLE:
            return True

        if self.state != STATE_EXPLORING:
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
                "requester_type": type(self).__name__,
                "provider_type": response.get("provider_type"),
                "shared_coordinate": response["coord"],
                "shared_coord_x": response.get("shared_coord_x"),
                "shared_coord_y": response.get("shared_coord_y"),
                "true_coordinate": response.get("true_coordinate"),
                "true_coord_x": response.get("true_coord_x"),
                "true_coord_y": response.get("true_coord_y"),
                "was_shared_coordinate_true": response.get(
                    "was_shared_coordinate_true"
                ),
                "provider_share_mode": response.get("provider_share_mode"),
                "provider_share_was_malicious": response.get(
                    "provider_share_was_malicious"
                ),
                "provider_distance": response.get(
                    "provider_distance", response.get("dist")
                ),
                "response_rank_by_distance": response.get("response_rank_by_distance"),
                "num_candidate_responses": response.get("num_candidate_responses"),
                "provider_trust_score_at_request": response.get(
                    "provider_trust_score_at_request"
                ),
                "provider_total_active_at_request": response.get(
                    "provider_total_active_at_request"
                ),
                "provider_context_matching_active_at_request": response.get(
                    "provider_context_matching_active_at_request"
                ),
                "provider_other_active_at_request": response.get(
                    "provider_other_active_at_request"
                ),
                "requester_trust_score_at_request": response.get(
                    "requester_trust_score_at_request"
                ),
                "requester_total_active_at_request": response.get(
                    "requester_total_active_at_request"
                ),
                "provider_total_burned_at_request": response.get(
                    "provider_total_burned_at_request"
                ),
                "provider_context_matching_burned_at_request": response.get(
                    "provider_context_matching_burned_at_request"
                ),
                "provider_other_burned_at_request": response.get(
                    "provider_other_burned_at_request"
                ),
                "provider_weighted_active_at_request": response.get(
                    "provider_weighted_active_at_request"
                ),
                "provider_weighted_burned_at_request": response.get(
                    "provider_weighted_burned_at_request"
                ),
                "requester_total_burned_at_request": response.get(
                    "requester_total_burned_at_request"
                ),
                "requester_weighted_active_at_request": response.get(
                    "requester_weighted_active_at_request"
                ),
                "requester_weighted_burned_at_request": response.get(
                    "requester_weighted_burned_at_request"
                ),
            },
        )

        return True

    def _start_exploring(self) -> None:
        self.target_coordinate = self.select_unexplored_coordinate()
        self.state = (
            STATE_EXPLORING if self.target_coordinate is not None else STATE_IDLE
        )


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
        vision_radius: int,
        trust_reject_threshold: float = DEFAULT_TRUST_REJECT_THRESHOLD,
        trust_accept_threshold: float = DEFAULT_TRUST_ACCEPT_THRESHOLD,
        maliciousness: float | None = 0.5,
        false_map_probability: float | None = None,
        false_negative_review_probability: float | None = None,
        false_positive_review_probability: float | None = None,
    ) -> None:
        super().__init__(
            model,
            cell,
            vision_radius,
            trust_reject_threshold,
            trust_accept_threshold,
        )

        self.maliciousness = (
            None
            if maliciousness is None
            else self._validate_probability(maliciousness, "maliciousness")
        )

        if self.maliciousness is not None:
            self.false_map_probability = self.maliciousness
            self.false_negative_review_probability = self.maliciousness
            self.false_positive_review_probability = self.maliciousness
        else:
            self.false_map_probability = self._validate_probability(
                false_map_probability, "false_map_probability"
            )
            self.false_negative_review_probability = self._validate_probability(
                false_negative_review_probability, "false_negative_review_probability"
            )
            self.false_positive_review_probability = self._validate_probability(
                false_positive_review_probability, "false_positive_review_probability"
            )

            self.false_map_probability = false_map_probability
            self.false_negative_review_probability = false_negative_review_probability
            self.false_positive_review_probability = false_positive_review_probability

    @override
    def share_map(self, requester: CellAgent, target: str) -> Coordinate | None:
        self.last_share_target = target
        self.last_share_coordinate = None
        self.last_share_was_malicious = False
        self.last_share_mode = "none"

        # if the target coordinates are not known
        # if target not in self.known_drop_offs:
        #     self.last_share_mode = "blocked_unknown_target"
        #     return None

        # if the requester is not trusted
        if not self.verify_vtp(requester.unique_id, MAP_DATA_SERVICE):
            self.last_share_mode = "blocked_untrusted_requester"
            return None

        if self._draw_probability(self.false_map_probability):
            coordinate = (
                self.random.randint(0, self.model.grid.width - 1),
                self.random.randint(0, self.model.grid.height - 1),
            )

            # more logging stuff
            self.last_share_coordinate = coordinate
            self.last_share_was_malicious = True
            self.last_share_mode = "malicious_random_coordinate"

            return coordinate

        coordinate = self.known_drop_offs[target]
        self.last_share_coordinate = coordinate
        self.last_share_mode = "honest_known_coordinate"

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

    @staticmethod
    def _validate_probability(value: float, name: str) -> float:
        value = float(value)

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} probability must be between 0.0 and 1.0")

        return value

    def _draw_probability(self, probability: float) -> bool:
        return self.random().random() <= probability
