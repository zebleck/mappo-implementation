from mesa import Agent
import numpy as np
from policies import HeuristicPolicy, PPOPolicy, LLMPolicy


class ResourceAgent(Agent):
    """An agent that gathers resources and interacts with others."""

    def __init__(self, unique_id, model, policy_type="heuristic", policy_params=None):
        super().__init__(unique_id, model)
        self.health = 100
        self.food_inventory = self.random.randint(0, 5)
        self.water_inventory = self.random.randint(0, 5)
        self.energy = 100
        self.alive = True
        self.max_inventory = 20

        # Initialize memory of other agents
        self.trust_scores = {}
        self.interaction_history = []

        # Set up policy
        self.policy = self._initialize_policy(policy_type, policy_params)

        # For learning policies
        self.last_observation = None
        self.last_action = None

    def _initialize_policy(self, policy_type, policy_params):
        if policy_type == "heuristic":
            return HeuristicPolicy()
        elif policy_type == "ppo":
            state_size = self._get_observation_size()
            action_size = 10  # 8 move directions + 1 gather + 1 rest
            return PPOPolicy(state_size, action_size)
        elif policy_type == "llm":
            return LLMPolicy(policy_params["llm_client"])
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

    def _get_observation(self):
        """Get the current observation of the environment."""
        x, y = self.pos

        # Get resource levels in Moore neighborhood
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True
        )

        resource_obs = []
        for cell_pos in neighborhood:
            cell = self.model.resource_cells[cell_pos[0]][cell_pos[1]]
            resource_obs.extend([cell.food, cell.water])

        # Agent's internal state
        agent_state = [
            self.health,
            self.food_inventory,
            self.water_inventory,
            self.energy,
        ]

        # Nearby agents
        nearby_agents = []
        for cell_pos in neighborhood:
            cell_agents = [
                agent
                for agent in self.model.grid.get_cell_list_contents(cell_pos)
                if isinstance(agent, ResourceAgent) and agent != self
            ]
            nearby_agents.extend(
                [
                    len(cell_agents),
                    sum(
                        self.trust_scores.get(agent.unique_id, 0)
                        for agent in cell_agents
                    ),
                ]
            )

        return {
            "resources": np.array(resource_obs),
            "agent_state": np.array(agent_state),
            "nearby_agents": np.array(nearby_agents),
        }

    def _get_observation_size(self):
        """Get the size of the observation space."""
        return (
            18
            + 4  # Resource observations (9 cells * 2 resources)
            + 18  # Agent state  # Nearby agents (9 cells * 2 features)
        )

    def _get_reward(self):
        """Calculate the reward based on survival."""
        if not self.alive:
            return -100.0  # Large penalty for death

        # Base reward for being alive
        reward = 1.0

        # Penalty for being close to death (low health)
        if self.health < 20:
            reward -= (20 - self.health) / 20  # Linear penalty up to -1.0

        return reward

    def step(self):
        """Execute one step of the agent."""
        if not self.alive:
            return

        # Get current observation
        observation = self._get_observation()

        # If we have a previous action, update the policy
        if self.last_observation is not None and self.last_action is not None:
            reward = self._get_reward()
            self.policy.update(
                self, self.last_observation, self.last_action, reward, observation
            )

        # Choose and execute action
        action_type, action_params = self.policy.choose_action(self, observation)
        self.current_action = (action_type, action_params)  # Store current action
        self._execute_action(action_type, action_params)

        # Store current observation and action for next update
        self.last_observation = observation
        self.last_action = (action_type, action_params)

        # Consume resources and update health
        self._consume_resources()

        # Die if health reaches 0
        if self.health <= 0:
            self.alive = False

    def _execute_action(self, action_type, action_params):
        """Execute the chosen action."""
        if action_type == "move":
            self._move(action_params.get("direction"))
        elif action_type == "gather":
            self._gather(action_params.get("direction"))
        elif action_type == "rest":
            self._rest()

    def _consume_resources(self):
        """Consume resources and update health."""
        self.model.logger.verbose(
            f"Agent {self.unique_id} consuming resources: Food={self.food_inventory}, Water={self.water_inventory}"
        )
        # Consume 1 unit every 5 steps
        if self.model.current_step % 5 == 0:
            if self.food_inventory > 0:
                self.food_inventory -= 1
                self.model.logger.verbose(
                    f"Agent {self.unique_id} consumed food, new inventory: {self.food_inventory}"
                )
            else:
                self.health -= 5
                self.model.logger.verbose(
                    f"Agent {self.unique_id} lost health due to no food: {self.health}"
                )

            if self.water_inventory > 0:
                self.water_inventory -= 1
                self.model.logger.verbose(
                    f"Agent {self.unique_id} consumed water, new inventory: {self.water_inventory}"
                )
            else:
                self.health -= 5
                self.model.logger.verbose(
                    f"Agent {self.unique_id} lost health due to no water: {self.health}"
                )

    def _move(self, direction=None):
        """Move in the specified direction."""
        if self.energy < 1:
            return

        if direction:
            x, y = self.pos
            new_x = (x + direction[0]) % self.model.grid.width
            new_y = (y + direction[1]) % self.model.grid.height
            new_pos = (new_x, new_y)

            # Check if the new position is valid
            if len(self.model.grid.get_cell_list_contents(new_pos)) < 3:
                self.model.grid.move_agent(self, new_pos)
                self.energy -= 1

    def _gather(self, direction=None):
        """Gather resources from current cell."""
        x, y = self.pos

        # Always gather from current cell, ignore direction
        cell = self.model.resource_cells[x][y]

        # Gather food if available
        if cell.food > 0 and self.food_inventory < self.max_inventory:
            gather_amount = min(cell.food, self.max_inventory - self.food_inventory)
            cell.food -= gather_amount
            self.food_inventory += gather_amount

        # Gather water if available
        if cell.water > 0 and self.water_inventory < self.max_inventory:
            gather_amount = min(cell.water, self.max_inventory - self.water_inventory)
            cell.water -= gather_amount
            self.water_inventory += gather_amount

    def _rest(self):
        """Rest to regain energy."""
        self.energy = min(100, self.energy + 10)
