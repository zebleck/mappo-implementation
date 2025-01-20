import numpy as np


class SimpleEnv:
    """
    A minimal multi-agent environment where:
    - Agents move on a small grid (e.g., 5x5)
    - Each agent tries to collect food (fixed locations)
    - No complex interactions, just movement and collection
    - Clear reward signal: +1 for collecting food, -0.1 per step
    """

    def __init__(self, size=5, n_agents=2, n_food=3):
        self.size = size
        self.n_agents = n_agents
        self.n_food = n_food

    def reset(self):
        # Initialize agent positions randomly
        self.agent_positions = np.random.randint(0, self.size, size=(self.n_agents, 2))

        # Fixed food positions
        self.food_positions = np.array([[1, 1], [3, 3], [1, 3]])

        # Get observations for all agents
        return [self._get_observation(i) for i in range(self.n_agents)]

    def step(self, actions):
        """
        Actions: 0-3 for up, right, down, left
        Returns: observations, rewards, done
        """
        rewards = np.zeros(self.n_agents)

        # Move agents
        for i, action in enumerate(actions):
            if action == 0:  # Up
                self.agent_positions[i][1] = min(
                    self.size - 1, self.agent_positions[i][1] + 1
                )
            elif action == 1:  # Right
                self.agent_positions[i][0] = min(
                    self.size - 1, self.agent_positions[i][0] + 1
                )
            elif action == 2:  # Down
                self.agent_positions[i][1] = max(0, self.agent_positions[i][1] - 1)
            elif action == 3:  # Left
                self.agent_positions[i][0] = max(0, self.agent_positions[i][0] - 1)

        # Check for food collection and compute rewards
        for i in range(self.n_agents):
            # Small negative reward per step
            rewards[i] = -0.1

            # Check if agent is on food
            agent_pos = self.agent_positions[i]
            for food_pos in self.food_positions:
                if np.array_equal(agent_pos, food_pos):
                    rewards[i] += 1.0

        # Get new observations
        observations = [self._get_observation(i) for i in range(self.n_agents)]

        # Episode ends after 100 steps
        done = False

        return observations, rewards, done

    def _get_observation(self, agent_idx):
        """
        Observation for each agent is:
        - Its own position (2)
        - Vector to nearest food (2)
        """
        obs = np.zeros(4)
        pos = self.agent_positions[agent_idx]

        # Own position
        obs[0:2] = pos

        # Vector to nearest food
        distances = np.linalg.norm(self.food_positions - pos, axis=1)
        nearest_food = self.food_positions[np.argmin(distances)]
        obs[2:4] = nearest_food - pos

        return obs

    def get_state(self):
        """Return the complete state of the environment."""
        return {
            "agent_positions": self.agent_positions,
            "food_positions": self.food_positions,
            "size": self.size,
            "n_agents": self.n_agents,
            "n_food": self.n_food,
            # Add any other state variables here
        }
