import numpy as np


class SimpleVisitTrackingEnv:
    """
    A minimal multi-agent environment where:
    - Agents move on a small grid (e.g., 5x5)
    - Each agent tries to collect food (fixed locations)
    - Agents observe local 3x3 window around them
    - Each agent maintains a visit map tracking last visited times
    - Rewards for food collection and exploring unvisited areas
    """

    def __init__(self, size=5, n_agents=2, n_food=3, view_size=3):
        self.size = size
        self.n_agents = n_agents
        self.n_food = n_food
        self.view_size = view_size
        self.steps = 0

    def reset(self):
        self.steps = 0

        # Initialize agent positions randomly
        self.agent_positions = np.random.randint(0, self.size, size=(self.n_agents, 2))

        # Fixed food positions
        self.food_positions = np.random.randint(0, self.size, size=(self.n_food, 2))

        # Initialize visit maps for each agent (tracks when each cell was last visited)
        self.visit_maps = np.full((self.n_agents, self.size, self.size), -1)

        # Mark initial positions as visited
        for i, pos in enumerate(self.agent_positions):
            self.visit_maps[i, pos[0], pos[1]] = 0

        # Get observations for all agents
        return [self._get_observation(i) for i in range(self.n_agents)]

    def step(self, actions):
        """
        Actions: 0-3 for up, right, down, left
        Returns: observations, rewards, done
        """
        self.steps += 1
        rewards = np.zeros(self.n_agents)

        # Move agents
        for i, action in enumerate(actions):
            old_pos = self.agent_positions[i].copy()

            if action == 0:  # Up
                self.agent_positions[i][1] = (
                    self.agent_positions[i][1] + 1
                ) % self.size
            elif action == 1:  # Right
                self.agent_positions[i][0] = (
                    self.agent_positions[i][0] + 1
                ) % self.size
            elif action == 2:  # Down
                self.agent_positions[i][1] = (
                    self.agent_positions[i][1] - 1
                ) % self.size
            elif action == 3:  # Left
                self.agent_positions[i][0] = (
                    self.agent_positions[i][0] - 1
                ) % self.size

            new_pos = self.agent_positions[i]

            # Update visit map for the agent
            """last_visit = self.visit_maps[i, new_pos[0], new_pos[1]]
            if last_visit == -1:  # Never visited before
                rewards[i] += 0.2  # Bonus for exploring new tile
            elif self.steps - last_visit > 20:  # Not visited in a while
                rewards[i] += 0.1  # Smaller bonus for revisiting after some time"""

            self.visit_maps[i, new_pos[0], new_pos[1]] = self.steps

        # Check for food collection and compute rewards
        for i in range(self.n_agents):
            # Small negative reward per step
            rewards[i] -= 0.1

            # Check if agent is on food
            agent_pos = self.agent_positions[i]
            for j, food_pos in enumerate(self.food_positions):
                if np.array_equal(agent_pos, food_pos):
                    rewards[i] += 1.0
                    # Respawn food in new random location
                    # self.food_positions[j] = np.random.randint(0, self.size, size=2)

        # Get new observations
        observations = [self._get_observation(i) for i in range(self.n_agents)]

        # Agents can't die
        done = False

        return observations, rewards, done

    def _get_observation(self, agent_idx):
        """
        Observation for each agent includes local view of surroundings (view_size x view_size x 2):
        - Channel 1: Food locations (binary)
        - Channel 2: Visit map (normalized time since last visit)
        """
        pos = self.agent_positions[agent_idx]

        # Initialize local view
        local_food = np.zeros((self.view_size, self.view_size))
        local_visits = np.zeros((self.view_size, self.view_size))

        # Calculate view range considering toroidal wrapping
        half_view = self.view_size // 2

        # Fill local view
        for i in range(-half_view, half_view + 1):
            for j in range(-half_view, half_view + 1):
                # Calculate wrapped world coordinates
                world_x = (pos[0] + i) % self.size
                world_y = (pos[1] + j) % self.size

                # Convert to local coordinates
                local_x = i + half_view
                local_y = j + half_view

                # Mark food locations
                for food_pos in self.food_positions:
                    if food_pos[0] == world_x and food_pos[1] == world_y:
                        local_food[local_x, local_y] = 1

                # Add visit information
                last_visit = self.visit_maps[agent_idx, world_x, world_y]
                if last_visit == -1:
                    local_visits[local_x, local_y] = 1  # Never visited
                else:
                    steps_since_visit = self.steps - last_visit
                    local_visits[local_x, local_y] = min(1.0, steps_since_visit / 20)

        # Flatten the observation
        obs = np.concatenate(
            [
                local_food.flatten(),  # Food view
                local_visits.flatten(),  # Visit map view
            ]
        )

        return obs

    def get_state(self):
        """Return the complete state of the environment."""
        return {
            "agent_positions": self.agent_positions,
            "food_positions": self.food_positions,
            "visit_maps": self.visit_maps,
            "size": self.size,
            "n_agents": self.n_agents,
            "n_food": self.n_food,
            "steps": self.steps,
        }

    def get_observation_size(self):
        """Return the size of the observation space."""
        # For each view_size x view_size window we have:
        # - One channel for food locations (binary)
        # - One channel for visit map (normalized time)
        return 2 * self.view_size * self.view_size
