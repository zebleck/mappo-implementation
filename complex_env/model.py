from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from agents import ResourceAgent
from resources import ResourceCell
import numpy as np
from utils import SimLogger


class ResourceWorld(Model):
    """A model for simulating resource gathering and agent interactions."""

    def __init__(
        self,
        width=25,
        height=25,
        num_agents=20,
        food_patches=5,
        water_patches=3,
        food_patch_size=3,
        water_patch_size=6,
        agent_policies=None,
        logger=None,
        replay_data=None,
    ):
        super().__init__()
        self.grid = MultiGrid(width, height, True)  # True enables wrapping
        self.schedule = RandomActivation(self)
        self.running = True
        self.current_step = 0

        self.food_patches = food_patches
        self.water_patches = water_patches
        self.food_patch_size = food_patch_size
        self.water_patch_size = water_patch_size

        # Initialize resource grid
        self.resource_cells = np.zeros((width, height), dtype=object)
        for x in range(width):
            for y in range(height):
                cell = ResourceCell(
                    self, (x, y)
                )  # Pass model and position to ResourceCell
                self.resource_cells[x][y] = cell
                self.grid.place_agent(cell, (x, y))  # Place cell in grid

        # Place food patches
        self._place_resource_patches(
            num_patches=self.food_patches,
            patch_size=self.food_patch_size,
            resource_type="food",
        )

        # Place water patches
        self._place_resource_patches(
            num_patches=self.water_patches,
            patch_size=self.water_patch_size,
            resource_type="water",
        )

        # Default to all heuristic agents if no policy distribution specified
        if agent_policies is None:
            agent_policies = [("heuristic", None)] * num_agents

        # Create agents with specified policies
        for i, (policy_type, policy_params) in enumerate(agent_policies):
            agent = ResourceAgent(
                i, self, policy_type=policy_type, policy_params=policy_params
            )
            # Find empty position or position with less than 3 agents
            while True:
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
                cell_contents = self.grid.get_cell_list_contents((x, y))
                if len(cell_contents) < 3:
                    self.grid.place_agent(agent, (x, y))
                    break
            self.schedule.add(agent)

        # Initialize data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Living_Agents": lambda m: sum(
                    1 for agent in m.schedule.agents if agent.alive
                ),
                "Total_Food": lambda m: sum(
                    cell.food for row in m.resource_cells for cell in row
                ),
                "Total_Water": lambda m: sum(
                    cell.water for row in m.resource_cells for cell in row
                ),
            },
            agent_reporters={
                "Health": "health",
                "Food": "food_inventory",
                "Water": "water_inventory",
                "Energy": "energy",
            },
        )

        self.logger = logger or SimLogger(SimLogger.NONE)
        self.replay_data = replay_data
        self.replay_step = 0

    def _place_resource_patches(self, num_patches, patch_size, resource_type):
        """Place resource patches on the grid."""
        for _ in range(num_patches):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)

            # Calculate patch bounds with wrapping
            for dx in range(-(patch_size // 2), patch_size // 2 + 1):
                for dy in range(-(patch_size // 2), patch_size // 2 + 1):
                    nx = (x + dx) % self.grid.width
                    ny = (y + dy) % self.grid.height

                    if resource_type == "food":
                        self.resource_cells[nx][ny].is_food_patch = True
                        self.resource_cells[nx][ny].food = 100
                    else:
                        self.resource_cells[nx][ny].is_water_patch = True
                        self.resource_cells[nx][ny].water = 100

    def step(self):
        """Advance the model by one step."""
        self.logger.debug(f"\nDEBUG: Starting model step {self.current_step}")
        self.logger.debug(
            f"DEBUG: Number of agents in scheduler: {len(self.schedule.agents)}"
        )

        self.datacollector.collect(self)
        self.logger.debug("DEBUG: Before scheduler step")
        for agent in self.schedule.agents:
            self.logger.debug(
                f"Agent {agent.unique_id}: Health={agent.health}, Alive={agent.alive}, Pos={agent.pos}"
            )

        if self.replay_data:
            # Use saved actions instead of policy
            for agent in self.schedule.agents:
                if agent.alive:
                    action = self.replay_data[self.replay_step].get(
                        str(agent.unique_id)
                    )
                    if action:
                        agent._execute_action(action["type"], action.get("params", {}))
            self.replay_step += 1
        else:
            self.schedule.step()

        self.logger.debug("\nDEBUG: After scheduler step")
        for agent in self.schedule.agents:
            self.logger.debug(
                f"Agent {agent.unique_id}: Health={agent.health}, Alive={agent.alive}, Pos={agent.pos}"
            )

        self.current_step += 1

        # Resource regeneration
        if self.current_step % 10 == 0:
            self.logger.debug("DEBUG: Regenerating food")
            self._regenerate_resources("food", 0.25)
        if self.current_step % 5 == 0:
            self.logger.debug("DEBUG: Regenerating water")
            self._regenerate_resources("water", 0.10)

        # Random storms
        if self.random.random() < 0.03:  # ~3% chance per step
            self._create_storm()

    def _regenerate_resources(self, resource_type, rate):
        """Regenerate resources at specified rate."""
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell = self.resource_cells[x][y]
                if resource_type == "food" and cell.is_food_patch:
                    cell.food = min(100, cell.food + (100 * rate))
                elif resource_type == "water" and cell.is_water_patch:
                    cell.water = min(100, cell.water + (100 * rate))

    def _create_storm(self):
        """Create a random storm that halves resources in affected area."""
        storm_x = self.random.randrange(self.grid.width)
        storm_y = self.random.randrange(self.grid.height)

        for dx in range(-2, 3):  # 5x5 storm area
            for dy in range(-2, 3):
                x = (storm_x + dx) % self.grid.width
                y = (storm_y + dy) % self.grid.height
                cell = self.resource_cells[x][y]
                cell.food //= 2
                cell.water //= 2

    def reset(self):
        """Reset the model for a new episode."""
        self.logger.debug("\nDEBUG: Starting reset")
        self.logger.debug(
            f"DEBUG: Number of agents before reset: {len(self.schedule.agents)}"
        )

        # Store agents before reset
        agents = list(self.schedule.agents)
        self.logger.debug(f"DEBUG: Stored {len(agents)} agents")

        # Clear the grid
        self.grid = MultiGrid(self.grid.width, self.grid.height, True)
        self.schedule = RandomActivation(self)
        self.current_step = 0

        self.logger.debug("\nDEBUG: Resetting resource cells")
        # Reset resource cells
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell = self.resource_cells[x][y]
                cell.food = 0
                cell.water = 0
                self.grid.place_agent(cell, (x, y))

        self.logger.debug("\nDEBUG: Placing resource patches")
        # Re-place resource patches
        self._place_resource_patches(
            num_patches=self.food_patches,
            patch_size=self.food_patch_size,
            resource_type="food",
        )
        self._place_resource_patches(
            num_patches=self.water_patches,
            patch_size=self.water_patch_size,
            resource_type="water",
        )

        self.logger.debug("\nDEBUG: Resetting agents")
        # Reset and re-add agents
        for i, agent in enumerate(agents):
            self.logger.debug(f"\nDEBUG: Processing agent {agent.unique_id}")
            # Reset agent state
            agent.health = 100
            agent.food_inventory = agent.random.randint(0, 5)
            agent.water_inventory = agent.random.randint(0, 5)
            agent.energy = 100
            agent.alive = True

            self.logger.debug(
                f"DEBUG: Agent {agent.unique_id} state reset - Health: {agent.health}, Alive: {agent.alive}"
            )

            # Place agent in new random position
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
                cell_contents = self.grid.get_cell_list_contents((x, y))
                if len(cell_contents) < 3:
                    self.grid.place_agent(agent, (x, y))
                    placed = True
                    self.logger.debug(
                        f"DEBUG: Agent {agent.unique_id} placed at position ({x}, {y})"
                    )
                attempts += 1

            if not placed:
                self.logger.warning(
                    f"WARNING: Could not place agent {agent.unique_id} after 100 attempts!"
                )

            # Re-add agent to scheduler
            self.schedule.add(agent)
            self.logger.debug(f"DEBUG: Agent {agent.unique_id} added to scheduler")

        self.logger.debug(
            f"\nDEBUG: Reset complete. Scheduler has {len(self.schedule.agents)} agents"
        )
        self.logger.debug("DEBUG: Checking final agent states:")
        for agent in self.schedule.agents:
            self.logger.debug(
                f"Agent {agent.unique_id}: Health={agent.health}, Alive={agent.alive}, Pos={agent.pos}"
            )

        # Store initial parameters if not already stored
        if not hasattr(self, "food_patches"):
            self.food_patches = 5
            self.water_patches = 3
            self.food_patch_size = 3
            self.water_patch_size = 6
