import json
import pygame
import numpy as np
from time import sleep


class EpisodeViewer:
    """Visualizes recorded episodes using Pygame."""

    def __init__(self, render_fn, window_size=800, fps=60):
        """
        Initialize the episode viewer.

        Args:
            render_fn: Function that renders the state. Should accept (screen, state, window_size) as arguments
            window_size: Size of the window in pixels
            fps: Frames per second for animation
        """
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Episode Viewer")
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.render_fn = render_fn

    def load_episode(self, filepath):
        """Load episode data from file."""
        with open(filepath, "r") as f:
            return json.load(f)

    def play_episode(self, filepath, get_display_state_fn, step_duration=0.5):
        """
        Play back an entire episode.

        Args:
            filepath: Path to episode file
            get_display_state_fn: Function that returns the state to display given (current_state, next_state, progress)
            step_duration: Duration of each step in seconds
        """
        episode_data = self.load_episode(filepath)
        running = True
        step_idx = 0
        transition_time = 0

        while running and step_idx < len(episode_data["steps"]):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            if not running:
                break

            current_state = episode_data["steps"][step_idx]["state"]
            next_state = (
                episode_data["steps"][step_idx + 1]["state"]
                if step_idx + 1 < len(episode_data["steps"])
                else current_state
            )

            transition_time += 1 / self.fps
            progress = min(transition_time / step_duration, 1.0)

            # Get the state to display using provided function
            display_state = get_display_state_fn(current_state, next_state, progress)

            # Render the state
            self.screen.fill((0, 0, 0))
            self.render_fn(self.screen, display_state, self.window_size)
            pygame.display.flip()

            if progress >= 1.0:
                step_idx += 1
                transition_time = 0

            self.clock.tick(self.fps)

        pygame.quit()


def simple_env_render(screen, state, window_size):
    """Default rendering function for SimpleEnv."""
    # Colors
    BACKGROUND = (255, 255, 255)
    GRID = (200, 200, 200)
    AGENT = (255, 0, 0)
    FOOD = (0, 255, 0)

    grid_size = state["size"]
    cell_size = window_size // grid_size

    # Fill background
    screen.fill(BACKGROUND)

    # Draw grid
    for i in range(grid_size + 1):
        pos = i * cell_size
        pygame.draw.line(screen, GRID, (pos, 0), (pos, window_size))
        pygame.draw.line(screen, GRID, (0, pos), (window_size, pos))

    # Draw food
    for i, pos in enumerate(state["food_positions"]):
        x, y = pos
        # Get pulsation scale if it exists
        scale = state.get("food_scales", [1.0] * len(state["food_positions"]))[i]

        # Calculate scaled size
        scaled_size = int(cell_size * 0.5 * scale)
        offset = (cell_size - scaled_size) // 2

        rect = pygame.Rect(
            x * cell_size + offset,
            y * cell_size + offset,
            scaled_size,
            scaled_size,
        )
        pygame.draw.rect(screen, FOOD, rect)

    # Draw agents
    for pos in state["agent_positions"]:
        x, y = pos
        pygame.draw.circle(
            screen,
            AGENT,
            (int((x + 0.5) * cell_size), int((y + 0.5) * cell_size)),
            cell_size // 3,
        )


def simple_env_interpolate(state1, state2, progress):
    """Interpolation function specific to SimpleEnv."""
    interpolated_state = state1.copy()
    grid_size = state1["size"]

    # Interpolate agent positions
    current_positions = np.array(state1["agent_positions"])
    next_positions = np.array(state2["agent_positions"])

    # Handle wrapped movement
    diff = next_positions - current_positions

    # Check for wrapping in both x and y directions
    diff = np.where(diff > grid_size / 2, diff - grid_size, diff)
    diff = np.where(diff < -grid_size / 2, diff + grid_size, diff)

    # Linear interpolation between positions
    interpolated_positions = current_positions + diff * progress

    # Wrap the interpolated positions to stay within grid bounds
    interpolated_positions = interpolated_positions % grid_size

    interpolated_state["agent_positions"] = interpolated_positions.tolist()

    # Add shrinking effect for food that agents are on
    food_scales = []
    agent_positions = np.array(state2["agent_positions"])

    for food_pos in state1["food_positions"]:
        # Check if any agent is exactly on this food position
        food_pos = np.array(food_pos)
        is_agent_on_food = any(
            np.array_equal(agent_pos, food_pos) for agent_pos in agent_positions
        )

        if is_agent_on_food:
            # Shrink to 20% of original size
            scale = 1.0 - (0.2 * progress)
        else:
            # Return to full size if no agent is on it
            scale = 1.0

        food_scales.append(scale)

    interpolated_state["food_scales"] = food_scales
    return interpolated_state


def ease_in_out_quad(t):
    """Quadratic ease-in-out function."""
    if t < 0.5:
        return 2 * t * t
    return 1 - ((-2 * t + 2) ** 2) / 2


def get_simple_env_display_state(current_state, next_state, progress):
    """Combines interpolation and easing for SimpleEnv states."""
    # Apply easing to progress
    eased_progress = ease_in_out_quad(progress)

    # Get interpolated state
    return simple_env_interpolate(current_state, next_state, eased_progress)


def view_episode(episode_num, episodes_dir="episodes"):
    """Convenience function to view a specific episode."""
    viewer = EpisodeViewer(render_fn=simple_env_render)
    viewer.play_episode(
        f"{episodes_dir}/episode_{episode_num}.json",
        get_display_state_fn=get_simple_env_display_state,
    )


if __name__ == "__main__":
    import sys

    episode_num = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    view_episode(episode_num)
