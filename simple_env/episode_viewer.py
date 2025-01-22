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
        scale = state.get("food_scales", [1.0] * len(state["food_positions"]))[i]

        scaled_size = int(cell_size * 0.5 * scale)
        offset = (cell_size - scaled_size) // 2

        rect = pygame.Rect(
            x * cell_size + offset,
            y * cell_size + offset,
            scaled_size,
            scaled_size,
        )
        pygame.draw.rect(screen, FOOD, rect)

    # Draw main agents
    fades = state.get("agent_fades", [1.0] * len(state["agent_positions"]))
    for pos, fade in zip(state["agent_positions"], fades):
        x, y = pos
        color = (*AGENT[:3], int(255 * fade))  # Add alpha channel
        pygame.draw.circle(
            screen,
            color,
            (int((x + 0.5) * cell_size), int((y + 0.5) * cell_size)),
            cell_size // 3,
        )

    # Draw ghost agents (for wrapped movement)
    for pos, fade in state.get("agent_ghosts", []):
        x, y = pos
        color = (*AGENT[:3], int(255 * fade))  # Add alpha channel
        pygame.draw.circle(
            screen,
            color,
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

    # Calculate direct movement and wrapped positions
    interpolated_positions = []
    ghost_positions = []  # For agents crossing boundaries
    agent_fades = []  # Track fade for each agent

    for curr_pos, next_pos in zip(current_positions, next_positions):
        diff = next_pos - curr_pos

        # Check if wrapping would give shorter path
        wrapped_diff = np.where(diff > grid_size / 2, diff - grid_size, diff)
        wrapped_diff = np.where(
            wrapped_diff < -grid_size / 2, wrapped_diff + grid_size, wrapped_diff
        )

        # If we're using wrapped movement, we need to show the agent at both positions
        if not np.array_equal(diff, wrapped_diff):
            # Calculate the leaving position normally
            leaving_pos = curr_pos + wrapped_diff * progress

            # For the entering position, start from outside the grid
            if wrapped_diff[0] < 0:  # Moving left
                enter_start = np.array([grid_size, curr_pos[1]])
                enter_end = next_pos
            elif wrapped_diff[0] > 0:  # Moving right
                enter_start = np.array([-1, curr_pos[1]])
                enter_end = next_pos
            elif wrapped_diff[1] < 0:  # Moving up
                enter_start = np.array([curr_pos[0], grid_size])
                enter_end = next_pos
            else:  # Moving down
                enter_start = np.array([curr_pos[0], -1])
                enter_end = next_pos

            entering_pos = enter_start + (enter_end - enter_start) * progress

            # Calculate fade for smooth transition
            fade_out = max(0, 1 - progress * 2)  # Fade out first half
            fade_in = max(0, progress * 2 - 1)  # Fade in second half

            # Add positions with their fade values
            interpolated_positions.append(leaving_pos % grid_size)
            ghost_positions.append((entering_pos, fade_in))
            agent_fades.append(fade_out)
        else:
            # Normal movement
            pos = curr_pos + wrapped_diff * progress
            interpolated_positions.append(pos % grid_size)
            agent_fades.append(1.0)

    interpolated_state["agent_positions"] = interpolated_positions
    interpolated_state["agent_ghosts"] = ghost_positions
    interpolated_state["agent_fades"] = agent_fades

    # Handle food scales
    food_scales = []
    agent_positions = np.array(state2["agent_positions"])

    for food_pos in state1["food_positions"]:
        food_pos = np.array(food_pos)
        is_agent_on_food = any(
            np.array_equal(agent_pos, food_pos) for agent_pos in agent_positions
        )

        if is_agent_on_food:
            scale = 1.0 - (0.2 * progress)
        else:
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
