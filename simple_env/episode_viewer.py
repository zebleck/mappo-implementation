import json
import pygame
import numpy as np
from time import sleep


class EpisodeViewer:
    """Visualizes recorded episodes using Pygame."""

    def __init__(self, render_fn, window_size=800):
        """
        Initialize the episode viewer.

        Args:
            render_fn: Function that renders the state. Should accept (screen, state, window_size) as arguments
            window_size: Size of the window in pixels
            grid_size: Number of cells in the grid
        """
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Episode Viewer")

        # Render function
        self.render_fn = render_fn

    def load_episode(self, filepath):
        """Load episode data from file."""
        with open(filepath, "r") as f:
            return json.load(f)

    def play_episode(self, filepath, delay=0.5):
        """Play back an entire episode."""
        episode_data = self.load_episode(filepath)

        running = True
        step_idx = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            if not running:
                break

            if step_idx < len(episode_data["steps"]):
                step = episode_data["steps"][step_idx]
                # Clear screen and render state
                self.screen.fill((0, 0, 0))  # Clear with black background
                self.render_fn(self.screen, step["state"], self.window_size)
                step_idx += 1
            else:
                running = False

            pygame.display.flip()
            sleep(delay)

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

    # Draw agents
    for pos in state["agent_positions"]:
        x, y = pos
        pygame.draw.circle(
            screen,
            AGENT,
            (int((x + 0.5) * cell_size), int((y + 0.5) * cell_size)),
            cell_size // 3,
        )

    # Draw food
    for pos in state["food_positions"]:
        x, y = pos
        rect = pygame.Rect(
            x * cell_size + cell_size // 4,
            y * cell_size + cell_size // 4,
            cell_size // 2,
            cell_size // 2,
        )
        pygame.draw.rect(screen, FOOD, rect)


def view_episode(episode_num, episodes_dir="episodes"):
    """Convenience function to view a specific episode."""
    viewer = EpisodeViewer(simple_env_render)
    viewer.play_episode(f"{episodes_dir}/episode_{episode_num}.json")


if __name__ == "__main__":
    import sys

    episode_num = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    view_episode(episode_num)
