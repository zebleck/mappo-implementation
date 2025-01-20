import json
import pygame
import numpy as np
from time import sleep


class EpisodeViewer:
    """Visualizes recorded episodes using Pygame."""

    def __init__(self, window_size=800, grid_size=5):
        pygame.init()
        self.window_size = window_size
        self.grid_size = grid_size
        self.cell_size = window_size // grid_size
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Episode Viewer")

        # Colors
        self.BACKGROUND = (255, 255, 255)
        self.GRID = (200, 200, 200)
        self.AGENT = (255, 0, 0)
        self.FOOD = (0, 255, 0)

    def load_episode(self, filepath):
        """Load episode data from file."""
        with open(filepath, "r") as f:
            return json.load(f)

    def draw_grid(self):
        """Draw the background grid."""
        self.screen.fill(self.BACKGROUND)
        for i in range(self.grid_size + 1):
            pos = i * self.cell_size
            pygame.draw.line(self.screen, self.GRID, (pos, 0), (pos, self.window_size))
            pygame.draw.line(self.screen, self.GRID, (0, pos), (self.window_size, pos))

    def draw_state(self, agent_positions, food_positions):
        """Draw the current state."""
        # Draw agents
        for pos in agent_positions:
            x, y = pos
            pygame.draw.circle(
                self.screen,
                self.AGENT,
                (int((x + 0.5) * self.cell_size), int((y + 0.5) * self.cell_size)),
                self.cell_size // 3,
            )

        # Draw food
        for pos in food_positions:
            x, y = pos
            rect = pygame.Rect(
                x * self.cell_size + self.cell_size // 4,
                y * self.cell_size + self.cell_size // 4,
                self.cell_size // 2,
                self.cell_size // 2,
            )
            pygame.draw.rect(self.screen, self.FOOD, rect)

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

            # Draw current state
            self.draw_grid()

            if step_idx < len(episode_data["steps"]):
                step = episode_data["steps"][step_idx]
                # Get positions from state instead of directly from step
                self.draw_state(
                    step["state"]["agent_positions"], step["state"]["food_positions"]
                )
                step_idx += 1
            else:
                running = False

            pygame.display.flip()
            sleep(delay)

        pygame.quit()


def view_episode(episode_num, episodes_dir="episodes"):
    """Convenience function to view a specific episode."""
    viewer = EpisodeViewer()
    viewer.play_episode(f"{episodes_dir}/episode_{episode_num}.json")


if __name__ == "__main__":
    import sys

    episode_num = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    view_episode(episode_num)
