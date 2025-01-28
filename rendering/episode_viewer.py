import json
import pygame


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

    def close(self):
        pygame.quit()
