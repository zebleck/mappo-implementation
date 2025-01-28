import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pygame
from rendering.episode_viewer import EpisodeViewer


def ultimatum_render(screen, state, window_size):
    """Rendering function for the Ultimatum Game."""
    # Colors
    BACKGROUND = (255, 255, 255)
    PROPOSER = (70, 130, 180)  # Steel Blue
    RESPONDER = (205, 92, 92)  # Indian Red
    TEXT_COLOR = (0, 0, 0)
    MONEY_COLOR = (50, 205, 50)  # Lime Green

    # Fill background
    screen.fill(BACKGROUND)

    # Initialize font
    pygame.font.init()
    font = pygame.font.SysFont("Arial", window_size // 20)
    large_font = pygame.font.SysFont("Arial", window_size // 15)

    # Draw agents
    agent_radius = window_size // 8

    # Proposer (left side)
    proposer_pos = (window_size // 4, window_size // 2)
    pygame.draw.circle(screen, PROPOSER, proposer_pos, agent_radius)

    # Responder (right side)
    responder_pos = (3 * window_size // 4, window_size // 2)
    pygame.draw.circle(screen, RESPONDER, responder_pos, agent_radius)

    # Draw labels
    proposer_label = font.render("Proposer", True, TEXT_COLOR)
    responder_label = font.render("Responder", True, TEXT_COLOR)

    screen.blit(
        proposer_label,
        (
            proposer_pos[0] - proposer_label.get_width() // 2,
            proposer_pos[1] + agent_radius + 10,
        ),
    )

    screen.blit(
        responder_label,
        (
            responder_pos[0] - responder_label.get_width() // 2,
            responder_pos[1] + agent_radius + 10,
        ),
    )

    # Draw offer information
    offer = state["offer"]
    if offer >= 0:  # Only show if an offer has been made
        # Draw offer amount
        offer_text = large_font.render(f"Offer: {offer}", True, MONEY_COLOR)
        screen.blit(
            offer_text,
            (window_size // 2 - offer_text.get_width() // 2, window_size // 4),
        )

        # Draw split visualization
        total_width = window_size // 2
        split_height = window_size // 15
        split_y = window_size * 3 // 4

        # Proposer's share (left)
        proposer_share = 100 - offer
        proposer_width = int((proposer_share / 100) * total_width)
        pygame.draw.rect(
            screen, PROPOSER, (window_size // 4, split_y, proposer_width, split_height)
        )

        # Responder's share (right)
        responder_width = int((offer / 100) * total_width)
        pygame.draw.rect(
            screen,
            RESPONDER,
            (window_size // 4 + proposer_width, split_y, responder_width, split_height),
        )

    # Draw decision if made
    if "decision" in state:
        decision = state["decision"]
        if decision is not None:
            decision_text = large_font.render(
                "ACCEPTED" if decision else "REJECTED",
                True,
                (0, 255, 0) if decision else (255, 0, 0),
            )
            screen.blit(
                decision_text,
                (window_size // 2 - decision_text.get_width() // 2, window_size // 6),
            )


def get_ultimatum_display_state(current_state, next_state, progress):
    """
    Prepare state for display with interpolation if needed.
    For Ultimatum Game, we'll just use discrete states without interpolation.
    """
    display_state = current_state.copy()

    # If we're transitioning to a new state and it contains a decision,
    # only show the decision after half the transition
    if progress > 0.5 and "decision" in next_state:
        display_state["decision"] = next_state["decision"]
        # Keep the offer from the current state even after decision
        if "offer" in current_state:
            display_state["offer"] = current_state["offer"]

    return display_state


def view_episode(episode_num, episodes_dir="episodes"):
    """Convenience function to view a specific episode."""
    viewer = EpisodeViewer(render_fn=ultimatum_render)
    viewer.play_episode(
        f"ultimatum_env/{episodes_dir}/episode_{episode_num}.json",
        get_display_state_fn=get_ultimatum_display_state,
    )
    viewer.close()


def view_all_episodes(episodes_dir="episodes"):
    """View all episodes in the directory in sequence."""
    import glob
    import os

    # Get all episode files sorted by number
    episode_pattern = f"ultimatum_env/{episodes_dir}/episode_*.json"
    episode_files = sorted(
        glob.glob(episode_pattern), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    if not episode_files:
        print(f"No episode files found matching pattern: {episode_pattern}")
        return

    viewer = EpisodeViewer(render_fn=ultimatum_render)

    for episode_file in episode_files:
        print(f"Playing episode: {os.path.basename(episode_file)}")

        viewer.play_episode(
            episode_file,
            get_display_state_fn=get_ultimatum_display_state,
        )

        # Small delay between episodes
        import time

        time.sleep(0.5)

    viewer.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # If episode number provided, play just that episode
        episode_num = int(sys.argv[1])
        view_episode(episode_num)
    else:
        # Otherwise play all episodes
        view_all_episodes()
