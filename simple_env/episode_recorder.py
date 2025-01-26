import json
from datetime import datetime
import os
import numpy as np


class EpisodeRecorder:
    """Records complete episode states and actions for later visualization."""

    def __init__(self, save_dir="episodes"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.current_episode = {"metadata": {}, "initial_state": None, "steps": []}

    def start_episode(self, episode_num, env_state, metadata=None):
        """Start recording a new episode."""
        self.current_episode = {
            "metadata": {
                "episode_num": episode_num,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {}),
            },
            "initial_state": self._convert_state_to_json(env_state),
            "steps": [],
        }

    def record_step(self, step_num, actions, observations, rewards, env_state):
        """Record a single step."""
        self.current_episode["steps"].append(
            {
                "step_num": step_num,
                "actions": [self._convert_to_json(act) for act in actions],
                "observations": [self._convert_to_json(obs) for obs in observations],
                "rewards": self._convert_to_json(rewards),
                "state": self._convert_state_to_json(env_state),
            }
        )

    def save_episode(self, episode_num):
        """Save the current episode to disk."""
        filename = f"episode_{episode_num}.json"
        filepath = os.path.join(self.save_dir, filename)

        with open(filepath, "w") as f:
            json.dump(self.current_episode, f, indent=2)

        return filepath

    def _convert_to_json(self, obj):
        """Convert numpy arrays and other types to JSON-serializable format."""
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_to_json(value) for key, value in obj.items()}
        return obj

    def _convert_state_to_json(self, state):
        """Convert entire state dict to JSON-serializable format."""
        return {key: self._convert_to_json(value) for key, value in state.items()}
