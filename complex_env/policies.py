from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.optim as optim
from networks import ActorCritic, Memory
import torch.nn.functional as F


class Policy(ABC):
    """Abstract base class for agent policies."""

    @abstractmethod
    def choose_action(self, agent, observation):
        """
        Choose an action based on the current observation.
        Returns: action_type, action_params
        """
        pass

    @abstractmethod
    def update(self, agent, observation, action, reward, next_observation):
        """Update the policy based on experience."""
        pass


class HeuristicPolicy(Policy):
    """The current rule-based policy we're using as baseline."""

    def choose_action(self, agent, observation):
        # Current logic from _choose_action
        if agent.energy < 20:
            return "rest", {}
        elif agent.food_inventory < 5 or agent.water_inventory < 5:
            cell = agent.model.resource_cells[agent.pos[0]][agent.pos[1]]
            if cell.food > 0 or cell.water > 0:
                return "gather", {}
            else:
                return "move", {}
        else:
            return "move", {}

    def update(self, agent, observation, action, reward, next_observation):
        # Heuristic policy doesn't learn
        pass


class PPOPolicy(Policy):
    """Proximal Policy Optimization policy."""

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon=0.2,
        c1=1,
        c2=0.01,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks and optimizer
        self.network = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Initialize memory
        self.memory = Memory()

        # Hyperparameters
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # clipping parameter
        self.c1 = c1  # value loss coefficient
        self.c2 = c2  # entropy coefficient

        # Training parameters
        self.batch_size = 32  # Smaller batch size
        self.n_epochs = 4
        self.update_timestep = 128  # More frequent updates

        # Add episode tracking
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_probs = []
        self.episode_values = []
        self.episode_masks = []

        self.timestep = 0

    def _preprocess_observation(self, observation):
        """Convert observation dict to tensor."""
        # Concatenate all observation components
        obs_array = np.concatenate(
            [
                observation["resources"],
                observation["agent_state"],
                observation["nearby_agents"],
            ]
        )
        return torch.FloatTensor(obs_array).to(self.device)

    def choose_action(self, agent, observation):
        """Choose an action using the current policy."""
        state = self._preprocess_observation(observation)

        with torch.no_grad():
            action_probs, value = self.network(state)
            action = torch.multinomial(action_probs, 1).item()
            action_prob = action_probs[action].item()

        # Store experience in episode buffer
        self.episode_states.append(state.cpu().numpy())
        self.episode_actions.append(action)
        self.episode_rewards.append(0)  # Will be updated later
        self.episode_probs.append(action_prob)
        self.episode_values.append(value.item())
        self.episode_masks.append(1)

        # Map numeric action to action type and direction
        # 0-7: move in 8 directions
        # 8: gather from current cell
        # 9: rest
        if action < 8:  # Move actions
            directions = [
                (1, 0),
                (1, 1),
                (0, 1),
                (-1, 1),
                (-1, 0),
                (-1, -1),
                (0, -1),
                (1, -1),
            ]
            return "move", {"direction": directions[action]}
        elif action == 8:  # Gather action
            return "gather", {}  # No direction needed for gathering
        else:  # Rest action
            return "rest", {}

    def update(self, agent, observation, action, reward, next_observation):
        """Update the last transition with the received reward."""
        if len(self.episode_rewards) > 0:
            self.episode_rewards[-1] = reward

    def end_episode(self):
        """Called at the end of each episode to update policy."""
        if len(self.episode_states) == 0:
            return

        # Convert episode data to tensors
        states = torch.FloatTensor(np.array(self.episode_states)).to(self.device)
        actions = torch.LongTensor(np.array(self.episode_actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.episode_rewards)).to(self.device)
        old_probs = torch.FloatTensor(np.array(self.episode_probs)).to(self.device)
        values = torch.FloatTensor(np.array(self.episode_values)).to(self.device)
        masks = torch.FloatTensor(np.array(self.episode_masks)).to(self.device)

        # Calculate returns and advantages
        returns = self._compute_returns(rewards, values, masks)
        advantages = returns - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for n epochs
        for _ in range(self.n_epochs):
            # Get current policy and value predictions
            action_probs, current_values = self.network(states)

            # Calculate ratio and surrogate loss
            dist = torch.distributions.Categorical(action_probs)
            current_probs = dist.probs.gather(1, actions.unsqueeze(1)).squeeze()
            ratio = current_probs / old_probs

            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

            # Calculate losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(current_values.squeeze(), returns)
            entropy = dist.entropy().mean()

            # Calculate total loss
            loss = policy_loss + self.c1 * value_loss - self.c2 * entropy

            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

        # Clear episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_probs = []
        self.episode_values = []
        self.episode_masks = []

    def _compute_returns(self, rewards, values, masks):
        """Compute returns using GAE (Generalized Advantage Estimation)."""
        returns = torch.zeros_like(rewards)
        running_return = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * masks[t]
            returns[t] = running_return

        return returns


class LLMPolicy(Policy):
    """Language model based policy."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def choose_action(self, agent, observation):
        # Format observation as prompt
        prompt = self._format_prompt(agent, observation)

        # Get LLM response
        response = self.llm.complete(prompt)

        # Parse response into action
        action_type, params = self._parse_response(response)
        return action_type, params

    def update(self, agent, observation, action, reward, next_observation):
        # LLM policy doesn't learn in traditional sense
        pass

    def _format_prompt(self, agent, observation):
        # TODO: Implement prompt engineering
        pass

    def _parse_response(self, response):
        # TODO: Implement response parsing
        pass
