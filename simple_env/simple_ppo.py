import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PPONetwork(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(input_size, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


class PPOAgent:
    def __init__(self, input_size=4, n_actions=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(input_size, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4)

        # PPO parameters
        self.gamma = 0.99
        self.epsilon = 0.2

        # Memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []

    def choose_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_probs, value = self.network(state)

        action = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs[action])

        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.log_probs.append(log_prob)

        return action

    def update(self):
        if len(self.states) == 0:
            return

        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.LongTensor(self.actions).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        old_log_probs = torch.stack(self.log_probs).detach()  # Detach old log probs

        # Compute returns
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update policy
        for _ in range(4):  # 4 epochs
            # Get new predictions
            action_probs, values = self.network(states)
            values = values.squeeze()

            # Get new log probs
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)

            # Compute advantages
            advantages = returns - values.detach()

            # Compute ratio and surrogate loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

            # Compute losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns - values).pow(2).mean()

            # Update network
            total_loss = actor_loss + critic_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), 0.5
            )  # Add gradient clipping
            self.optimizer.step()

        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
