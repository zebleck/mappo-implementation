import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class UltimatumNetwork(nn.Module):
    def __init__(self, input_size, proposer_actions, responder_actions):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU())

        # Separate heads for proposer and responder
        self.proposer_head = nn.Sequential(
            nn.Linear(64, proposer_actions), nn.Softmax(dim=-1)
        )

        self.responder_head = nn.Sequential(
            nn.Linear(64, responder_actions), nn.Softmax(dim=-1)
        )

        self.value_head = nn.Sequential(nn.Linear(64, 1))

    def forward(self, x, is_proposer):
        shared_features = self.shared(x)
        if is_proposer:
            return self.proposer_head(shared_features), self.value_head(shared_features)
        return self.responder_head(shared_features), self.value_head(shared_features)


class UltimatumAgent:
    def __init__(self, input_size=2, proposer_actions=101, responder_actions=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = UltimatumNetwork(
            input_size, proposer_actions, responder_actions
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)

        # Memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

        # Parameters
        self.gamma = 0.99
        self.epsilon = 0.2

    def choose_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        is_proposer = bool(state[0].item())  # First element indicates role

        action_probs, value = self.network(state, is_proposer)

        # Add exploration noise more safely
        action_probs = action_probs.detach().cpu().numpy()
        noise = np.random.normal(0, 0.1, size=action_probs.shape)
        action_probs = action_probs + noise
        # Ensure valid probabilities
        action_probs = np.maximum(action_probs, 1e-6)  # Avoid zeros
        action_probs = action_probs / action_probs.sum()  # Renormalize

        # Safety check for NaN values
        if np.any(np.isnan(action_probs)):
            action_probs = np.ones_like(action_probs) / len(action_probs)

        action = np.random.choice(len(action_probs), p=action_probs)
        log_prob = torch.log(torch.FloatTensor([action_probs[action]]).to(self.device))

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)

        return action

    def update(self):
        if len(self.states) == 0:
            return

        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.LongTensor(self.actions).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        old_log_probs = torch.stack(self.log_probs).detach()

        # Compute returns
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update policy multiple times
        for _ in range(4):
            # Get new predictions
            is_proposer = states[:, 0] > 0.5
            action_probs = []
            values = []

            for state, proposer in zip(states, is_proposer):
                prob, val = self.network(state, proposer)
                action_probs.append(prob)
                values.append(val)

            action_probs = torch.stack([p[a] for p, a in zip(action_probs, actions)])
            values = torch.cat(values)

            # Compute ratio and losses
            ratio = torch.exp(torch.log(action_probs) - old_log_probs)
            advantages = returns - values.detach()

            policy_loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages,
            ).mean()

            value_loss = 0.5 * (returns - values).pow(2).mean()

            # Update network
            total_loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
