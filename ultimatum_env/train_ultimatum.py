import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rendering.episode_recorder import EpisodeRecorder
from simple_ultimatum_env import SimpleUltimatumEnv
from simple_ultimatum_agent import UltimatumAgent
import os
import torch

# Create save directory if it doesn't exist
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# Create environment and agents
env = SimpleUltimatumEnv()
agents = [UltimatumAgent() for _ in range(2)]

# Initialize recorder
recorder = EpisodeRecorder()

# Training loop
n_episodes = 100000
save_interval = 10000  # Save models every 10k episodes

for episode in range(n_episodes):
    observations = env.reset()
    episode_reward = [0, 0]
    done = False

    # Start recording if it's a checkpoint episode
    if (episode + 1) % 1000 == 0:  # Save every 1000th episode
        recorder.start_episode(
            episode + 1,
            {
                "offer": -1,  # No offer made yet
                "decision": None,  # No decision made yet
            },
            metadata={
                "total_episodes": n_episodes,
                "n_agents": env.n_agents,
            },
        )

    step = 0
    while not done:
        # Get actions from all agents
        actions = []
        for i, (agent, obs) in enumerate(zip(agents, observations)):
            action = agent.choose_action(obs)
            actions.append(action)

        # Environment step
        new_observations, rewards, done = env.step(actions)

        # Store rewards
        for i, reward in enumerate(rewards):
            agents[i].rewards.append(reward)
            episode_reward[i] += reward

        # Record step if it's a checkpoint episode
        if (episode + 1) % 1000 == 0:
            # Create state for visualization
            state = {
                "offer": actions[env.current_proposer] if step == 0 else -1,
                "decision": actions[1 - env.current_proposer] if step == 1 else None,
            }
            recorder.record_step(step, actions, new_observations, rewards, state)

        observations = new_observations
        step += 1

    # Update agents
    for agent in agents:
        agent.update()

    # Logging
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}")
        print(f"Proposer reward: {episode_reward[env.current_proposer]}")
        print(f"Responder reward: {episode_reward[1-env.current_proposer]}")
        print("------------------------")

    # Save episode if it's a checkpoint
    if (episode + 1) % 1000 == 0:
        filepath = recorder.save_episode(episode + 1)
        print(f"Saved episode to {filepath}")

    # Save models periodically
    if (episode + 1) % save_interval == 0:
        for i, agent in enumerate(agents):
            checkpoint = {
                "model_state_dict": agent.network.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "episode": episode,
            }
            torch.save(
                checkpoint,
                os.path.join(save_dir, f"agent_{i}_checkpoint_{episode+1}.pt"),
            )
        print(f"Saved models at episode {episode+1}")

# Save final models
for i, agent in enumerate(agents):
    checkpoint = {
        "model_state_dict": agent.network.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "episode": n_episodes,
    }
    torch.save(checkpoint, os.path.join(save_dir, f"agent_{i}_final.pt"))
print("Training completed. Final models saved.")
