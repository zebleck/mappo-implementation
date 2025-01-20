from model import ResourceWorld
import numpy as np
from policies import PPOPolicy
import time
import torch
from utils import SimLogger
import json
import os
from datetime import datetime

# For just training info (blue colored output)
logger = SimLogger(level=SimLogger.TRAINING, log_file="training")

# For training info plus debug info
# logger = SimLogger(level=SimLogger.DEBUG, log_file="training")

# For all details including verbose agent actions
# logger = SimLogger(level=SimLogger.VERBOSE, log_file="training")

# Create a mix of PPO and heuristic agents
num_agents = 20
num_ppo_agents = 10

agent_policies = [("ppo", None)] * num_ppo_agents + [("heuristic", None)] * (
    num_agents - num_ppo_agents
)

# Create and run the model
model = ResourceWorld(
    width=25,
    height=25,
    num_agents=num_agents,
    agent_policies=agent_policies,
    logger=logger,  # Pass logger to model
)

# After creating the model
logger.debug("\nInitial agent health:")
for agent in model.schedule.agents:
    logger.debug(f"Agent {agent.unique_id}: Health={agent.health}, Alive={agent.alive}")

# Training loop
n_episodes = 10000
max_steps_per_episode = 200
best_mean_survival_time = 0

logger.training("Starting training...")
start_time = time.time()

# Get reference to PPO agents for easier access
ppo_agents = [
    agent for agent in model.schedule.agents if isinstance(agent.policy, PPOPolicy)
]

# Create episodes directory if it doesn't exist
os.makedirs("episodes", exist_ok=True)

for episode in range(n_episodes):
    model.reset()
    episode_rewards = []
    survival_time = 0

    # Initialize episode history
    episode_history = {
        "episode": episode + 1,
        "mean_reward": 0,  # Will update at end
        "survival_time": 0,  # Will update at end
        "initial_state": {
            "resource_cells": [
                [
                    {
                        "food": cell.food,
                        "water": cell.water,
                        "is_food_patch": cell.is_food_patch,
                        "is_water_patch": cell.is_water_patch,
                    }
                    for cell in row
                ]
                for row in model.resource_cells
            ],
            "agents": [
                {
                    "id": agent.unique_id,
                    "policy": "ppo"
                    if isinstance(agent.policy, PPOPolicy)
                    else "heuristic",
                    "position": agent.pos,
                    "health": agent.health,
                    "food": agent.food_inventory,
                    "water": agent.water_inventory,
                    "energy": agent.energy,
                }
                for agent in model.schedule.agents
            ],
        },
        "steps": [],
    }

    # Run episode
    for step in range(max_steps_per_episode):
        # Record pre-step state
        step_data = {
            "step": step,
            "actions": {},
            "states": {
                "resource_cells": [
                    [{"food": cell.food, "water": cell.water} for cell in row]
                    for row in model.resource_cells
                ],
                "agents": [
                    {
                        "id": agent.unique_id,
                        "position": agent.pos,
                        "health": agent.health,
                        "food": agent.food_inventory,
                        "water": agent.water_inventory,
                        "energy": agent.energy,
                        "alive": agent.alive,
                    }
                    for agent in model.schedule.agents
                ],
            },
        }

        # Record actions (need to modify agent step to store action)
        for agent in model.schedule.agents:
            if agent.alive:
                action_type, action_params = agent.policy.choose_action(
                    agent, agent._get_observation()
                )
                step_data["actions"][str(agent.unique_id)] = {
                    "type": action_type,
                    "params": action_params,
                }

        model.step()
        survival_time += 1

        # Add step data to history
        episode_history["steps"].append(step_data)

        # Log total health of all agents
        total_health = sum(
            agent.health for agent in model.schedule.agents if agent.alive
        )
        logger.debug(f"Step {step}, Total Health: {total_health}")

        # Check if all PPO agents are dead
        ppo_agents_alive = [agent for agent in ppo_agents if agent.alive]

        if not ppo_agents_alive:
            break

        # Collect rewards
        ppo_rewards = [agent._get_reward() for agent in ppo_agents_alive]
        episode_rewards.append(np.mean(ppo_rewards))

    # Calculate episode statistics
    mean_reward = np.mean(episode_rewards) if episode_rewards else -100
    mean_survival_time = survival_time

    # Update best performance
    if mean_survival_time > best_mean_survival_time:
        best_mean_survival_time = mean_survival_time

    # Update PPO policies at the end of each episode
    for agent in ppo_agents:
        agent.policy.end_episode()

    # Update episode statistics
    episode_history["mean_reward"] = mean_reward
    episode_history["survival_time"] = survival_time

    # Save episode if it's a checkpoint
    if (episode + 1) % 10 == 0:
        elapsed_time = time.time() - start_time
        logger.training(f"Episode {episode + 1}/{n_episodes}")
        logger.training(f"Mean Reward: {mean_reward:.2f}")
        logger.training(f"Mean Survival Time: {mean_survival_time:.1f} steps")
        logger.training(f"Best Survival Time: {best_mean_survival_time:.1f} steps")
        logger.training(f"Elapsed Time: {elapsed_time:.1f}s")

        # Add learning statistics
        if hasattr(ppo_agents[0].policy, "network"):
            with torch.no_grad():
                sample_state = ppo_agents[0].policy._preprocess_observation(
                    ppo_agents[0]._get_observation()
                )
                probs, _ = ppo_agents[0].policy.network(sample_state)
                logger.training(f"Action Probabilities: {probs.cpu().numpy()}")
        logger.training("-" * 50)

        # Save episode data
        with open(f"episodes/episode_{episode+1}.json", "w") as f:
            json.dump(episode_history, f)

logger.close()
