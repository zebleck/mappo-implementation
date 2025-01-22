from simple_env.simple_vist_tracking_env import SimpleVisitTrackingEnv
from simple_ppo import PPOAgent
from episode_recorder import EpisodeRecorder

# Create environment and agents
env = SimpleVisitTrackingEnv(size=15, n_agents=2)
agents = [PPOAgent(input_size=env.get_observation_size()) for _ in range(2)]

# Initialize recorder
recorder = EpisodeRecorder()

# Training loop
n_episodes = 10000
max_steps = 100

for episode in range(n_episodes):
    observations = env.reset()
    episode_rewards = [0] * env.n_agents

    # Start recording if it's a checkpoint episode
    if (episode + 1) % 100 == 0:  # Save every 100th episode
        recorder.start_episode(
            episode + 1,
            env.get_state(),  # Get complete environment state
            metadata={
                "total_episodes": n_episodes,
                "max_steps": max_steps,
                "n_agents": env.n_agents,
            },
        )

    for step in range(max_steps):
        # Get actions from all agents
        actions = [agent.choose_action(obs) for agent, obs in zip(agents, observations)]

        # Environment step
        new_observations, rewards, done = env.step(actions)

        # Store rewards
        for i, reward in enumerate(rewards):
            agents[i].rewards.append(reward)
            episode_rewards[i] += reward

        observations = new_observations
        if done:
            break

        # Record step if it's a checkpoint episode
        if (episode + 1) % 100 == 0:
            recorder.record_step(
                step,
                actions,
                observations,
                rewards,
                env.get_state(),  # Get complete environment state
            )

    # Update all agents
    for agent in agents:
        agent.update()

    # Log progress
    if (episode + 1) % 10 == 0:
        print(
            f"Episode {episode+1}, Average Rewards: {[round(r, 2) for r in episode_rewards]}"
        )

    # Save episode if it's a checkpoint
    if (episode + 1) % 100 == 0:
        filepath = recorder.save_episode(episode + 1)
        print(f"Saved episode to {filepath}")
