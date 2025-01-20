# 1. Environment Setup

## Spatial Layout

- **Grid Size**: 25 × 25 cells (2D discrete grid)
- **Boundaries**: Wrapping edges (a toroidal grid). An agent exiting the right edge appears on the left edge, etc.

## Resource Placement & Dynamics

### Types of Resources:
- **Food**: Placed in small patches (e.g., 3 × 3 clusters) scattered randomly
- **Water**: Fewer but larger patches (e.g., 6 × 6 clusters)

### Regeneration:
- Food patches replenish at 25% of maximum capacity every 10 timesteps
- Water patches replenish at 10% of maximum capacity every 5 timesteps
- **Depletion Threshold**: Resources can be fully depleted if over-gathered. Once a cell's resource drops to zero, it starts regenerating from zero at the specified rates

## Environmental Events

- **Storms**: Every 30-40 timesteps, a random storm appears in a 5 × 5 area, halving the resource levels there for 5 timesteps

## Movement & Interaction Constraints

- **Movement Cost**: Agents lose 1 "energy unit" (abstract or derived from food/water) each time they move
- **Collision Rule**: Up to 3 agents can occupy the same cell. More than that is disallowed (incentive to disperse)

# 2. Agent Specification

- **Number of Agents**: 20 total
- **Initial Placement**: Randomly placed across the grid, ensuring no more than 2 agents in the same cell at initialization

## 2.1 Agent Internal State

Each agent maintains:

- **Health / Survival Timer**: Decreases by 1 every timestep, unless replenished via resource consumption. If it hits 0, the agent "dies" and is removed from the simulation
- **Food Inventory**: Can hold up to 20 units. Consumption of 1 unit per 5 timesteps to avoid health decay
- **Water Inventory**: Can hold up to 20 units. Consumption of 1 unit per 5 timesteps to avoid health decay
- **Energy**: A separate counter used for movement actions. Agents must spend 1 unit of energy to move; energy is restored by consuming food or water at specific ratios

### Goals & Preferences:
- **Primary Goal**: Survive as long as possible
- **Secondary Goal**: Accumulate additional resources (food and water) beyond bare survival needs
- **Trust/Memory**: Each agent keeps a trust score for every other agent it has encountered, updated after trades or communications

## 2.2 Perception

### Local Sensing
Each agent sees the state of its current cell and the 8 surrounding cells (a Moore neighborhood). This includes:
- Resource levels (food, water) in these cells
- Identities and visible inventories of other agents in these cells (if visible inventories are allowed)
- Any public messages broadcast by these neighbors

### Historical Logs
Agents remember their interactions (trades, communications, conflicts) with others for up to 50 timesteps.

## 2.3 Action Space

At each timestep, each agent chooses exactly one of the following actions:

1. **Move**: Move to one of the eight neighboring cells (costs 1 energy)
2. **Gather**: Collect up to min(remaining capacity, free space in inventory) resource units from the current cell
3. **Trade**:
   - Offer some amount of food/water in exchange for another resource
   - This action requires the presence of at least one other agent in the same cell
4. **Communicate**:
   - Send a text-based message to neighbors (those in the same or adjacent cells)
   - The message is generated or guided by the agent's LLM
   - Examples: "I have extra water, willing to trade for food," "There's a storm coming in the northeast," or "Beware agent #7, they're hoarding resources"
5. **Consume**: Actively choose to eat or drink resources to replenish health or energy (if not done automatically)
6. **Rest** (optional): Gain a small health/energy boost if an agent does nothing else, simulating rest or "idle" action

## 2.4 Decision-Making: LLM

### LLM for Communication & Reasoning
A local context prompt is constructed at each timestep for each agent, containing:
- Current local environment details (resource amounts, neighboring agents, relevant historical notes)
- The agent's internal state (inventories, health)
- Excerpts of recent messages or conversation threads

Then the LLM first thinks about the current state of the world and the agent's goals and then proposes:
- A strategic summary: e.g., "We are running low on food. Attempt to gather from patch at (X, Y)"
- A candidate message to share with neighbors
- Suggestions on alliance formation or trade

And then another call to the LLM to formulate the action using structured output.

# 3. Simulation Loop & Scheduling

## Initialization:

1. Place resources and agents
2. Set each agent's initial inventories (random 0–5 units of each resource)
3. Assign each agent a random "personality seed" that influences its LLM prompt context (e.g., "cautious," "gregarious," "opportunistic")

## Per Timestep:

1. **Agent Perception**: Each agent sees local resource info, neighbors, and incoming messages
2. **LLM Reasoning** (Internal): The agent queries its LLM with an up-to-date prompt. Receives suggestions or text to send
4. **Action Selection**: The agent chooses one action
5. **Action Execution**:
   - Move/gather/trade/communicate/consume/rest
   - If the action is Communicate, the LLM-generated message is broadcast to neighbors
6. **Environment Update**:
   - Deduct resource from cells if gathered
   - Adjust resource levels if a storm triggers
   - Replenish resources based on regeneration rates
   - Agents who lack sufficient resources to keep health above 0 are removed
7. **Logging**: Log agent actions, communications, resource changes

## Post-Timestep Analysis:

Compute any relevant metrics (e.g., average resource per agent, trust network updates, system-level resource distribution)

## Simulation Termination:

The scenario might run for 1000 timesteps or until all agents are dead (whichever comes first)

# 4. Data Collection & Outputs

## Agent-Level Logs

- **Action Trace**: (agent_id, timestep, action_type, relevant parameters)
- **Communication Logs**: Actual messages, who received them, sentiment or key terms extracted
- **Inventory & Health Changes**: Time-series of each agent's resource levels, health, and energy

## System-Level Metrics

- **Population Over Time**: Number of living agents at each timestep
- **Resource Map**: A snapshot of food/water distribution at periodic intervals (e.g., every 50 timesteps)
- **Interaction Network**: A dynamic graph where edges represent trades or direct communications. Keep track of edge weights (frequency, trust level)

## Derived Metrics for Hypothesis Testing

- **Communication Entropy**: For each batch of timesteps, measure the diversity (information content) of messages
- **Cooperation vs. Competition**: Use the Gini coefficient on final resource distribution to see how unequal resource ownership becomes
- **Alliance Stability**: Count how many timesteps alliances last on average (if agents share some notion of membership or mutual trust)
- **Critical Transitions**: Detect large changes in the ratio of cooperative vs. conflict behaviors. For instance, if the average trust drops below a threshold and conflict spikes, label it as a phase shift

# 5. Example Scenario Flow

## Timesteps 1–50:

- Agents roam, discover resources, gather food/water
- Simple "getting to know neighbors" communications might form small alliances or resource-sharing pacts

## Timestep 30 (Storm Event):

- A localized storm hits cells (10–14, 10–14). Resource levels in that area halve
- Agents in that region must decide whether to relocate or negotiate trade with neighbors elsewhere

## Timestep 50–100:

- Some alliances grow; some break down due to betrayal or resource scarcity
- LLM-driven communications might start showing strategic language: "I'll share water if you help me gather in X region"

## Timestep 200 (Seasonal Shift):

- Water regenerates faster, food regenerates slower
- Agents with low food reserves scramble to find new patches or start actively trading water for food

## Timestep 500:

- The system might see emergent "hubs" where certain agents have large resource reserves and many trading partners. A few might form "coalitions" controlling key patches
- Communication logs show negotiation and possibly deception

## Timestep 1000 or Terminal:

- Some agents have perished due to starvation, conflict, or isolation
- Final states are logged. Researchers analyze which communication patterns correlated with survival or wealth accumulation

# 6. Extensions and Variations

- **Institutional Rules**: Add partial enforcement or "laws" that penalize theft if discovered
- **Complex Resource Graph**: Introduce more resource types with different utility or synergy (e.g., wood + stone = tools, which increase gathering efficiency)
- **Larger Populations & Scaling**: Increase agent count to 100 or 500 to see mass emergent structures (trading hubs, large alliances, wars)
- **Heterogeneous Agent Architectures**: Some agents only rely on heuristics or RL, while others have LLM-based communication, to see how language-based negotiation competes with silent foragers

# Summary

In this specific scenario:

- A 50×50 grid with two resources (food, water), random resource patches, and periodic storms or seasonal changes creates a dynamic environment
- 20 agents, each with health, inventory, and energy constraints, must move, gather, trade, or communicate
- Agents use a hybrid LLM + RL policy: the LLM proposes communications, while RL handles low-level optimization
- Outputs include detailed agent logs, communication transcripts, interaction networks, and resource distributions, enabling analysis of how emergent cooperation or competition arises under changing conditions