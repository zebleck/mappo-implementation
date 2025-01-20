from model import ResourceWorld
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from agents import ResourceAgent
import json
import os
from datetime import datetime


def agent_portrayal(agent):
    """Define how agents are portrayed in the visualization."""
    if hasattr(agent, "food"):  # Resource cell
        portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0}

        # Color based on resources
        if agent.food > 0:
            portrayal["Color"] = f"rgb(0, {min(255, int(agent.food * 2.55))}, 0)"
        elif agent.water > 0:
            portrayal["Color"] = f"rgb(0, 0, {min(255, int(agent.water * 2.55))})"
        else:
            portrayal["Color"] = "white"

    else:  # Agent
        if not agent.alive:
            return None

        portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 1}

        # Color based on health
        health_color = min(255, int(agent.health * 2.55))
        portrayal["Color"] = f"rgb({255-health_color}, {health_color}, 0)"

        # Add text showing inventories
        portrayal["text"] = f"F:{agent.food_inventory} W:{agent.water_inventory}"
        portrayal["text_color"] = "black"

    return portrayal


def load_and_replay_episode(episode_num):
    """Load a saved episode and create a visualization server."""
    # Load episode data
    with open(f"episodes/episode_{episode_num}.json", "r") as f:
        episode_data = json.load(f)

    # Create model with same initial state
    model = ResourceWorld(
        width=25,
        height=25,
        num_agents=20,
        agent_policies=[
            (agent["policy"], None) for agent in episode_data["initial_state"]["agents"]
        ],
        replay_data=[step["actions"] for step in episode_data["steps"]],
    )

    # Set up initial state
    model.reset()
    initial_state = episode_data["initial_state"]

    # Set resource cells
    for x, row in enumerate(initial_state["resource_cells"]):
        for y, cell_data in enumerate(row):
            cell = model.resource_cells[x][y]
            cell.food = cell_data["food"]
            cell.water = cell_data["water"]
            cell.is_food_patch = cell_data["is_food_patch"]
            cell.is_water_patch = cell_data["is_water_patch"]

    # Set agent states
    for agent_data in initial_state["agents"]:
        agent = next(
            a for a in model.schedule.agents if a.unique_id == agent_data["id"]
        )
        agent.pos = tuple(agent_data["position"])
        agent.health = agent_data["health"]
        agent.food_inventory = agent_data["food"]
        agent.water_inventory = agent_data["water"]
        agent.energy = agent_data["energy"]

    # Create visualization elements
    grid = CanvasGrid(agent_portrayal, 25, 25, 500, 500)

    charts = [
        ChartModule(
            [
                {"Label": "Living_Agents", "Color": "green"},
                {"Label": "Total_Food", "Color": "red"},
                {"Label": "Total_Water", "Color": "blue"},
            ]
        )
    ]

    # Create and return server
    server = ModularServer(
        model.__class__,
        [grid] + charts,
        "Resource Gathering Model",
        {
            "width": 25,
            "height": 25,
            "num_agents": 20,
            "agent_policies": [
                (agent["policy"], None)
                for agent in episode_data["initial_state"]["agents"]
            ],
            "replay_data": [step["actions"] for step in episode_data["steps"]],
        },
    )

    return server


if __name__ == "__main__":
    # Check command line args for episode number
    import sys

    if len(sys.argv) > 1:
        episode_num = int(sys.argv[1])
        server = load_and_replay_episode(episode_num)
        server.launch()
    else:
        print("Please specify episode number to replay")
