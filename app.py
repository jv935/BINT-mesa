import numpy as np
import mesa
from mesa.visualization import SolaraViz, SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from agents import DeliveryAgent, DropOffLocationAgent
from model import BintWorldModel


model_params = {
    "num_agents": {
        "type": "SliderInt",
        "value": 5,
        "label": "Number of agents",
        "min": 2,
        "max": 50,
        "step": 1,
    },
    "num_drop_offs": {
        "type": "SliderInt",
        "value": 3,
        "label": "Number of drop offs",
        "min": 1,
        "max": 20,
        "step": 1,
    },
    "width": {
        "type": "SliderInt",
        "value": 10,
        "label": "Width",
        "min": 1,
        "max": 100,
        "step": 1,
    },
    "height": {
        "type": "SliderInt",
        "value": 10,
        "label": "Height",
        "min": 1,
        "max": 100,
        "step": 1,
    }
}


def agent_portrayal(agent: mesa.Agent):
    if isinstance(agent, DeliveryAgent):
        return AgentPortrayalStyle(size=np.min([(agent.points+10)*2, 100]))
    elif isinstance(agent, DropOffLocationAgent):
        return AgentPortrayalStyle(size=100, marker="s", color="black")
    else:
        return None


def property_layer_portrayal(layer: mesa.discrete_space.PropertyLayer):
    if layer.name == "drop_off_locations":
        return PropertyLayerStyle(color="black", alpha=0.8)

    return None


bint = BintWorldModel()

renderer = SpaceRenderer(model=bint, backend="matplotlib")
renderer.draw_structure(lw=2, ls="solid", color="black", alpha=0.5)
renderer.setup_agents(agent_portrayal).draw_agents()
#renderer.setup_propertylayer(property_layer_portrayal).draw_propertylayer()


import solara
from matplotlib.figure import Figure
from mesa.visualization.utils import update_counter

@solara.component
def ScoreBar(model: mesa.Model):
    update_counter.get()
    agent_ids = np.array(model.agents.select(agent_type=DeliveryAgent).get("unique_id"))
    agent_points = np.array(model.agents.select(agent_type=DeliveryAgent).get("points"))

    fig = Figure()
    ax = fig.subplots()
    ax.bar(agent_ids, agent_points, color="skyblue", edgecolor="black")
    ax.set_title("Scoreboard")
    ax.set_xlabel("Agent ID")
    ax.set_ylabel("Points")
    ax.set_ylim(bottom=0)

    return solara.FigureMatplotlib(fig)


page = SolaraViz(
    model=bint,
    renderer=renderer,
    components=[ScoreBar],
    model_params=model_params,
    name="test"
)

page