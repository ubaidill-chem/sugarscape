from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from core import SugarScape


def visualize_state(model: SugarScape) -> Figure:
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()

    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    plot_map(model, axes[0, 0])
    plot_dynamics(model_data.loc[:, ["population", "deaths"]], axes[0, 1], "Population Over Time", "# agents")
    plot_group_dynamics(agent_data, axes[0, 2], "Traits over time", "Value")
    plot_dynamics(model_data["gini"], axes[1, 0], ylabel="Gini Coefficient")
    plot_lorenz(model, axes[1, 1])
    return figure


def plot_map(model: SugarScape, ax: Axes):
    vmin, vmax = model.grid.sugar_cap.data.min(), model.grid.sugar_cap.data.max()
    sns.heatmap(model.grid.sugar_level.data, ax=ax, square=True, cbar=False, cmap='Greens', vmin=vmin, vmax=vmax)
    coords = np.array([cell.coordinate for cell in model.agents.get('cell')]) + 0.5
    df = pd.DataFrame(np.reshape(coords, shape=(-1, 2)), columns=["x", "y"])
    df["sugar"] = model.agents.get('sugar')

    sns.scatterplot(df, x='x', y='y', hue='sugar', palette='flare', s=18, ax=ax)
    ax.set(title="Sugar and Agents Map", xticks=[], yticks=[])
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)


def plot_dynamics(data: Union[pd.DataFrame, pd.Series], ax: Axes, title: str = None, ylabel: str = None):
    is_multi = isinstance(data, pd.DataFrame)
    if not is_multi:
        ylabel = ylabel or str(data.name).capitalize()
    title = title or f"{ylabel} Over Time"

    data_tidy = data.reset_index().melt(id_vars="index", var_name="Variable", value_name="Value")
    sns.lineplot(data=data_tidy, x="index", y="Value", hue="Variable" if is_multi else None, ax=ax)
    ax.set(xlabel="Step", ylabel=ylabel, title=title, ylim=(0, None))
    ax.grid(True)
    if is_multi:
        ax.legend()


def plot_group_dynamics(data: Union[pd.DataFrame, pd.Series], ax: Axes, title: str = None, ylabel: str = None):
    is_multi = isinstance(data, pd.DataFrame)
    if not is_multi:
        ylabel = ylabel or str(data.name).capitalize()
    title = title or f"{ylabel} Over Time"

    data_tidy = data.reset_index().melt(id_vars=["Step", "AgentID"], var_name="Variable", value_name="Value")
    sns.lineplot(data=data_tidy, x="Step", y="Value", hue="Variable" if is_multi else None, errorbar="sd", ax=ax)
    ax.set(xlabel="Step", ylabel=ylabel, title=title, ylim=(0, None))
    ax.grid(True)
    if is_multi:
        ax.legend()


def plot_lorenz(model: SugarScape, ax: Axes):
    wealths = np.sort(np.array(model.agents.get("sugar")))
    cumsum = np.insert(np.cumsum(wealths), 0, 0)
    cumsum *= 100 / cumsum[-1]
    pop = np.linspace(0, 100, len(wealths) + 1)

    sns.lineplot(x=pop, y=cumsum, ax=ax, label="lorenz Curve")
    sns.lineplot(x=[0, 100], y=[0, 100], ax=ax, label="equality line")
    ax.set(xlabel="% of population", ylabel="Cumulative % of total wealth", title='Lorenz Curve')
    ax.grid(True)
    ax.legend()
