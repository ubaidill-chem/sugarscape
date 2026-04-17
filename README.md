
# Sugarscape

A Python implementation of the classic Sugarscape model based on Epstein and Axtell (1996).

## Overview

Sugarscape is an agent-based model simulating a population of agents on a grid competing for sugar resources. Agents have genetic traits (metabolism, vision) and must move, consume sugar, and reproduce to survive.

## Installation

```bash
pip install requirements.txt
```

## Quick Start

```python
from sugarscape import SugarScape, GrowbackRule, MovementRule, ReplacementRule

model = SugarScape(
    ([GrowbackRule(1)], [MovementRule(), ReplacementRule(60, 80)]),
    n_agents=400,
    seed=42
)

for _ in range(1000):
    model.step()
```

## Features

- **Agent Rules**: Movement, replacement, and custom agent behaviors
- **Environment Rules**: Sugar growback and resource management
- **Data Collection**: Track population, wealth distribution (Gini coefficient), and agent traits
- **Visualization**: Population dynamics, agent distribution, Lorenz curves

## Model Components

- `Ant`: Individual agents with genetic traits and sugar holdings
- `SugarScape`: Main model class managing the grid and simulation
- `GrowbackRule`: Sugar regeneration at each location
- `MovementRule`: Agent pathfinding and movement
- `ReplacementRule`: Reproduction and agent replacement

## References

Epstein, J. M., & Axtell, R. L. (1996). Growing Artificial Societies: Social Science from the Bottom Up.
