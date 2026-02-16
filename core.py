from __future__ import annotations
from typing import Literal, Sequence, Any, Callable, Union

import mesa
import numpy as np
from mesa.discrete_space import CellAgent, Cell, OrthogonalVonNeumannGrid, PropertyLayer

from rules import ER, AR, RuleSet, GeneGenerator


class Ant(CellAgent):
    def __init__(self, model: SugarScape, cell: Cell, **gene: Any):
        super().__init__(model)
        self.model: SugarScape = model
        self.cell = cell
        self.gene = gene
        self.sugar = self.init_endowment
        self.age = 0

    def __getattr__(self, item):
        return self.gene.get(item)

    def die(self):
        self.remove()
        self.model.handle_death(self)

    def collect(self):
        """
        Sugar collected but not eaten - what an agent gathers beyond its metabolism - is added to the agent's sugar
        holdings. (p. 24)
        """
        self.sugar += self.cell.sugar_level
        self.cell.sugar_level = 0

    def eat(self):
        """
        The agent's metabolism is simply the amount of sugar it burns per time step, or iteration. (p. 24)

        If at any time the agent's sugar wealth falls to zero or below - that is, it has been unable to accumulate
        enough sugar to satisfy its metabolic demands - then we say that the agent has starved to death, and it is
        removed from the sugarscape. (p. 25)
        """
        self.sugar -= self.metabolism

    def move(self, dest_func: Callable[[Cell, float], Cell]):
        dest = dest_func(self.cell, self.vision)
        self.move_to(dest)
        self.collect()
        self.eat()

    def step(self):
        self.age += 1
        if self.sugar < 0 or self.age > self.max_age:
            self.die()


class SugarScape(mesa.Model):
    def __init__(self, rules: tuple[Sequence[ER], Sequence[AR]], *, 
                 mode: Literal["norm", "uniform"] = "uniform",
                 metabolism_rng: tuple[int, int] = (1, 4),
                 vision_rng: tuple[int, int] = (1, 6),
                 init_endowment_rng: tuple[int, int] = (5, 25),
                 model_reporters: dict[str, Union[str, Callable[[mesa.Model], Any], list]] = None,
                 agent_reporters: dict[str, Union[str, Callable[[mesa.Model], Any], list]] = None,
                 n_agents=400, seed=None):
        
        super().__init__(seed=seed)
        sugar_map = np.loadtxt("sugar-map.txt")
        self.grid = OrthogonalVonNeumannGrid(sugar_map.shape, torus=True, capacity=1, random=self.random)
        self.gene_gen = GeneGenerator(mode, metabolism_rng, vision_rng, init_endowment_rng)
        self.n_agents = n_agents
        self.deaths = 0

        self.grid.add_property_layer(PropertyLayer.from_data("sugar_cap", sugar_map))
        self.grid.add_property_layer(PropertyLayer.from_data("sugar_level", sugar_map))

        model_reporters, agent_reporters = self.setup_reporters(model_reporters, agent_reporters)
        self.datacollector = mesa.DataCollector(model_reporters, agent_reporters)

        self.rules = RuleSet(self, *rules)
        self.rules.setup_gene_generator(self.gene_gen)
        self.generate_ants(self.n_agents)

    def setup_reporters(self, model_reporters, agent_reporters):
        model_reporters = model_reporters or {}
        agent_reporters = agent_reporters or {}
        model_reporters.update({"population": lambda m: len(m.agents), "gini": self.gini, "deaths": "deaths"})
        agent_reporters.update({"metabolism": "metabolism", "vision": "vision"})
        return model_reporters,agent_reporters

    def generate_ants(self, n_agents):
        Ant.create_agents(self, n_agents, cell=self.random.sample(self.grid.empties.cells, k=n_agents),
                          **self.gene_gen.generate_genes(n_agents))

    def step(self):
        self.deaths = 0
        self.rules.apply_to_model(self.steps)
        self.agents.do("step")
        self.datacollector.collect(self)

    def handle_death(self, agent: Ant):
        self.rules.handle_death(self.steps, agent)
        self.deaths += 1

    @staticmethod
    def gini(model: SugarScape):
        wealths = np.sort(np.array(model.agents.get("sugar")))
        n = wealths.size
        weights = np.linspace(n - 0.5, 0.5, n)
        B = np.dot(wealths, weights) / n / wealths.sum()
        return 1 - 2 * B
