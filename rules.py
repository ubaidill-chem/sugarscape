from __future__ import annotations

import math
from typing import Literal, TypeVar, Sequence, Any, TYPE_CHECKING

import numpy as np
from mesa.discrete_space import Cell, OrthogonalVonNeumannGrid, OrthogonalMooreGrid

if TYPE_CHECKING:
    from core import SugarScape, Ant


class GeneGenerator:
    """
    Each agent has a genetic endowment consisting of a sugar metabolism and a level of vision. (p. 23-24)

    Metabolism is uniformly distributed with a minimum of 1 and a maximum of 4. (p. 24)

    Vision is initially distributed uniformly across agents with values ranging from 1 to 6, unless stated otherwise.
    All agents are given some initial endowment of sugar, which they carry with them as they move about the sugarscape
    (p. 24).
    """
    def __init__(self, mode: Literal["norm", "uniform"] = "uniform", *,
                 metabolism_rng: tuple[int, int] = (1, 4),
                 vision_rng: tuple[int, int] = (1, 6),
                 init_endowment_rng: tuple[int, int] = (5, 25)):
        self.mode = mode
        self.trait_ranges = {
            "metabolism": metabolism_rng,
            "vision": vision_rng,
            "init_endowment": init_endowment_rng
        }
        self.trait_defaults = {
            "max_age": np.inf
        }

    def set_trait(self, trait_name: str, val_range: tuple[int, int]):
        self.trait_ranges[trait_name] = val_range

    def get_random_vals(self, val_range: tuple[int, int], n_agents):
        low, high = val_range
        if self.mode == "norm":
            mean = (low + high) / 2
            std = (high - low) / 6
            return np.clip(np.round(np.random.normal(mean, std, n_agents)), 1, None)
        elif self.mode == "uniform":
            return np.round(np.random.uniform(low, high, n_agents))

    def generate_genes(self, n_agents: int = 1) -> dict[str, Sequence[Any]]:
        gene = {trait: self.get_random_vals(val_range, n_agents) for trait, val_range in self.trait_ranges.items()}
        gene.update({t: np.full(n_agents, vals) for t, vals in self.trait_defaults.items() if t not in gene})
        return gene


class Rule:
    def __init__(self, offset: int = 0):
        self.offset = offset
        self._model = None

    def is_active(self, step: int) -> bool:
        return step >= self.offset

    @property
    def model(self) -> SugarScape:
        if self._model is not None:
            return self._model
        raise ValueError("Model has not been set yet.")

    def bind_model(self, model):
        self._model = model


class EnvRule(Rule):
    def __init__(self):
        super().__init__()

    def apply_to_env(self):
        pass


class AgentRule(Rule):
    def __init__(self):
        super().__init__()

    def apply_to_agents(self):
        pass

    def handle_death(self, agent):
        pass

    def setup_genes(self, gene_generator: GeneGenerator):
        pass


ER = TypeVar('ER', bound=EnvRule)
AR = TypeVar('AR', bound=AgentRule)


class RuleSet:
    def __init__(self, model, env_rules: Sequence[ER], agent_rules: Sequence[AR]):
        self.env_rules = list(env_rules)
        if any(not isinstance(env_rule, EnvRule) for env_rule in env_rules):
            raise TypeError("The first arg of RuleSet must contain only EnvRules")

        self.agent_rules = list(agent_rules)
        if any(not isinstance(agent_rule, AgentRule) for agent_rule in agent_rules):
            raise TypeError("The second arg of RuleSet must contain only AgentRules")

        self.model = model
        for rule in self.env_rules + self.agent_rules:
            rule.bind_model(model)

    def apply_to_model(self, step: int):
        for env_rule in self.env_rules:
            if env_rule.is_active(step):
                env_rule.apply_to_env()
        for agent_rule in self.agent_rules:
            if agent_rule.is_active(step):
                agent_rule.apply_to_agents()

    def setup_gene_generator(self, gene_generator: GeneGenerator):
        for agent_rule in self.agent_rules:
            agent_rule.setup_genes(gene_generator)

    def handle_death(self, step: int, agent: Ant):
        for agent_rule in self.agent_rules:
            if agent_rule.is_active(step):
                agent_rule.handle_death(agent)


class GrowbackRule(EnvRule):
    """
    Sugarscape growback rule G_a: At each lattice position, sugar grows back at a rate of a units per time interval
    up to the capacity at that position. (p. 23)
    """
    def __init__(self, alpha: int | float):
        super().__init__()
        self.alpha = alpha

    def apply_to_env(self):
        caps = self.model.grid.sugar_cap.data
        self.model.grid.sugar_level.data += self.alpha
        self.model.grid.sugar_level.data = np.clip(self.model.grid.sugar_level.data, 0, caps)


class MovementRule(AgentRule):
    def __init__(self, mode: Literal['original', 'sugar_per_dist'] = 'original'):
        """
        Agent movement rule M :
        - Look out as far as vision permits in the four principal lattice directions and identify the unoccupied site(s)
        having the most sugar;
        - If the greatest sugar value appearson multiple sites then select the nearest one;
        - Move to this site;
        - Collect all the sugar at this new position
        (p. 25)
        """
        super().__init__()
        strats = {'original': self.original_find_dest, 'sugar_per_dist': self.sugar_per_dist_find_dest}
        self.dest_func = strats[mode]

        self.width = None
        self.height = None
        self.dist_1d = None
        self.dist_2d = None

    def bind_model(self, model):
        super().bind_model(model)
        metric_2d = {OrthogonalVonNeumannGrid: self._manhattan_dist, OrthogonalMooreGrid: math.hypot}

        grid = self.model.grid
        self.width, self.height = grid.dimensions
        self.dist_1d = self._toroidal_dist if grid.torus else self._linear_dist
        self.dist_2d = metric_2d.get(type(grid))

    @staticmethod
    def _linear_dist(u1, u2, _u_tot):
        return abs(u1 - u2)

    @staticmethod
    def _toroidal_dist(u1, u2, u_tot):
        diff = abs(u1 - u2)
        return min(diff, abs(u_tot - diff))

    @staticmethod
    def _manhattan_dist(dx, dy):
        return abs(dx) + abs(dy)

    def cell_dist(self, cell1: Cell, cell2: Cell) -> float:
        x1, y1 = cell1.coordinate
        x2, y2 = cell2.coordinate
        dx = self.dist_1d(x1, x2, self.width)
        dy = self.dist_1d(y1, y2, self.height)
        return self.dist_2d(dx, dy)

    def apply_to_agents(self):
        self.model.agents.shuffle_do("move", dest_func=self.dest_func)

    def original_find_dest(self, ori_cell: Cell, vision: float) -> Cell:
        max_sugar = 0
        sugary_dests = []
        for cell in ori_cell.get_neighborhood(radius=vision, include_center=True):
            if cell.is_full and cell.coordinate != ori_cell.coordinate:
                continue
            if cell.sugar_level > max_sugar:
                max_sugar = cell.sugar_level
                sugary_dests = [cell]
            elif cell.sugar_level == max_sugar:
                sugary_dests.append(cell)

        shortest_dist = np.inf
        near_dests = []
        for cell in sugary_dests:
            dist = self.cell_dist(ori_cell, cell)
            if dist < shortest_dist:
                shortest_dist = dist
                near_dests = [cell]
            elif dist == shortest_dist:
                near_dests.append(cell)

        return ori_cell.random.choice(near_dests)

    def sugar_per_dist_find_dest(self, ori_cell: Cell, vision: float) -> Cell:
        max_sugar_per_dist = 0
        destinations = []
        for cell in ori_cell.get_neighborhood(radius=vision, include_center=True):
            if cell.is_full and cell.coordinate != ori_cell.coordinate:
                continue
            dist = self.cell_dist(ori_cell, cell)
            sugar_per_dist = cell.sugar_level / dist
            if sugar_per_dist > max_sugar_per_dist:
                max_sugar_per_dist = cell.sugar_level
                destinations = [cell]
            elif sugar_per_dist == max_sugar_per_dist:
                destinations.append(cell)

        return ori_cell.random.choice(destinations)


class ReplacementRule(AgentRule):
    """
    Agent replacement rule R [a, b]: When an agent dies it is replaced by an agent of age 0 having random genetic
    attributes, random position on the sugarscape, random initial endowment, and a maximum age randomly selected from
    the range [a, b]. (p. 32-33)
    """
    def __init__(self, a, b):
        super().__init__()
        self.max_age_range = (a, b)

    def setup_genes(self, gene_generator: GeneGenerator):
        gene_generator.set_trait("max_age", self.max_age_range)

    def handle_death(self, agent):
        self.model.generate_ants(1)
