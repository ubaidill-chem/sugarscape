import tqdm

from core import SugarScape
from rules import GrowbackRule, MovementRule, GeneGenerator, ReplacementRule
from viz import visualize_state

if __name__ == '__main__':
    GG = GeneGenerator()
    G1 = GrowbackRule(1)
    M = MovementRule()
    R60_80 = ReplacementRule(60, 80)

    n_steps = 1000
    model = SugarScape(([G1], [M, R60_80]), n_agents=400, seed=42)
    for step in tqdm.tqdm(range(n_steps)):
        model.step()
    fig = visualize_state(model)
    fig.tight_layout()
    fig.show()
