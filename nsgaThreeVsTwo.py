from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament

from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.ref_dirs import get_reference_directions


gaTwo = NSGA2(
    pop_size = 100,
    sampling  = FloatRandomSampling(),
    selection = TournamentSelection(func_comp=binary_tournament),
    crossover = SBX(eta=15, prob=0.9),
    mutation = PM(eta=20),
    survival = RankAndCrowding(),
    output = MultiObjectiveOutput(),
)

gaThree = NSGA3(
)