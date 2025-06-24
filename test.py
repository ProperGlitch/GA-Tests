from pymoo.algorithms.moo.nsga2 import NSGA2
from pprint import pprint

def summarize_nsga2(algorithm):
    return {
        "Population Size": algorithm.pop_size,
        "Sampling": type(algorithm.sampling).__name__,
        "Crossover": {
            "Type": type(algorithm.crossover).__name__,
            "Probability": getattr(algorithm.crossover, "prob", None),
            "Eta": getattr(algorithm.crossover, "eta", None)
        },
        "Mutation": {
            "Type": type(algorithm.mutation).__name__,
            "Probability": getattr(algorithm.mutation, "prob", None),
            "Eta": getattr(algorithm.mutation, "eta", None)
        },
        "Eliminate Duplicates": algorithm.eliminate_duplicates
    }


pprint(summarize_nsga2(NSGA2(pop_size=100,sampling=0.1)))