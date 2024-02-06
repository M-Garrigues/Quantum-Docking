"""Module containing the ray tuner code."""

from collections.abc import Callable

import networkx
import pulser
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch

import src.config.pulser as global_conf


def optimise_QAA_parameters(
    tunable: Callable,
    graph: networkx.Graph,
    register: pulser.Register,
    num_samples=10,
    gpus_per_trial=1,
) -> dict:
    """Launch the optimization process.

    Args:
        tunable (Callable): The function to optimize.
        graph (networkx.Graph): The graph corresponding to the given register
        register (pulser.Register): The loaded register to optimise the parameters on.
        num_samples (int, optional): The number of configurations the algorithm tries.
            Defaults to 10.
        gpus_per_trial (int, optional): No of gpu used for one trial. Defaults to 1.
    """
    config = {
        "rise_time": tune.randint(16, (global_conf.MAX_COHERENCE_TIME - 16) / (3)),
        "fall_time": tune.randint(16, (global_conf.MAX_COHERENCE_TIME - 16) / (3 / 2)),
        "omega": tune.uniform(0, 15),
        "init_detuning": tune.uniform(1, 8),
        "final_detuning": tune.uniform(0, 8),
    }

    hyperopt_search = HyperOptSearch(
        metric="score",
        mode="max",
    )

    result = tune.run(
        tune.with_parameters(tunable, graph=graph, register=register),
        config=config,
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        num_samples=num_samples,
        search_alg=hyperopt_search,
        verbose=0,
    )

    best_trial = result.get_best_trial(metric="score", mode="max", scope="all")

    print(f"Best trial config: {best_trial.config}")
    print(
        f"Best trial final score: {best_trial.last_result['score']}",
    )

    return best_trial.config
