from collections import Counter

from src.solver.ml.dataset import data_from_file
from src.solver.ml.model import ParametersModel, ParamsPrediction
from src.solver.opt_vqaa import QAA
from src.solver.tuner.scoring import ResultScore, score
from src.solver.utils.graph_register import GraphRegister


def MLQAA(
    register_file: str,
    store_results: bool = False,
) -> tuple[Counter, ResultScore, ParamsPrediction]:
    """Machine Learning Quantum Adiabatic Algorithm.

    Uses Graph Machine Learning to predict each parameter of
    a QAA algorithm to solve the MIS problem of a graph.

    Args:
        register_file (str): A file path defining a register with graph interactions.
        store_results (bool): Whether to store results in the register's file.

    Returns:
        (dict, ResultScore):
            counts (dict): Counter of the results.
            results (ResultScore): Detail of the results' score.
            params (dict): The final parameters used for QAA.
    """
    register = GraphRegister.from_json(register_file)
    graph = register.graph

    model = ParametersModel("data/models/v2")
    adjacency_matrix, _ = data_from_file(register_file)
    params = model.predict(adjacency_matrix=adjacency_matrix)
    counts = QAA(
        config=params.dict,
        register=register,
    )

    final_score = score(counts, graph)

    if store_results:
        metadata = {"score": final_score.dict, "params": params.dict}
        register.set_metadata(metadata)
        register.to_json_file(register_file)

    return counts, final_score, params
