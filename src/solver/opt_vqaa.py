import networkx as nx
import numpy as np
import pulser
from pulser import Pulse, Sequence
from pulser.devices import Chadoq2
from pulser.waveforms import RampWaveform
from pulser_simulation import QutipEmulator

from src.solver.tuner.qaa_tuner import optimise_QAA_parameters
from src.solver.tuner.scoring import ResultScore, score
from src.solver.utils.graph_register import GraphRegister


def VQAA(
    register_file: str,
    optimisation_rounds: int,
    store_results: bool = False,
) -> tuple[dict, ResultScore, dict]:
    """Variationnal Quantum Adiabatic Algorithm, using Hyperopt as optimiser.

    Args:
        register_file (str): A file path defining a register with graph interactions.
        optimisation_rounds (int): The number of samples the optimiser should generate.
        store_results (bool): Whether to store results in the register's file.

    Returns:
        (dict, ResultScore):
            counts (dict): Counter of the results.
            results (ResultScore): Detail of the results' score.
            params (dict): The final parameters used for QAA.
    """
    register = GraphRegister.from_json(register_file)
    graph = register.graph

    params = optimise_QAA_parameters(
        QAA_scorer,
        register=register,
        graph=graph,
        gpus_per_trial=0,
        num_samples=optimisation_rounds,
    )

    counts = QAA(
        config=params,
        register=register,
    )

    final_score = score(counts, graph)

    if store_results:
        metadata = {"score": final_score.dict, "params": params}
        register.set_metadata(metadata)
        register.to_json_file(register_file)

    return counts, final_score, params


def adiabatic_sequence(
    device: pulser.devices.Device,
    register: pulser.Register,
    rise_time: int,
    fall_time: int,
    omega: float,
    init_detuning: float,
    final_detuning: float,
) -> pulser.Sequence:
    """Creates the adiabatic sequence

    Args:
        device: physical device simulation
        omega: frequency of the pulse
        register: arrangement of atoms in a quantum processor
        rise_time: time rise for Rampwaveform
        fall_time: time fall for Rampwaveform
        init_detuning: initial init_detuning for Rampwaveform
        final_detuning: final init_detuning for Rampwaveform

    Returns:
        sequence
    """
    delta_0 = -init_detuning
    delta_f = final_detuning

    t_sweep = (delta_f - delta_0) / (2 * np.pi * 10) * 1000

    t_sweep = t_sweep - (t_sweep % 4)
    rise_time = rise_time - (rise_time % 4)
    fall_time = fall_time - (fall_time % 4)

    rise = Pulse.ConstantDetuning(
        RampWaveform(rise_time, 0.0, omega),
        delta_0,
        0.0,
    )

    sweep = Pulse.ConstantAmplitude(
        omega,
        RampWaveform(t_sweep, delta_0, delta_f),
        0.0,
    )
    fall = Pulse.ConstantDetuning(
        RampWaveform(fall_time, omega, 0.0),
        delta_f,
        0.0,
    )

    sequence = Sequence(register, device)
    sequence.declare_channel("ising", "rydberg_global")
    sequence.add(rise, "ising")
    sequence.add(sweep, "ising")
    sequence.add(fall, "ising")

    return sequence


def QAA(config, register):
    (
        rise_time,
        fall_time,
        omega,
        init_detuning,
        final_detuning,
    ) = config.values()

    seq = adiabatic_sequence(
        Chadoq2,
        register=register,
        rise_time=rise_time,
        fall_time=fall_time,
        omega=omega,
        init_detuning=init_detuning,
        final_detuning=final_detuning,
    )

    simul = QutipEmulator.from_sequence(seq, sampling_rate=0.1)
    res = simul.run()
    counts = res.sample_final_state(N_samples=5000)  # Sample from the state vector

    return counts


def QAA_scorer(config, graph: nx.Graph, register: GraphRegister) -> dict:
    """Score function to maximise in VQAA"""
    counts = QAA(config, register)
    config_score = score(counts, graph)

    return {"score": config_score.total}
