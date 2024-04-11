"""Module containing the VQAA solver."""
from itertools import islice

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pulser import Pulse, Sequence
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform, RampWaveform
from pulser_simulation import QutipEmulator
from scipy.optimize import minimize


def get_cost(bitstring, G, penalty=10) -> float:
    """Calculating the cost of a single configuration of the graph"""
    z = np.array(list(bitstring), dtype=int)
    A = np.array(nx.adjacency_matrix(G).todense())

    # Add penalty and bias:
    cost = penalty * (z.T @ np.triu(A) @ z) - np.sum(z)
    return cost


def get_avg_cost(counts, G, penalty_term) -> float:
    """Weighted average over all the configurations of the graph"""
    avg_cost = sum(counts[key] * get_cost(key, G, penalty_term) for key in counts)
    avg_cost = avg_cost / sum(counts.values())  # Divide by total samples

    return avg_cost


def func_complex(param, *args) -> float:
    """Cost function to minimize in VQAA"""
    G = args[0][0]
    register = args[0][1]
    penalty_term = args[0][2]

    C = complex_quantum_loop(param, register)
    cost = get_avg_cost(C, G, penalty_term)

    return cost


def func_simple(param, *args) -> float:
    G = args[0][0]
    register = args[0][1]
    C = simple_quantum_loop(param, register)
    cost = get_avg_cost(C, G, args[0][2])

    return cost


def simple_adiabatic_sequence(device, register, time, Omega=3.271543, detuning=5) -> Sequence:
    """Creates the adiabatic sequence

    Args:
        device: physical device simulation
        Omega: Frecuency
        register: arrangement of atoms in a quantum processor
        time: time of the adiabatic process
        detuning: detuning use

    Returns:
        sequence
    """
    delta_0 = -detuning
    delta_f = -delta_0

    adiabatic_pulse = Pulse(
        InterpolatedWaveform(time, [1e-9, Omega, 1e-9]),
        InterpolatedWaveform(time, [delta_0, 0, delta_f]),
        0,
    )

    sequence = Sequence(register, device)
    sequence.declare_channel("ising", "rydberg_global")

    sequence.add(adiabatic_pulse, "ising")

    return sequence


def complex_adiabatic_sequence(
    device,
    register,
    time1,
    time2,
    Omega=3.271543,
    detuning=5,
    detuning2=4,
) -> Sequence:
    """Creates the adiabatic sequence

    Args:
        device: physical device simulation
        Omega: Frecuency
        register: arrangement of atoms in a quantum processor
        time1: time rise for Rampwaveform
        time2: time fall for Rampwaveform
        detuning: initial detuning for Rampwaveform
        detuning2: final detuning for Rampwaveform

    Returns:
        sequence
    """
    delta_0 = -detuning
    delta_f = detuning2

    t_sweep = (delta_f - delta_0) / (2 * np.pi * 10) * 1000

    rise = Pulse.ConstantDetuning(
        RampWaveform(time1, 0.0, Omega),
        delta_0,
        0.0,
    )
    sweep = Pulse.ConstantAmplitude(
        Omega,
        RampWaveform(t_sweep, delta_0, delta_f),
        0.0,
    )
    fall = Pulse.ConstantDetuning(
        RampWaveform(time2, Omega, 0.0),
        delta_f,
        0.0,
    )

    sequence = Sequence(register, device)
    sequence.declare_channel("ising", "rydberg_global")
    sequence.add(rise, "ising")
    sequence.add(sweep, "ising")
    sequence.add(fall, "ising")

    return sequence


# Building the quantum loop


def complex_quantum_loop(parameters, register):
    params = np.array(parameters)

    (
        parameter_time1,
        parameter_time2,
        parameter_omega,
        parameter_detuning1,
        parameter_detuning2,
    ) = np.reshape(params.astype(int), 5)

    seq = complex_adiabatic_sequence(
        DigitalAnalogDevice,
        register,
        parameter_time1,
        parameter_time2,
        Omega=parameter_omega,
        detuning=parameter_detuning1,
        detuning2=parameter_detuning2,
    )

    simul = QutipEmulator.from_sequence(seq, sampling_rate=0.1)
    res = simul.run()
    counts = res.sample_final_state(N_samples=5000)  # Sample from the state vector

    return counts


def simple_quantum_loop(parameters, register):
    params = np.array(parameters)

    parameter_time, parameter_omega, parameter_detuning = np.reshape(params.astype(int), 3)
    seq = simple_adiabatic_sequence(
        DigitalAnalogDevice,
        register,
        parameter_time,
        Omega=parameter_omega,
        detuning=parameter_detuning,
    )

    simul = QutipEmulator.from_sequence(seq, sampling_rate=0.1)
    res = simul.run()
    counts = res.sample_final_state(N_samples=5000)  # Sample from the state vector
    # print(counts)

    return counts


def VQAA(
    atomic_register,
    graph,
    penalty,
    omega_range=(1, 5),
    detuning_range=(1, 5),
    time_range=(8, 25),
    minimizer_method="Nelder-Mead",
    repetitions=10,
    simple_sequence=True,
    complex_sequence=False,
) -> list:
    """Main function for the VQAA algorithm.

    Args:
        atomic_register (_type_): _description_
        graph (_type_): _description_
        penalty (_type_): _description_
        omega_range (tuple, optional): _description_. Defaults to (1, 5).
        detuning_range (tuple, optional): _description_. Defaults to (1, 5).
        time_range (tuple, optional): _description_. Defaults to (8, 25).
        minimizer_method (str, optional): _description_. Defaults to "Nelder-Mead".
        repetitions (int, optional): _description_. Defaults to 10.
        simple_sequence (bool, optional): _description_. Defaults to True.
        complex_sequence (bool, optional): _description_. Defaults to False.

    Returns:
        list: List of all final parameters.
    """
    scores = []
    params = []
    testing = []

    for repetition in range(repetitions):
        testing.append(repetition)
        random_omega = np.random.uniform(omega_range[0], omega_range[1])
        random_detuning2 = np.random.uniform(detuning_range[0], detuning_range[1])
        random_detuning1 = np.random.uniform(detuning_range[0], detuning_range[1])
        random_time1 = 1000 * np.random.uniform(
            time_range[0],
            time_range[1],
        )  # np.random.randint(time_range[0], time_range[1])
        random_time2 = 1000 * np.random.uniform(
            time_range[0],
            time_range[1],
        )  # np.random.randint(time_range[0], time_range[1])

        if complex_sequence is True:
            res = minimize(
                func_complex,
                args=[graph, atomic_register, penalty],
                x0=np.r_[
                    random_time1,
                    random_time2,
                    random_omega,
                    random_detuning1,
                    random_detuning2,
                ],
                method=minimizer_method,
                tol=1e-5,
                options={"maxiter": 200},  # 20
            )

        if simple_sequence is True:
            res = minimize(
                func_simple,
                args=[graph, atomic_register, penalty],
                x0=np.r_[random_time1, random_omega, random_detuning1],
                method=minimizer_method,
                tol=1e-5,
                options={"maxiter": 200},
            )

        # print(res.fun)
        scores.append(res.fun)
        params.append(res.x)

    optimal_parameters = params[np.argmin(scores)]

    return optimal_parameters


def solver_VQAA(
    atomic_register,
    graph,
    penalty_term,
    number_best_solutions=5,
    omega_range=(1, 5),
    detuning_range=(2, 5),
    time_range=(8, 28),
    minimizer_method="Nelder-Mead",
    repetitions=10,
    simple_sequence=True,
    complex_sequence=False,
):
    """Variational Quantum Adiabatic Algorithm solver

    Args:
        atomic_register: The atomic register representing the problem in the quantum device
        graph: The networkx graph used before the encoding to the register
        penalty_term: Penalty term for the cost fucntion to optimize
        number_best_solutions: The amount of solutions to output from the best ones
        omega_range: The range of frequencies to used for the optimizer parameters. Default (1,5)
        detuning_range: The range of detuning to used for the optimizer parameters. Default (1,5)
        time_range:Range of time evolution for QAA to used in optimizer parameters.Default (8,25)
        minimizer_method: Minimizer to use from scipy. Default Nelder-Mead
        repetitions: The number of times to repeat the optimization. Default(10)
        simple_sequence: A simple adiabatic sequence with InterpolatedWaveform. Default True
        complex_sequence: A complex adiabatic sequence with  RampWaveform. Default False

    Returns:
        counts_sorted: The dictionary of counts of the QAA with the optimal parameters
        opt_params:  Optimal parameters for the QAA
        solution: The list of solutions given the optimal parameters

    """

    opt_params = VQAA(
        atomic_register,
        graph,
        penalty_term,
        omega_range,
        detuning_range,
        time_range,
        minimizer_method,
        repetitions,
        simple_sequence,
        complex_sequence,
    )
    if simple_sequence is True:
        counts = simple_quantum_loop(opt_params, atomic_register)

    if complex_sequence is True:
        counts = complex_quantum_loop(opt_params, atomic_register)

    counts_sorted = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

    solution = []
    for item in islice(
        counts_sorted,
        number_best_solutions,
    ):  # use islice(d.items(), 3) to iterate over key/value pairs
        element = 0
        solutions_iterations = []
        for bit_solution in item:
            if int(bit_solution) == 1:
                solutions_iterations.append(atomic_register.qubit_ids[element])
            element += 1
        solution.append(solutions_iterations)

    return counts_sorted, opt_params, solution


def plot_distribution(C):
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    plt.figure(figsize=(12, 6))
    plt.xlabel("bitstings")
    plt.ylabel("counts")
    plt.bar(C.keys(), C.values(), width=0.5)
    plt.xticks(rotation="vertical")
    plt.show()
