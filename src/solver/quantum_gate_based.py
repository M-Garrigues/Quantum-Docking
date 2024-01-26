from openqaoa import QAOA
from openqaoa.backends import create_device
from openqaoa.problems import MIS
from openqaoa.utilities import ground_state_hamiltonian


def solver_QAOA_gate_based(
    graph,
    p_layer=2,
    optimizer="nelder-mead",
    print_circuit=False,
    print_cost_history=False,
    print_hamiltonian=False,
):
    problem = MIS(graph)
    qubo_problem = problem.qubo

    if print_hamiltonian:
        print(qubo_problem.hamiltonian.expression)

    qiskit_device = create_device(location="local", name="qiskit.shot_simulator")
    q_problem = QAOA()
    q_problem.set_device(qiskit_device)
    q_problem.set_circuit_properties(
        p=p_layer,
        param_type="standard",
        init_type="rand",
        mixer_hamiltonian="x",
    )
    q_problem.set_backend_properties(n_shots=1024, seed_simulator=1)
    q_problem.set_classical_optimizer(
        method=optimizer,
        maxiter=200,
        tol=0.001,
        optimization_progress=True,
        cost_progress=True,
        parameter_log=True,
    )
    q_problem.compile(qubo_problem)

    if print_circuit:
        q_problem.backend.parametric_circuit.draw()

    q_problem.optimize()
    correct_solution = ground_state_hamiltonian(q_problem.cost_hamil)

    opt_results = q_problem.result

    if print_cost_history:
        opt_results.plot_cost(figsize=(7, 4), label="qaoa")

    print("Best solution:", correct_solution[1])
    opt_results.plot_probabilities(label="Probability distribution - QAOA over MIS")
