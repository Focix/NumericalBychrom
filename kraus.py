import numpy as np

def full_kraus_evolution(H, c_ops_list, rho0, t_final, dt):
    """
    Perform full Kraus evolution of a density matrix under a Hamiltonian and collapse operators.

    Parameters:
    H : ndarray
        The Hamiltonian of the system (2D array).
    c_ops_list : list of ndarray
        List of collapse operators (each a 2D array).
    rho0 : ndarray
        Initial density matrix (2D array).
    t_final : float
        Final time for the evolution.
    dt : float
        Time step for the evolution.

    Returns:
    times : ndarray
        Array of time points.
    rhos : list of ndarray
        List of density matrices at each time point.
    """
    num_steps = int(t_final / dt)
    times = np.linspace(0, t_final, num_steps + 1)
    rhos = [rho0]

    M_list = [None] + [c_op * np.sqrt(dt) for c_op in c_ops_list]  # Kraus operators
    J = np.zeros_like(M_list[1])
    for M in M_list[1:]:
        J += M.conj().T @ M
    M_list[0] = np.eye(H.shape[0]) - 1j * dt * (H - 1j/2 * J)  # No-jump operator

    for step in range(num_steps):
        rho = rhos[-1]
        # Unitary evolution
        for M in M_list:
            rho += M @ rho @ M.conj().T

        rhos.append(rho)

    return times, rhos
