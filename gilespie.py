"""
Python port of the Julia module `Gillespie`.

Original Julia module:
    - verify_working()
    - find_nearest(a, x)
    - gillespie(...)
    - state_at_time_on_trajectory(...)
    - expectation_at_time_on_trajectory(...)   # has a known bug in original
    - compute_states_at_times(...)
    - compute_expectation_values_at_times(...) # has a known bug in original
    - state_at_time_on_trajectory_recomputing_V(...)

Dependencies:
    - numpy
    - scipy (for matrix exponentials)
    - tqdm (for progress bar)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm
from tqdm.auto import tqdm
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
#  verify_working
# ---------------------------------------------------------------------------

def verify_working() -> bool:
    """
    verify_working()

    Verifies whether the library has been imported correctly.
    When called, prints a version string and returns True.

    Returns
    -------
    bool
        True
    """
    print("The package has been imported correctly, version 0.2b (Python port)")
    return True


# ---------------------------------------------------------------------------
#  find_nearest
# ---------------------------------------------------------------------------

def find_nearest(a: np.ndarray, x: float) -> np.ndarray:
    """
    find_nearest(a, x)

    Returns indices in sorted array `a` (1D) that are closest to `x`.

    This is a simplified version tailored to the usage pattern in this module:
    `a` is a strictly increasing time grid (no duplicates). For such a case,
    the nearest point is unique. We return it as a length-1 array of indices.

    Parameters
    ----------
    a : np.ndarray
        1D sorted array.
    x : float
        Value to match.

    Returns
    -------
    np.ndarray
        Array of indices of the closest element(s). In this implementation,
        it is always of length 1.
    """
    a = np.asarray(a)
    if a.size == 0:
        return np.array([], dtype=int)

    idx = int(np.argmin(np.abs(a - x)))
    return np.array([idx], dtype=int)


# ---------------------------------------------------------------------------
#  gillespie
# ---------------------------------------------------------------------------

def gillespie(
    H: np.ndarray,
    M_l: List[np.ndarray],
    psi0: np.ndarray,
    t_final: float,
    dt: float,
    number_trajectories: int,
    verbose: bool = False
) -> Tuple[List[List[Dict[str, Any]]], List[np.ndarray], np.ndarray]:
    """
    gillespie(
        H,
        M_l,
        psi0,
        t_final,
        dt,
        number_trajectories,
        verbose=False
    )

    Simulates the jumps, according to the Gillespie algorithm, for the given
    dynamics.

    Parameters
    ----------
    H : np.ndarray
        Hamiltonian matrix (complex, shape (d, d)).
    M_l : list of np.ndarray
        List of jump operators (each complex, shape (d, d)).
    psi0 : np.ndarray
        Initial state vector of the system (complex, shape (d,)).
    t_final : float
        Final time of the evolution.
    dt : float
        Time increment considered.
    number_trajectories : int
        Number of trajectories of the simulation.
    verbose : bool, optional
        If True, prints more output. For large simulations this can produce
        a lot of text.

    Returns
    -------
    trajectories_results : list of list of dict
        List of trajectories, each trajectory is a list of dictionaries with:
            "AbsTime"       : float
            "TimeSinceLast" : float
            "JumpChannel"   : int or None
            "psiAfter"      : np.ndarray (state vector)
    V : list of np.ndarray
        List of pre-computed no-jump non-Hermitian evolution operators.
    t_range : np.ndarray
        Array of times at which the V operators are computed.
    """
    # Time range 0:dt:t_final (inclusive-style, like Julia StepRangeLen)
    t_range = np.arange(0.0, t_final + 0.5 * dt, dt, dtype=float)

    # Constructs the overall jump operator J = sum M†M.
    J = np.zeros_like(M_l[0], dtype=np.complex128)
    for M in M_l:
        J += M.conj().T @ M

    # Effective (non-Hermitian) Hamiltonian.
    He = H - 1j / 2.0 * J

    # Constructs the no-jump evolution operators and Qs(t) = V† J V.
    V: List[np.ndarray] = []
    Qs: List[np.ndarray] = []
    for t in t_range:
        ev_op = expm(-1j * He * t)
        V.append(ev_op)
        nsd_wtd = ev_op.conj().T @ J @ ev_op
        Qs.append(nsd_wtd)

    # Prints the matrix norm for the latest Qs.
    error = norm(Qs[-1])
    print(f"-> Truncation error given by norm of latest Qs matrix: {error}")

    # List for the results: trajectories_results[trajectory][event_dict]
    trajectories_results: List[List[Dict[str, Any]]] = []

    # Cycle over the trajectories.
    for _ in tqdm(range(number_trajectories), desc="Gillespie evolution..."):
        # Initial state.
        psi = np.asarray(psi0, dtype=np.complex128).copy()
        # Absolute time.
        tau = 0.0

        results: List[Dict[str, Any]] = []
        dict_initial: Dict[str, Any] = {
            "AbsTime": 0.0,
            "TimeSinceLast": 0.0,
            "JumpChannel": None,
            "psiAfter": psi.copy()
        }
        results.append(dict_initial)

        while tau < t_final:
            dict_jump: Dict[str, Any] = {}

            # Compute the waiting time distribution, exploiting the pre-computed part.
            Ps: List[float] = []
            for Q in Qs:
                # ψ' * Q * ψ
                wtd = np.real(np.vdot(psi, Q @ psi))
                Ps.append(wtd)

            Ps_arr = np.array(Ps, dtype=float)
            total_P = Ps_arr.sum()
            if total_P <= 0:
                # In the Julia implementation this would likely error; here we stop.
                break
            probs_t = Ps_arr / total_P

            # Sample from the waiting time distribution.
            n_T = int(np.random.choice(len(t_range), p=probs_t))

            # Increase the absolute time.
            delta_t = float(t_range[n_T])
            tau += delta_t
            dict_jump.update({
                "AbsTime": tau,
                "TimeSinceLast": delta_t,
            })

            # Update the state with no-jump evolution.
            psi = V[n_T] @ psi

            # Chooses where to jump.
            weights: List[float] = []
            for M in M_l:
                # ψ' * M' * M * ψ
                w = np.real(np.vdot(psi, (M.conj().T @ M) @ psi))
                weights.append(w)

            weights_arr = np.array(weights, dtype=float)
            total_w = weights_arr.sum()
            if total_w <= 0:
                break
            probs_jump = weights_arr / total_w

            n_jump = int(np.random.choice(len(M_l), p=probs_jump))
            dict_jump["JumpChannel"] = n_jump

            # Update the state after the jump.
            psi = M_l[n_jump] @ psi
            norm_state = norm(psi)
            if norm_state == 0:
                break
            # Renormalize the state.
            psi = psi / norm_state
            dict_jump["psiAfter"] = psi.copy()

            if verbose:
                print(dict_jump)

            results.append(dict_jump)

        trajectories_results.append(results)

    return trajectories_results, V, t_range


# ---------------------------------------------------------------------------
#  state_at_time_on_trajectory
# ---------------------------------------------------------------------------

def state_at_time_on_trajectory(
    t_range: np.ndarray,
    relevant_times: np.ndarray,
    V: List[np.ndarray],
    trajectory_data: List[Dict[str, Any]],
) -> List[np.ndarray]:
    """
    state_at_time_on_trajectory(
        t_range,
        relevant_times,
        V,
        trajectory_data
    )

    Taking as input the output of the `gillespie` function, fills the gaps
    between the jumps.

    Parameters
    ----------
    t_range : np.ndarray
        Times at which the V operators are computed.
    relevant_times : np.ndarray
        Times at which the state has to be computed (can also not coincide
        with `t_range`).
    V : list of np.ndarray
        List of non-Hermitian evolution operators, computed at times
        specified in `t_range`.
    trajectory_data : list of dict
        A list of dictionaries in the form output by `gillespie`.

    Returns
    -------
    v_states : list of np.ndarray
        Vector of quantum pure states (state vectors) at each of the times
        requested in `relevant_times`.
    """
    # Creates an array of states.
    v_states: List[np.ndarray] = []

    # Array of jump times.
    jump_times = np.array([ev["AbsTime"] for ev in trajectory_data], dtype=float)
    # Array of states after the jumps.
    psi_after_jumps = [ev["psiAfter"] for ev in trajectory_data]

    # Cycles over the jump times (all except the last one).
    for n_jump in range(len(jump_times) - 1):
        next_jump_time = jump_times[n_jump + 1]
        # Relevant times between this jump and the following one.
        relevant_times_in_interval = [
            t for t in relevant_times if jump_times[n_jump] <= t < next_jump_time
        ]
        # For each such time:
        for t_abs in relevant_times_in_interval:
            psi = psi_after_jumps[n_jump]
            delta = t_abs - jump_times[n_jump]
            # Closest index in t_range to this delta.
            n_t = find_nearest(t_range, delta)[0]  # 0-based in Python
            # norm = sqrt(ψ' * V[n_t]' * V[n_t] * ψ)
            norm_val = np.sqrt(
                np.real(np.vdot(psi, (V[n_t].conj().T @ V[n_t]) @ psi))
            )
            psi_evolved = V[n_t] @ psi
            psi_evolved = psi_evolved / norm_val
            v_states.append(psi_evolved)

    # Now compute the state for all times after the latest jump.
    last_jump_absolute_time = jump_times[-1]
    relevant_times_after_last_jump = [
        t for t in relevant_times if t >= last_jump_absolute_time
    ]
    for t_abs in relevant_times_after_last_jump:
        psi = psi_after_jumps[-1]
        delta = t_abs - last_jump_absolute_time
        n_t = find_nearest(t_range, delta)[0]
        norm_val = np.sqrt(
            np.real(np.vdot(psi, (V[n_t].conj().T @ V[n_t]) @ psi))
        )
        psi_evolved = V[n_t] @ psi
        psi_evolved = psi_evolved / norm_val
        v_states.append(psi_evolved)

    return v_states


# ---------------------------------------------------------------------------
#  expectation_at_time_on_trajectory  (KNOWN BUG AS IN ORIGINAL)
# ---------------------------------------------------------------------------

def expectation_at_time_on_trajectory(
    t_range: np.ndarray,
    relevant_times: np.ndarray,
    V: List[np.ndarray],
    trajectory_data: List[Dict[str, Any]],
    E_l: List[np.ndarray]
) -> List[np.ndarray]:
    """
    expectation_at_time_on_trajectory(
        t_range,
        relevant_times,
        V,
        trajectory_data,
        E_l
    )

    Takes as input the output of the `gillespie` function and computes the
    expectation values of operators along the trajectory.

    NOTE (from original Julia code): THIS FUNCTION HAS A RELEVANT BUG,
    STILL UNDER TEST.

    Parameters
    ----------
    t_range : np.ndarray
        Times at which the V operators are computed.
    relevant_times : np.ndarray
        Times at which the state has to be computed (can also not coincide
        with `t_range`).
    V : list of np.ndarray
        List of non-Hermitian evolution operators, computed at times
        specified in `t_range`.
    trajectory_data : list of dict
        List of dictionaries in the form output by `gillespie`.
    E_l : list of np.ndarray
        List of Hermitian measurement operators of which the expectation
        value has to be computed.

    Returns
    -------
    expectations_v : list of np.ndarray
        expectations_v[n_E][n_t] is the expectation value for operator
        indexed by n_E at time index n_t. (Buggy index logic is preserved.)
    """
    # Creates an array of expectation values for each operator.
    expectations_v: List[np.ndarray] = []
    for _E in E_l:
        v = np.zeros(len(relevant_times), dtype=float)
        expectations_v.append(v)

    # Jump times and states after jumps.
    jump_times = np.array([ev["AbsTime"] for ev in trajectory_data], dtype=float)
    psi_after_jumps = [ev["psiAfter"] for ev in trajectory_data]

    # Cycles over the jump times (except last one).
    for n_jump in range(len(jump_times) - 1):
        next_jump_time = jump_times[n_jump + 1]
        relevant_times_in_interval = [
            t for t in relevant_times if jump_times[n_jump] <= t < next_jump_time
        ]
        for t_abs in relevant_times_in_interval:
            psi = psi_after_jumps[n_jump]
            delta = t_abs - jump_times[n_jump]
            n_t = find_nearest(t_range, delta)[0]

            norm_val = np.sqrt(
                np.real(np.vdot(psi, (V[n_t].conj().T @ V[n_t]) @ psi))
            )
            psi_evolved = V[n_t] @ psi
            psi_evolved = psi_evolved / norm_val

            # Cycles over operators to compute expectation values.
            for n_E, E in enumerate(E_l):
                exp_val = np.vdot(psi_evolved, E @ psi_evolved)
                # NOTE: bug from original: uses n_t as index in relevant_times,
                # which might not match. We keep the behavior.
                expectations_v[n_E][n_t] = float(np.real(exp_val))

    # After the latest jump.
    last_jump_absolute_time = jump_times[-1]
    relevant_times_after_last_jump = [
        t for t in relevant_times if t >= last_jump_absolute_time
    ]
    for t_abs in relevant_times_after_last_jump:
        psi = psi_after_jumps[-1]
        delta = t_abs - last_jump_absolute_time
        n_t = find_nearest(t_range, delta)[0]

        norm_val = np.sqrt(
            np.real(np.vdot(psi, (V[n_t].conj().T @ V[n_t]) @ psi))
        )
        psi_evolved = V[n_t] @ psi
        psi_evolved = psi_evolved / norm_val

        for n_E, E in enumerate(E_l):
            exp_val = np.vdot(psi_evolved, E @ psi_evolved)
            expectations_v[n_E][n_t] = float(np.real(exp_val))

    return expectations_v


# ---------------------------------------------------------------------------
#  compute_states_at_times
# ---------------------------------------------------------------------------

def compute_states_at_times(
    H: np.ndarray,
    M_l: List[np.ndarray],
    psi0: np.ndarray,
    t_final: float,
    dt: float,
    number_trajectories: int,
    verbose: bool = False,
    compute_V_each_step: bool = False
) -> List[List[np.ndarray]]:
    """
    compute_states_at_times(
        H,
        M_l,
        psi0,
        t_final,
        dt,
        number_trajectories,
        verbose=False,
        compute_V_each_step=False
    )

    Function for external access, computes the states at the specified times
    (using both `gillespie` and `state_at_time_on_trajectory` when appropriate).

    Parameters
    ----------
    H : np.ndarray
        System Hamiltonian.
    M_l : list of np.ndarray
        List of jump operators.
    psi0 : np.ndarray
        Initial (pure) state of the system.
    t_final : float
        Final time of the evolution.
    dt : float
        Time step for the evolution.
    number_trajectories : int
        Number of trajectories to be considered.
    verbose : bool, optional
        If True, prints more output.
    compute_V_each_step : bool, optional
        If True, does not re-use pre-computed values for the no-jump evolution
        operator at all steps, but computes the operator directly.

    Returns
    -------
    results : list of list of np.ndarray
        results[n_traj][n_t] is the state on trajectory n_traj at time t_t.
    """
    trajectories_results, V, t_range = gillespie(
        H, M_l, psi0, t_final, dt, number_trajectories, verbose=verbose
    )
    print()  # like the bare println() in Julia

    results: List[List[np.ndarray]] = []

    # Constructs the overall jump operator.
    J = np.zeros_like(M_l[0], dtype=np.complex128)
    for M in M_l:
        J += M.conj().T @ M
    # Effective (non-Hermitian) Hamiltonian.
    He = H - 1j / 2.0 * J

    if compute_V_each_step:
        # Recompute exp(-i He Δt) for each time gap.
        for n_trajectory in tqdm(
            range(len(trajectories_results)),
            desc="Filling in the gaps..."
        ):
            v_states = state_at_time_on_trajectory_recomputing_V(
                t_range,
                t_range,
                trajectories_results[n_trajectory],
                He
            )
            results.append(v_states)
    else:
        for n_trajectory in tqdm(
            range(len(trajectories_results)),
            desc="Filling in the gaps..."
        ):
            v_states = state_at_time_on_trajectory(
                t_range,
                t_range,
                V,
                trajectories_results[n_trajectory]
            )
            results.append(v_states)

    return results


# ---------------------------------------------------------------------------
#  compute_expectation_values_at_times  (KNOWN BUG AS IN ORIGINAL)
# ---------------------------------------------------------------------------

def compute_expectation_values_at_times(
    H: np.ndarray,
    M_l: List[np.ndarray],
    E_l: List[np.ndarray],
    psi0: np.ndarray,
    t_final: float,
    dt: float,
    number_trajectories: int,
    verbose: bool = False
) -> List[List[np.ndarray]]:
    """
    compute_expectation_values_at_times(
        H,
        M_l,
        E_l,
        psi0,
        t_final,
        dt,
        number_trajectories,
        verbose=False
    )

    Function for external access, computes the expectation values of all the
    required operators at the specified times.

    NOTE (from original Julia code): THIS FUNCTION HAS A RELEVANT BUG,
    STILL UNDER TEST.

    Parameters
    ----------
    H : np.ndarray
        System Hamiltonian.
    M_l : list of np.ndarray
        List of jump operators.
    E_l : list of np.ndarray
        List of operators of which the expectation value has to be computed.
    psi0 : np.ndarray
        Initial (pure) state of the system.
    t_final : float
        Final time of the evolution.
    dt : float
        Time step for the evolution.
    number_trajectories : int
        Number of trajectories to be considered.
    verbose : bool, optional
        If True, prints more output.

    Returns
    -------
    results : list of list of np.ndarray
        results[n_traj][n_E][n_t] is the expectation value for operator n_E
        on trajectory n_traj at time index n_t.
    """
    # Gillespie evolution.
    trajectories_results, V, t_range = gillespie(
        H, M_l, psi0, t_final, dt, number_trajectories, verbose=verbose
    )

    results: List[List[np.ndarray]] = []

    # Holes filling and computation of expectation values.
    for n_trajectory in tqdm(
        range(len(trajectories_results)),
        desc="Filling in the gaps..."
    ):
        v_expectations = expectation_at_time_on_trajectory(
            t_range,
            t_range,
            V,
            trajectories_results[n_trajectory],
            E_l
        )
        results.append(v_expectations)

    return results


# ---------------------------------------------------------------------------
#  state_at_time_on_trajectory_recomputing_V
# ---------------------------------------------------------------------------

def state_at_time_on_trajectory_recomputing_V(
    t_range: np.ndarray,
    relevant_times: np.ndarray,
    trajectory_data: List[Dict[str, Any]],
    He: np.ndarray
) -> List[np.ndarray]:
    """
    state_at_time_on_trajectory_recomputing_V(
        t_range,
        relevant_times,
        trajectory_data,
        He
    )

    Like `state_at_time_on_trajectory`, but recomputes the no-jump evolution
    operator exp(-i He Δt) at each step, instead of reusing a precomputed list.

    Parameters
    ----------
    t_range : np.ndarray
        Times at which the (precomputed) operators would be defined.
        Here it's only used as a grid to choose Δt values from.
    relevant_times : np.ndarray
        Times at which the state has to be computed.
    trajectory_data : list of dict
        Trajectory data from `gillespie`.
    He : np.ndarray
        Effective (non-Hermitian) Hamiltonian.

    Returns
    -------
    v_states : list of np.ndarray
        State vectors at all requested times.
    """
    v_states: List[np.ndarray] = []

    jump_times = np.array([ev["AbsTime"] for ev in trajectory_data], dtype=float)
    psi_after_jumps = [ev["psiAfter"] for ev in trajectory_data]

    # Over all jumps except last.
    for n_jump in range(len(jump_times) - 1):
        next_jump_time = jump_times[n_jump + 1]
        relevant_times_in_interval = [
            t for t in relevant_times if jump_times[n_jump] <= t < next_jump_time
        ]
        for t_abs in relevant_times_in_interval:
            psi = psi_after_jumps[n_jump]
            delta = t_abs - jump_times[n_jump]
            ev_op = expm(-1j * He * delta)
            norm_val = np.sqrt(
                np.real(np.vdot(psi, (ev_op.conj().T @ ev_op) @ psi))
            )
            psi_evolved = ev_op @ psi
            psi_evolved = psi_evolved / norm_val
            v_states.append(psi_evolved)

    # After the last jump.
    last_jump_absolute_time = jump_times[-1]
    relevant_times_after_last_jump = [
        t for t in relevant_times if t >= last_jump_absolute_time
    ]
    for t_abs in relevant_times_after_last_jump:
        psi = psi_after_jumps[-1]
        delta = t_abs - last_jump_absolute_time
        ev_op = expm(-1j * He * delta)
        norm_val = np.sqrt(
            np.real(np.vdot(psi, (ev_op.conj().T @ ev_op) @ psi))
        )
        psi_evolved = ev_op @ psi
        psi_evolved = psi_evolved / norm_val
        v_states.append(psi_evolved)

    return v_states
