# simulate the game
import numpy as np
import random

from odeintw import odeintw

import utils
from solver_cons import optimization
import icnn_pytorch_constrained as icnn
import torch
from itertools import product
import matplotlib.pyplot as plt
from reachability.validation_scripts import datapoint_collect

# for lqr strategies
def compute_uncons_system():
    def dPdt(P, t, A, B, Q, R, S, ):
        n = A.shape[0]
        m = B.shape[1]

        if S is None:
            S = np.zeros((n, m))

        return -(A.T @ P + P @ A - (P @ B + S) @ np.linalg.inv(R) @ (B.T @ P + S.T) + Q)

    def dPhi(Phi, t, A):
        return np.dot(A, Phi)

    def d(Phi, K, B, R, z):
        ds = np.zeros((len(Phi), 1))
        B_temp = np.array([B for _ in range(len(Phi))])

        B = B_temp
        for i in range(len(Phi)):
            # ds[i] = z.T @ Phi[i, :, :] @ K[i, :, :] @ B/R @ B.T @ K[i, :, :] @ Phi[i, :, :] @ z
            ds[i] = (z.T @ Phi[i, :, :].T @ K[i, :, :].T @ B[i] @ np.linalg.inv(R) @ B[i].T @ K[i, :, :] @ Phi[i, :,
                                                                                                           :] @ z)
        return ds

    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    Q = np.zeros((4, 4))
    R1 = np.array([[0.05, 0], [0, 0.025]])
    PT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    tspan = np.linspace(0, 1, 11)
    tspan = np.flip(tspan)
    K1 = odeintw(dPdt, PT, tspan, args=(A, B, Q, R1, None,))

    K1 = np.flip(K1, axis=0)

    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    t_span = np.linspace(0, 1, 11)
    t_span = np.flip(t_span)
    PhiT = np.eye(4)

    Phi_sol = odeintw(dPhi, PhiT, t_span, args=(A,))
    Phi_sol = np.flip(Phi_sol, axis=0)

    B2 = B

    # R2 = 0.1 * np.eye(2)
    R2 = np.array([[0.05, 0], [0, 0.1]])
    K2 = odeintw(dPdt, PT, tspan, args=(A, B2, Q, R2, None,))
    K2 = np.flip(K2, axis=0)

    return K1, K2, Phi_sol


# run multiple times for realzns.
fig, axs = plt.subplots(3, 1)
for run in range(1):  # how many simulations
    ux_max = 6
    uy_max = 12
    dx_max = 6
    dy_max = 4

    n = 10

    uxs = np.linspace(-ux_max, ux_max, n)
    uys = np.linspace(-uy_max, uy_max, n)
    dxs = np.linspace(-dx_max, dx_max, n)
    dys = np.linspace(-dy_max, dy_max, n)
    us = list(product(uxs, uys))
    ds = list(product(dxs, dys))
    umap = {k: v for (k, v) in enumerate(us)}
    dmap = {k: v for (k, v) in enumerate(ds)}

    game = 'cons'  # for unconstrained game

    p = 0.5  # probability of selecting goal 1

    # p1_type = random.choices([1, -1], [p, 1 - p])[0]  # choose randomly or select below
    p1_type = -1  # set player type
    type_map = {1: 'Goal 1', -1: 'Goal 2'}

    h = [256] * 10

    DT = 0.1
    total_steps = int(1 / DT)
    r = 0.05  # collision radius

    models = [icnn.SingleBVPNet(in_features=9, out_features=1, type='relu', mode='mlp', hidden_features=h[i],
                                num_hidden_layers=5, dropout=0, r=r) for i in range(total_steps)]

    checkpoints = [f'../trained_models/cons/t_{i}/checkpoints_dir/model_final.pth'
                   for i in range(1, total_steps + 1)]

    loaded_check = [torch.load(checkpoint, map_location=torch.device("cpu")) for checkpoint in checkpoints]

    try:
        model_weights = [loaded['model'] for loaded in loaded_check]
    except:
        model_weights = [loaded for loaded in loaded_check]

    for i in range(len(models)):
        models[i].load_state_dict(model_weights[i])
        models[i].eval()


    K1, K2, Phi = compute_uncons_system()
    R1 = np.array([[0.05, 0], [0, 0.025]])
    R2 = np.array([[0.05, 0], [0, 0.1]])
    states = []
    values = []
    U = []
    D = []
    selection = []  # selection probability
    print(f'Goal is: {type_map[p1_type]}')
    curr_x1 = np.array([-0.5, 0, 0, 0])  # x y pos vel for p1
    curr_x2 = np.array([-0.4, -0.08, 0, 0])  # if type 1 [-0.4, 0.08, 0, 0] only for
    print(f'Current position is : {curr_x1}, {curr_x2} and the belief is {p}\n')
    p_t = p

    t = 1  # backward time
    ts = np.arange(0, 1, DT)

    curr_pos = np.hstack((curr_x1, curr_x2, p_t))

    infeasible = datapoint_collect.check_feasibility(t,
                                                     torch.from_numpy(curr_pos[:-1]).unsqueeze(0).to(torch.float32))

    if infeasible:
        raise ValueError("initial state not feasible! sample new state and run again.")

    states.append(curr_pos)
    times = [t]

    while t >= DT:
        model_idx = int(total_steps * t) - 1
        curr_model = models[model_idx]
        next_model = models[model_idx - 1]
        coords_in = torch.from_numpy(curr_pos).to(torch.float32).unsqueeze(0)
        coords_in = utils.normalize_to_max(coords_in, ux_max, uy_max, dx_max, dy_max)
        feed_to_model = {'coords': coords_in}
        v_curr = curr_model(feed_to_model)['model_out'].detach().cpu().numpy()

        violation = utils.check_violation(coords_in[:, :2].numpy(), coords_in[:, 4:6].numpy())
        if violation:
            v_curr = utils.PENALTY
            raise "Game Over! Constraint violated!"

        values.append(v_curr[0])
        (lam_j, p_1, p_2), u_1, u_2, d_1, d_2, _ = optimization(p_t, v_curr,
                                                             torch.from_numpy(curr_pos).to(torch.float32),
                                                             np.round(t - DT, 3), next_model, R1, R2, K1, K2, Phi,
                                                             t, DT=DT, game=game, total=total_steps, ux_high=ux_max,
                                                             uy_high=uy_max, dx_high=dx_max, dy_high=dy_max)


        print(f'lamda_1 = {lam_j:.2f}, p_1 = {p_1:.2f},  p_2 = {p_2:.2f}\n')

        # action selection for first splitting point (lam_1)
        # calculate probability of each action
        if p1_type == -1:
            p_i = 1 - p_t
            p_1j = 1 - p_1
            p_2j = 1 - p_2
        else:
            p_i = p_t
            p_1j = p_1
            p_2j = p_2

        if lam_j == 1:
            a0_p = 1
            a1_p = 0
        else:
            a0_p = (lam_j * p_1j) / p_i
            a1_p = ((1 - lam_j) * p_2j) / p_i

        print(f'At t = {1 - t:.2f}, P1 with type {type_map[p1_type]} has the following options: \n')
        try:
            print(f'P1 could take action {str(umap[u_1])} with probability {a0_p:.2f} and move belief to {p_1:.2f}')
            print(
                f'P1 could take action {str(umap[u_2])} with probability {a1_p:.2f} and move belief to {p_2:.2f}\n')
        except:
            print(f'P1 could take action {str(u_1)} with probability {a0_p:.2f} and move belief to {p_1:.2f}')
            print(f'P1 could take action {str(u_2)} with probability {a1_p:.2f} and move belief to {p_2:.2f}\n')

        dist = [a0_p, a1_p]
        a_idx = [0, 1]
        action_idx = random.choices(a_idx, dist)[0]

        if action_idx == 0:
            action_1 = u_1
            action_2 = d_1  # best response
            p_t = p_1
        else:
            action_1 = u_2
            action_2 = d_2  # best response
            p_t = p_2

        try:
            action_1 = np.array(umap[action_1]).reshape(1, -1)
            action_2 = np.array(dmap[action_2]).reshape(1, -1)
        except:
            pass

        print(f'P1 chooses action: {action_1} and moves the belief to p_t = {p_t:.2f}')
        print(f'P2 chooses action: {action_2} (using minimax)')

        U.append(action_1)
        D.append(action_2)

        curr_x = utils.go_forward(curr_pos, action_1, action_2, DT)

        curr_x = utils.normalize_position_highDim(torch.from_numpy(curr_x)).squeeze().numpy()

        print(f'The current state is: {curr_x}\n')

        t = np.round(t - DT, 3)
        curr_pos = np.hstack((curr_x, [p_t]))

        feasibility_check_states = utils.normalize_to_max(torch.from_numpy(curr_x).unsqueeze(0),
                                                          ux_max, uy_max, dx_max, dy_max)
        infeasible = datapoint_collect.check_feasibility(t, feasibility_check_states.to(torch.float32))

        states.append(curr_pos)
        times.append(t)

        if infeasible:
            print(f"\nNew state is in the unsafe region of BRT. Game Ends.\n")
            break

    # final time
    g1 = utils.GOAL_1
    g2 = utils.GOAL_2

    G = [g1, g2]
    values.append(
        utils.final_cost(curr_x.reshape(1, -1)[:, :2], curr_x.reshape(1, -1)[:, 4:6], G, np.array([[p_t]]),
                         game=game).squeeze().item())
    values = np.vstack(values)


    states = np.vstack(states)


    # uncomment to plot
    times = [1 - t for t in times]

    x1 = states[:, 0]
    y1 = states[:, 1]
    x2 = states[:, 4]
    y2 = states[:, 5]

    p_t = states[:, -1]


    g1, g2 = utils.GOAL_1, utils.GOAL_2

    marker_angles_1 = np.arctan2(y1, x1)
    marker_angles_2 = np.arctan2(y2, x2)

    U = np.vstack(U)
    D = np.vstack(D)

    axs[2].plot(times[:-1], U[:, 0], label='$u_x$')
    axs[2].plot(times[:-1], U[:, 1], label='$u_y$')
    axs[2].plot(times[:-1], D[:, 0], '-.', label='$d_x$')
    axs[2].plot(times[:-1], D[:, 1], '--', label='$d_y$')
    axs[2].set_xlim([-0.05, 1])
    axs[2].legend()

    axs[0].set_title(f"Goal Selected: {type_map[p1_type]} ")
    if p1_type == -1:
        axs[0].scatter(g1[0], g1[1], marker='o', facecolor='none', edgecolor='magenta')
        axs[0].scatter(g2[0], g2[1], marker='o', facecolor='magenta', edgecolor='magenta')
    else:
        axs[0].scatter(g1[0], g1[1], marker='o', facecolor='magenta', edgecolor='magenta')
        axs[0].scatter(g2[0], g2[1], marker='o', facecolor='none', edgecolor='magenta')

    axs[0].annotate("1", (g1[0] + 0.01, g1[1]))
    axs[0].annotate("2", (g2[0] + 0.01, g2[1]))

    axs[0].scatter(x1[0], y1[0], marker='*', color='red')
    axs[0].scatter(x2[0], y2[0], marker='*', color='blue')
    axs[0].plot(x1, y1, color='red', label='A', marker='o', markersize=2)
    axs[0].plot(x2, y2, color='blue', label='D', marker='o', markersize=2)
    axs[0].set_xlim([-1, 1])
    axs[0].set_ylim([-1, 1])
    axs[0].legend()

    axs[1].plot(np.linspace(0, times[-1], len(times)), p_t, linewidth=2)
    axs[1].set_xlim([0, 1])

    axs[1].set_xlabel('time (t)')
    axs[1].set_ylabel('belief (p_t)')
    axs[1].set_ylim([-0.1, 1])
plt.show()
