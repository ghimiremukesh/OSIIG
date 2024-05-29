# solve the optimization problem
import numpy as np
from itertools import product
from scipy.integrate import solve_ivp

import torch
import utils


num_steps = 1


def get_optimal_u(x, p, R, K, Phi, t, total_steps):
    index = int((1 - t) * total_steps)
    try:
        ztheta = torch.tensor([0, 1, 0, 0]) * (2 * p - 1).to(torch.float32)
    except:
        ztheta = np.array([0, 1, 0, 0]) * (2 * p - 1)

    u = utils.get_analytical_u(K[index, :, :], R, Phi[index, :, :],
                               x[..., :4].T, ztheta.T)

    return u

def optimization(p, v_curr, curr_x, t, model, R1, R2, K1, K2, Phi, t_prev, DT=0.1, game='cons', total=10, D1=None,
                 D2=None, v1x_max=6, v1y_max=12, v2x_max=6, v2y_max=4):
    def constraint(var):
        lam_1 = var[0]
        lam_2 = 1 - lam_1
        p_1 = var[1]
        p_2 = var[2]

        return ((lam_1 * p_1 + lam_2 * p_2) == p)

    def objective(var):
        # here, V_curr is the value at the current state
        # V_next is the max min value at the previous state leading to current state
        lam_1 = var[0]
        lam_2 = 1 - lam_1
        p_1 = var[1]
        p_2 = var[2]


        g1 = utils.GOAL_1
        g2 = utils.GOAL_2

        G = [g1, g2]


        lam_j = np.array([[lam_1], [lam_2]])
        v_next = np.zeros((2, 1))

        p_next_1 = p_1 * torch.ones((1, 1))
        p_next_2 = p_2 * torch.ones((1, 1))

        X = curr_x.reshape(1, -1)
        x1 = X[:, :-1]
        x2 = torch.cat((X[:, 4:8], X[:, :4]), dim=1)
        u1 = get_optimal_u(x1.numpy(), p_next_1.numpy(), R1, K1, Phi, t_prev, total)
        d1 = get_optimal_u(x2.numpy(), p_next_1.numpy(), R2, K2, Phi, t_prev, total)


        u2 = get_optimal_u(x1.numpy(), p_next_2.numpy(), R1, K1, Phi, t_prev, total)
        d2 = get_optimal_u(x2.numpy(), p_next_2.numpy(), R2, K2, Phi, t_prev, total)


        x_prime_1 = torch.from_numpy(utils.go_forward(x1, u1, d1, dt=DT))
        x_prime_2 = torch.from_numpy(utils.go_forward(x1, u2, d2, dt=DT))

        x_next_1 = torch.hstack((x_prime_1, p_next_1))
        x_next_2 = torch.hstack((x_prime_2, p_next_2))


        if t == 0:
            v_next_1 = utils.final_cost(x_next_1[:, :2], x_next_1[:, 4:6], G, p_next_1.reshape(-1, 1).numpy(), game=game).reshape(-1, ) + \
                       DT * (np.sum(np.multiply(np.diag(R1), u1 ** 2), axis=-1) - np.sum(np.multiply(np.diag(R2), d1 ** 2), axis=-1)).reshape(-1, )


            v_next_2 = utils.final_cost(x_next_2[:, :2], x_next_2[:, 4:6], G, p_next_2.reshape(-1, 1).numpy(), game=game).reshape(-1, ) + \
                       DT * (np.sum(np.multiply(np.diag(R1), u2 ** 2), axis=-1) - np.sum(np.multiply(np.diag(R2), d2 ** 2), axis=-1)).reshape(-1, )

        else:
            coords_1 = x_next_1
            coords_1 = utils.normalize_to_max(coords_1, v1x_max, v1y_max, v2x_max, v2y_max)

            coords_2 = x_next_2
            coords_2 = utils.normalize_to_max(coords_2, v1x_max, v1y_max, v2x_max, v2y_max)

            v_next_1 = model({'coords': coords_1.to(torch.float32)})['model_out'].detach().numpy().reshape(-1, ) + \
                     DT * (np.sum(np.multiply(np.diag(R1), u1 ** 2), axis=-1) - np.sum(np.multiply(np.diag(R2), d1 ** 2), axis=-1)).reshape(-1, )
            v_next_2 = model({'coords': coords_2.to(torch.float32)})['model_out'].detach().numpy().reshape(-1, ) + \
                       DT * (np.sum(np.multiply(np.diag(R1), u2 ** 2), axis=-1) - np.sum(np.multiply(np.diag(R2), d2 ** 2), axis=-1)).reshape(-1, )

        v_next[0] = v_next_1
        v_next[1] = v_next_2

        return abs((v_curr - np.matmul(lam_j.T, v_next)).item())

    # lam = np.linspace(0, 0.num_steps, 1)
    lam = np.linspace(0, 1, 11)
    ps = np.linspace(0, 1, 11)
    grid = product(lam, ps, ps)


    reduced = filter(constraint, grid)
    res = min(reduced, key=objective)

    l_1, p_1, p_2 = res

    p_next_1 = p_1 * torch.ones((1, 1))
    p_next_2 = p_2 * torch.ones((1, 1))
    X = curr_x.reshape(1, -1)
    x1 = X[:, :-1]
    x2 = torch.cat((X[:, 4:8], X[:, :4]), dim=1)


    if t == 0:
        u1 = get_optimal_u(x1.numpy(), p_next_1.numpy(), R1, K1, Phi, t_prev, total)
        d1 = get_optimal_u(x2.numpy(), p_next_1.numpy(), R2, K2, Phi, t_prev, total)

        u2 = get_optimal_u(x1.numpy(), p_next_2.numpy(), R1, K1, Phi, t_prev, total)
        d2 = get_optimal_u(x2.numpy(), p_next_2.numpy(), R2, K2, Phi, t_prev, total)

    else:
        u1 = get_optimal_u(x1.numpy(), p_next_1.numpy(), R1, K1, Phi, t_prev, total)
        d1 = get_optimal_u(x2.numpy(), p_next_1.numpy(), R2, K2, Phi, t_prev, total)

        u2 = get_optimal_u(x1.numpy(), p_next_2.numpy(), R1, K1, Phi, t_prev, total)
        d2 = get_optimal_u(x2.numpy(), p_next_2.numpy(), R2, K2, Phi, t_prev, total)

    return res, u1, u2, d1, d2
