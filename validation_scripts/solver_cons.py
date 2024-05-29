# solve the optimization problem
import copy

import numpy as np
import multiprocessing as mp
import time
import concurrent.futures
from itertools import product, zip_longest, repeat
import torch.nn as nn
from odeintw import odeintw
import torch
import utils
from reachability.validation_scripts.datapoint_collect import check_feasibility


num_steps = 1


def xdot(x, t, p, R1, R2):
    A = 0 * np.eye(2)
    B = np.eye(2)

    x1 = x[:, :2]
    x2 = x[:, 2:4]

    ztheta = np.array([0, 1]) * (2 * p - 1)
    # compute K, Phi at the given t
    Q = np.zeros((2, 2))
    PT = np.eye(2)
    tspan = np.linspace(t, 1, 2)
    tspan = np.flip(tspan)
    K1 = odeintw(utils.dPdt, PT, tspan, args=(A, B, Q, R1, None,))

    K1 = K1[-1]  # K at current time

    K2 = odeintw(utils.dPdt, PT, tspan, args=(A, B, Q, R2, None,))

    K2 = K2[-1]

    # Phi
    PhiT = np.eye(2)
    Phi = odeintw(utils.dPhi, PhiT, tspan, args=(A,))
    Phi = Phi[-1]  # Phi at current time

    u = -np.linalg.inv(R1) @ B.T @ K1 @ x1.T + np.linalg.inv(R1) @ B.T @ K1 @ Phi @ (ztheta.T)
    d = -np.linalg.inv(R2) @ B.T @ K2 @ x2.T + np.linalg.inv(R2) @ B.T @ K2 @ Phi @ (ztheta.T)

    x1_dot = A @ x1.T + B @ u
    x2_dot = A @ x2.T + B @ d

    return np.concatenate((x1_dot.T, x2_dot.T), axis=1)


def get_optimal_u(x, p, R, K, Phi, t, total_steps):
    index = int((1 - t) * total_steps)
    try:
        ztheta = torch.tensor([0, 1, 0, 0]) * (2 * p - 1).to(torch.float32)
    except:
        ztheta = np.array([0, 1, 0, 0]) * (2 * p - 1)

    u = utils.get_analytical_u(K[index, :, :], R, Phi[index, :, :],
                               x[..., :4].T, ztheta.T)

    return u

def compute_minimax(p, curr_x, model, R1, R2, K1, K2, Phi, t, t_prev, total=10, game='cons', ux_high=6,
                 uy_high=12, dx_high=6, dy_high=4, DT=0.1):
    p_next = p * torch.ones((1, 1))
    g1 = utils.GOAL_1
    g2 = utils.GOAL_2

    G = [g1, g2]

    X = curr_x.reshape(1, -1)
    x1 = X[:, :-1]
    x2 = torch.cat((X[:, 4:8], X[:, :4]), dim=1)
    u1 = get_optimal_u(x1.numpy(), p_next.numpy(), R1, K1, Phi, t_prev, total)
    d1 = get_optimal_u(x2.numpy(), p_next.numpy(), R2, K2, Phi, t_prev, total)

    x_prime_1 = torch.from_numpy(utils.go_forward(x1, u1, d1, dt=DT))

    coords_ = x_prime_1
    coords_ = utils.normalize_to_max(coords_, ux_high, uy_high, dx_high, dy_high)

    infeasible = check_feasibility(t, coords_.to(torch.float32))

    if infeasible.all():
        x_next = utils.point_dyn(X, ux_high, uy_high, dx_high, dy_high, dt=DT, n=10)
        X_next = torch.from_numpy(utils.make_pairs(x_next[:, :4], x_next[:, 4:8], 10 * 10))
        X_next = utils.normalize_to_max(X_next, ux_high, uy_high, dx_high, dy_high)

        infeasible_ = check_feasibility(t, X_next.to(torch.float32)).squeeze()
        p_next = p * torch.ones_like(X_next[:, 0]).reshape(-1, 1)

        x_next_1 = torch.hstack((X_next, p_next))

        if t == 0:
            v_next_1 = utils.final_cost(x_next_1[:, :2], x_next_1[:, 4:6], G, p_next.reshape(-1, 1).numpy(),
                                        game=game).reshape(-1, 100, 100) + \
                       DT * utils.inst_cost(ux_high, uy_high, dx_high, dy_high, R1, R2, n=10).reshape(-1, 100, 100)

            v_next_1[infeasible_.reshape(-1, 100, 100)] = utils.PENALTY

            v_next_1 = np.min(np.max(v_next_1, 2), 1)
        else:
            coords_1 = x_next_1


            v_next_1 = model({'coords': coords_1.to(torch.float32)})['model_out'].detach().numpy().reshape(-1, 100,
                                                                                                           100) + \
                       DT * utils.inst_cost(ux_high, uy_high, dx_high, dy_high, R1, R2, n=10).reshape(-1, 100, 100)

            v_next_1[infeasible_.reshape(-1, 100, 100)] = utils.PENALTY

            v_next_1 = np.min(np.max(v_next_1, 2), 1)

        return v_next_1
    else:
        x_prime_1 = torch.from_numpy(utils.go_forward(x1, u1, d1, dt=DT))


        x_next_1 = torch.hstack((x_prime_1, p_next))


        # for checking values
        # p = np.linspace(0, 1, 11)
        # u1s = get_optimal_u(x1.numpy(), p.reshape(-1, 1), R1, K1, Phi, t_prev, total)
        # d1s = get_optimal_u(x2.numpy(), p.reshape(-1, 1), R2, K2, Phi, t_prev, total)
        # xp = utils.go_forward(x1, u1s, d1s)
        # coord = np.concatenate((xp, p.reshape(-1, 1)), axis=1)
        # coords = torch.from_numpy(coord)
        # coords = utils.normalize_to_max(coords, ux_high, uy_high, dx_high, dy_high)
        # vs = model({'coords': coords.to(torch.float32)})['model_out'].detach().numpy().reshape(-1, ) + DT * (
        #                        np.sum(np.multiply(np.diag(R1), u1s ** 2), axis=-1) -
        #                        np.sum(np.multiply(np.diag(R2), d1s ** 2), axis=-1))
        # import matplotlib.pyplot as plt
        # plt.plot(p, vs)
        # plt.show
        # plt.plot(vs, alpha=(1-t))

        if t == 0:
            v_next_1 = utils.final_cost(x_next_1[:, :2], x_next_1[:, 4:6], G, p_next.reshape(-1, 1).numpy(),
                                        game=game).reshape(-1, ) + DT * (
                               np.sum(np.multiply(np.diag(R1), u1 ** 2), axis=-1) -
                               np.sum(np.multiply(np.diag(R2), d1 ** 2), axis=-1))


        else:
            coords_1 = x_next_1
            coords_1 = utils.normalize_to_max(coords_1, ux_high, uy_high, dx_high, dy_high)


            v_next_1 = model({'coords': coords_1.to(torch.float32)})['model_out'].detach().numpy().reshape(-1, ) + \
                       DT * (np.sum(np.multiply(np.diag(R1), u1 ** 2), axis=-1) -
                             np.sum(np.multiply(np.diag(R2), d1 ** 2), axis=-1))

        return v_next_1


def optimization(p, v_curr, curr_x, t, model, R1, R2, K1, K2, Phi, t_prev, DT=0.1, game='cons', total=10, ux_high=6,
                 uy_high=12, dx_high=6, dy_high=4):
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
        # if lam_1 == 1:
        #     p_2 = 1 - p_1
        # else:
        #     p_2 = (p - lam_1 * p_1) / (lam_2)

        g1 = utils.GOAL_1
        g2 = utils.GOAL_2

        G = [g1, g2]

        # ts = np.around(np.linspace(0, 1, 11), 2)
        # t_step = int(np.where(ts == t)[0] + 1)
        # ts = np.around(np.linspace(0, 1, total + 1), 3)
        # ts = np.flip(ts)

        # t_step = np.where(ts == np.round(t, 3))[0].item()  # in forward time

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

        coords_ = torch.cat((x_prime_1, x_prime_2))
        coords_ = utils.normalize_to_max(coords_, ux_high, uy_high, dx_high, dy_high)

        infeasible = check_feasibility(t, coords_.to(torch.float32))

        if infeasible.all():
            x_next = utils.point_dyn(X, ux_high, uy_high, dx_high, dy_high, dt=DT, n=10)
            X_next = torch.from_numpy(utils.make_pairs(x_next[:, :4], x_next[:, 4:8], 10 * 10))
            X_next = utils.normalize_to_max(X_next, ux_high, uy_high, dx_high, dy_high)

            infeasible_ = check_feasibility(t, X_next.to(torch.float32)).squeeze()
            p_next_1 = p_1 * torch.ones_like(X_next[:, 0]).reshape(-1, 1)
            p_next_2 = p_2 * torch.ones_like(X_next[:, 0]).reshape(-1, 1)

            x_next_1 = torch.hstack((X_next, p_next_1))
            x_next_2 = torch.hstack((X_next, p_next_2))

            if t == 0:
                v_next_1 = utils.final_cost(x_next_1[:, :2], x_next_1[:, 4:6], G, p_next_1.reshape(-1, 1).numpy(),
                                            game=game).reshape(-1, 100, 100) + \
                           DT * utils.inst_cost(ux_high, uy_high, dx_high, dy_high, R1, R2, n=10).reshape(-1, 100, 100)

                v_next_1[infeasible_.reshape(-1, 100, 100)] = utils.PENALTY

                v_next_1 = np.min(np.max(v_next_1, 2), 1)

                v_next_2 = utils.final_cost(x_next_2[:, :2], x_next_2[:, 4:6], G, p_next_2.reshape(-1, 1).numpy(),
                                            game=game).reshape(-1, 100, 100) + \
                           DT * utils.inst_cost(ux_high, uy_high, dx_high, dy_high, R1, R2, n=10).reshape(-1, 100, 100)

                v_next_2[infeasible_.reshape(-1, 100, 100)] = utils.PENALTY

                v_next_2 = np.min(np.max(v_next_2, 2), 1)
            else:
                coords_1 = x_next_1
                coords_2 = x_next_2

                v_next_1 = model({'coords': coords_1.to(torch.float32)})['model_out'].detach().numpy().reshape(-1, 100,
                                                                                                               100) + \
                           DT * utils.inst_cost(ux_high, uy_high, dx_high, dy_high, R1, R2, n=10).reshape(-1, 100, 100)

                v_next_1[infeasible_.reshape(-1, 100, 100)] = utils.PENALTY

                v_next_1 = np.min(np.max(v_next_1, 2), 1)

                v_next_2 = model({'coords': coords_2.to(torch.float32)})['model_out'].detach().numpy().reshape(-1, 100,
                                                                                                               100) + \
                           DT * utils.inst_cost(ux_high, uy_high, dx_high, dy_high, R1, R2, n=10).reshape(-1, 100, 100)

                v_next_2[infeasible_.reshape(-1, 100, 100)] = utils.PENALTY

                v_next_2 = np.min(np.max(v_next_2, 2), 1)


            v_next[0] = v_next_1
            v_next[1] = v_next_2

            return abs((v_curr - np.matmul(lam_j.T, v_next)).item())
        else:
            x_prime_1 = torch.from_numpy(utils.go_forward(x1, u1, d1, dt=DT))
            x_prime_2 = torch.from_numpy(utils.go_forward(x1, u2, d2, dt=DT))

            x_next_1 = torch.hstack((x_prime_1, p_next_1))
            x_next_2 = torch.hstack((x_prime_2, p_next_2))


            # for checking values
            # p = np.linspace(0, 1, 11)
            # u1s = get_optimal_u(x1.numpy(), p.reshape(-1, 1), R1, K1, Phi, t_prev, total)
            # d1s = get_optimal_u(x2.numpy(), p.reshape(-1, 1), R2, K2, Phi, t_prev, total)
            # xp = utils.go_forward(x1, u1s, d1s)
            # coord = np.concatenate((xp, p.reshape(-1, 1)), axis=1)
            # coords = torch.from_numpy(coord)
            # coords = utils.normalize_to_max(coords, ux_high, uy_high, dx_high, dy_high)
            # vs = model({'coords': coords.to(torch.float32)})['model_out'].detach().numpy().reshape(-1, ) + DT * (
            #                        np.sum(np.multiply(np.diag(R1), u1s ** 2), axis=-1) -
            #                        np.sum(np.multiply(np.diag(R2), d1s ** 2), axis=-1))
            # import matplotlib.pyplot as plt
            # plt.plot(p, vs)
            # plt.scatter(0.5, v_curr)
            # plt.show()
            # plt.plot(vs, alpha=(1-t))


            if t == 0:
                v_next_1 = utils.final_cost(x_next_1[:, :2], x_next_1[:, 4:6], G, p_next_1.reshape(-1, 1).numpy(),
                                            game=game).reshape(-1, ) + DT * (
                                   np.sum(np.multiply(np.diag(R1), u1 ** 2), axis=-1) -
                                   np.sum(np.multiply(np.diag(R2), d1 ** 2), axis=-1))

                v_next_2 = utils.final_cost(x_next_2[:, :2], x_next_2[:, 4:6], G, p_next_2.reshape(-1, 1).numpy(),
                                            game=game).reshape(-1, ) + DT * (
                                   np.sum(np.multiply(np.diag(R1), u2 ** 2), axis=-1) -
                                   np.sum(np.multiply(np.diag(R2), d2 ** 2), axis=-1))

            else:
                coords_1 = x_next_1
                coords_1 = utils.normalize_to_max(coords_1, ux_high, uy_high, dx_high, dy_high)

                coords_2 = x_next_2
                coords_2 = utils.normalize_to_max(coords_2, ux_high, uy_high, dx_high, dy_high)

                v_next_1 = model({'coords': coords_1.to(torch.float32)})['model_out'].detach().numpy().reshape(-1, ) + \
                           DT * (np.sum(np.multiply(np.diag(R1), u1 ** 2), axis=-1) -
                                 np.sum(np.multiply(np.diag(R2), d1 ** 2), axis=-1))
                v_next_2 = model({'coords': coords_2.to(torch.float32)})['model_out'].detach().numpy().reshape(-1, ) + \
                           DT * (np.sum(np.multiply(np.diag(R1), u2 ** 2), axis=-1) -
                                 np.sum(np.multiply(np.diag(R2), d2 ** 2), axis=-1))

            v_next[0] = v_next_1
            v_next[1] = v_next_2

            return abs((v_curr - np.matmul(lam_j.T, v_next)).item())


    # lam = np.linspace(0, 0.num_steps, 1)
    # first check if minmax is greater
    minimax_p = compute_minimax(p, curr_x, model, R1, R2, K1, K2, Phi, t, t_prev)

    if v_curr > minimax_p:
        p_1 = p
        l_1 = 1
        l_2 = 0
        p_2 = 0
        res = l_1, p_1, p_2
        advantage = 0
    else:
        advantage = minimax_p.item() - v_curr.item()
        lam = np.linspace(0, 1, 11)
        ps = np.linspace(0, 1, 11)
        grid = product(lam, ps, ps)


        reduced = filter(constraint, grid)
        res = min(reduced, key=objective)

        l_1, p_1, p_2 = res

    g1 = utils.GOAL_1
    g2 = utils.GOAL_2

    G = [g1, g2]

    X = curr_x.reshape(1, -1)

    p_next_1 = p_1 * torch.ones((1, 1))
    p_next_2 = p_2 * torch.ones((1, 1))

    x1 = X[:, :-1]
    x2 = torch.cat((X[:, 4:8], X[:, :4]), dim=1)

    u1 = get_optimal_u(x1.numpy(), p_next_1.numpy(), R1, K1, Phi, t_prev, total)
    d1 = get_optimal_u(x2.numpy(), p_next_1.numpy(), R2, K2, Phi, t_prev, total)

    u2 = get_optimal_u(x1.numpy(), p_next_2.numpy(), R1, K1, Phi, t_prev, total)
    d2 = get_optimal_u(x2.numpy(), p_next_2.numpy(), R2, K2, Phi, t_prev, total)

    x_prime_1 = torch.from_numpy(utils.go_forward(x1, u1, d1, dt=DT))
    x_prime_2 = torch.from_numpy(utils.go_forward(x1, u2, d2, dt=DT))

    coords_ = torch.cat((x_prime_1, x_prime_2))
    coords_ = utils.normalize_to_max(coords_, ux_high, uy_high, dx_high, dy_high)
    infeasible = check_feasibility(t, coords_.to(torch.float32))

    if ~infeasible.any():
        return res, u1, u2, d1, d2, advantage

    else:
        print(u1, d1)
        print(u2, d2)
        print(p_1, p_2)
        print("Above LQR controls will lead to unsafe zone! ")


        x_next = utils.point_dyn(X, ux_high, uy_high, dx_high, dy_high, dt=DT, n=10)
        X_next = torch.from_numpy(utils.make_pairs(x_next[:, :4], x_next[:, 4:8], 10 * 10))
        X_next = utils.normalize_to_max(X_next, ux_high, uy_high, dx_high, dy_high)

        infeasible_ = check_feasibility(t, X_next.to(torch.float32))

        p_next_1 = p_1 * torch.ones_like(X_next[:, 0]).reshape(-1, 1)
        p_next_2 = p_2 * torch.ones_like(X_next[:, 0]).reshape(-1, 1)

        x_next_1 = torch.hstack((X_next, p_next_1))
        x_next_2 = torch.hstack((X_next, p_next_2))

        if t == 0:
            v_next_1 = utils.final_cost(x_next_1[:, :2], x_next_1[:, 4:6], G, p_next_1.reshape(-1, 1).numpy(),
                                        game=game).reshape(-1, 100, 100) + \
                       DT * utils.inst_cost(ux_high, uy_high, dx_high, dy_high, R1, R2, n=10).reshape(-1, 100, 100)

            v_next_1[infeasible_.reshape(-1, 100, 100)] = utils.PENALTY

            v_next_2 = utils.final_cost(x_next_2[:, :2], x_next_2[:, 4:6], G, p_next_2.reshape(-1, 1).numpy(),
                                        game=game).reshape(-1, 100, 100) + \
                       DT * utils.inst_cost(ux_high, uy_high, dx_high, dy_high, R1, R2, n=10).reshape(-1, 100, 100)

            v_next_2[infeasible_.reshape(-1, 100, 100)] = utils.PENALTY

        else:
            coords_1 = x_next_1
            coords_2 = x_next_2

            v_next_1 = model({'coords': coords_1.to(torch.float32)})['model_out'].detach().numpy().reshape(-1, 100, 100) + \
                       DT * utils.inst_cost(ux_high, uy_high, dx_high, dy_high, R1, R2, n=10).reshape(-1, 100, 100)

            v_next_1[infeasible_.reshape(-1, 100, 100)] = utils.PENALTY

            v_next_2 = model({'coords': coords_2.to(torch.float32)})['model_out'].detach().numpy().reshape(-1, 100, 100) + \
                       DT * utils.inst_cost(ux_high, uy_high, dx_high, dy_high, R1, R2, n=10).reshape(-1, 100, 100)

            v_next_2[infeasible_.reshape(-1, 100, 100)] = utils.PENALTY
        # for debugging
        # import pandas as pd
        # df = pd.DataFrame(v_next_1.reshape(100, 100))
        # df.to_csv('test.csv')

        u1 = np.argmin(np.max(v_next_1, 2))
        u2 = np.argmin(np.max(v_next_2, 2))

        d1 = np.argmax(v_next_1, 2).reshape(-1, )[u1]
        d2 = np.argmax(v_next_2, 2).reshape(-1, )[u2]

        return res, u1, u2, d1, d2, advantage
