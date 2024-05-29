import copy

import numpy as np
from tqdm import tqdm
from itertools import product
import torch
import os
import types
import diff_operators

GOAL_1 = (0, 1)
GOAL_2 = (0, -1)

PENALTY = 20

def normalize_to_max(x, v1x_max, v1y_max, v2x_max, v2y_max):

    x1 = torch.clip(x[:, :2], min=-1, max=1)
    x2 = torch.clip(x[:, 4:6], min=-1, max=1)

    v1_x = x[:, 2]
    v1_y = x[:, 3]

    v2_x = x[:, 6]
    v2_y = x[:, 7]

    a = -1
    b = 1

    v1_x_b = -1 + (b - a) * (v1_x + v1x_max) / (v1x_max + v1x_max)
    v1_y_b = -1 + (b - a) * (v1_y + v1y_max) / (v1y_max + v1y_max)

    v2_x_b = -1 + (b - a) * (v2_x + v2x_max) / (v2x_max + v2x_max)
    v2_y_b = -1 + (b - a) * (v2_y + v2y_max) / (v2y_max + v2y_max)

    x_norm = copy.deepcopy(x)
    x_norm[:, :2] = x1
    x_norm[:, 2] = v1_x_b
    x_norm[:, 3] = v1_y_b
    x_norm[:, 4:6] = x2
    x_norm[:, 6] = v2_x_b
    x_norm[:, 7]= v2_y_b

    return x_norm

def normalize_position_highDim(x):
    x1 = torch.clip(x[:, :2], min=-1, max=1)
    x2 = torch.clip(x[:, 4:6], min=-1, max=1)

    x[:, :2] = x1
    x[:, 4:6] = x2

    return x

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def go_forward(x, U, D, dt=0.1, a=0.5):
    ux = U[:, 0].reshape(-1, 1)
    uy = U[:, 1].reshape(-1, 1)

    dx = D[:, 0].reshape(-1, 1)
    dy = D[:, 1].reshape(-1, 1)

    X1 = x[..., :4]
    X2 = x[..., 4:8]

    # for p1
    x1 = X1[..., 0].reshape(-1, 1)
    y1 = X1[..., 1].reshape(-1, 1)
    vx1 = X1[..., 2].reshape(-1, 1)
    vy1 = X1[..., 3].reshape(-1, 1)

    x1dot = vx1
    y1dot = vy1
    vx1dot = ux
    vy1dot = uy

    # a = 1 # in hexner the constant is 1. in point dynamics it should be 0.5

    x1_new = x1 + x1dot * dt + a * ux * dt ** 2
    y1_new = y1 + y1dot * dt + a * uy * dt ** 2
    vx1_new = vx1 + vx1dot * dt
    vy1_new = vy1 + vy1dot * dt

    # for p2
    x2 = X2[..., 0].reshape(-1, 1)
    y2 = X2[..., 1].reshape(-1, 1)
    vx2 = X2[..., 2].reshape(-1, 1)
    vy2 = X2[..., 3].reshape(-1, 1)

    x2dot = vx2
    y2dot = vy2
    vx2dot = dx
    vy2dot = dy

    x2_new = x2 + x2dot * dt + a * dx * dt ** 2
    y2_new = y2 + y2dot * dt + a * dy * dt ** 2
    vx2_new = vx2 + vx2dot * dt
    vy2_new = vy2 + vy2dot * dt

    return np.hstack((x1_new, y1_new, vx1_new, vy1_new, x2_new, y2_new, vx2_new, vy2_new))





def point_dyn(x, ux_max, uy_max, dx_max, dy_max, n=10, dt=0.1):
    """
    Point dynamics with acceleration control for all possible actions
    :param dt: time step
    :param n: num action space
    :param X: Joint state of players
    :param u_max: upper bound for control
    :param d_max: upper bound for control
    :return: new states: [X1, X2, ...., Xn] containing all possible states
    """

    uxs = np.linspace(-ux_max, ux_max, n)
    uys = np.linspace(-uy_max, uy_max, n)
    dxs = np.linspace(-dx_max, dx_max, n)
    dys = np.linspace(-dy_max, dy_max, n)
    us = list(product(uxs, uys))
    ds = list(product(dxs, dys))
    umap = {k: v for (k, v) in enumerate(us)}
    dmap = {k: v for (k, v) in enumerate(ds)}

    U = np.array([i for i in range(n * n)]).reshape(1, -1)
    U = np.repeat(U, x[..., 2].shape[0], axis=0)

    D = np.array([i for i in range(n * n)]).reshape(1, -1)
    D = np.repeat(D, x[..., 2].shape[0], axis=0)

    action_array_u = np.array([umap[i] for i in range(len(umap))])[U]
    action_array_d = np.array([dmap[i] for i in range(len(dmap))])[D]

    x1 = x[:, 0].reshape(-1, 1)
    y1 = x[:, 1].reshape(-1, 1)
    vx1 = x[:, 2].reshape(-1, 1)
    vy1 = x[:, 3].reshape(-1, 1)

    x2 = x[:, 4].reshape(-1, 1)
    y2 = x[:, 5].reshape(-1, 1)
    vx2 = x[:, 6].reshape(-1, 1)
    vy2 = x[:, 7].reshape(-1, 1)

    x1dot = vx1
    y1dot = vy1

    x2dot = vx2
    y2dot = vy2

    vx1dot = action_array_u[:, :, 0]
    vy1dot = action_array_u[:, :, 1]

    vx2dot = action_array_d[:, :, 0]
    vy2dot = action_array_d[:, :, 1]

    x1_new = x1 + x1dot * dt + 0.5 * vx1dot * (dt ** 2)
    y1_new = y1 + y1dot * dt + 0.5 * vy1dot * (dt ** 2)
    vx1_new = vx1 + vx1dot * dt
    vy1_new = vy1 + vy1dot * dt

    x2_new = x2 + x2dot * dt + 0.5 * vx2dot * (dt ** 2)
    y2_new = y2 + y2dot * dt + 0.5 * vy2dot * (dt ** 2)
    vx2_new = vx2 + vx2dot * dt
    vy2_new = vy2 + vy2dot * dt

    X_new = np.hstack((x1_new.reshape(-1, 1), y1_new.reshape(-1, 1), vx1_new.reshape(-1, 1), vy1_new.reshape(-1, 1),
                       x2_new.reshape(-1, 1), y2_new.reshape(-1, 1), vx2_new.reshape(-1, 1),
                       vy2_new.reshape(-1, 1)))

    return X_new


def check_violation(X1, X2, R=0.05):
    """
    Check for state constraint violation.

    :param X1: Player 1's state
    :param X2: Player 2's state
    :param R: Safety radius of player 1
    :return: boolean (constraints violated or not)
    """
    violation = np.linalg.norm(X1 - X2, axis=-1) - R
    # violation[violation >= 0] = 1
    # violation[violation < 0] = 20

    return violation < 0


def inst_cost(ux_max, uy_max, dx_max, dy_max, R1, R2, n=10):
    uxs = np.linspace(-ux_max, ux_max, n)
    uys = np.linspace(-uy_max, uy_max, n)
    dxs = np.linspace(-dx_max, dx_max, n)
    dys = np.linspace(-dy_max, dy_max, n)
    us = list(product(uxs, uys))
    ds = list(product(dxs, dys))

    umap = {k: v for (k, v) in enumerate(us)}
    dmap = {k: v for (k, v) in enumerate(ds)}

    U = np.array([i for i in range(n * n)]).reshape(1, -1)

    D = np.array([i for i in range(n * n)]).reshape(1, -1)

    action_array_u = np.array([umap[i] for i in range(len(umap))])[U]
    action_array_d = np.array([dmap[i] for i in range(len(dmap))])[D]

    loss1 = np.sum(np.multiply(np.diag(R1), action_array_u ** 2), axis=-1)
    loss2 = np.sum(np.multiply(np.diag(R2), action_array_d ** 2), axis=-1)

    payoff = np.sum((list(product(loss1.flatten(), -loss2.flatten()))), axis=1)

    return payoff

def compute_phat_from_network(model):
    """
    Estimate range of p_hat given the value function neural network

    :param model: trained primal value function
    :return: abs(p_hat_max) which determines the range for p_hat -- (-p_hat_max, p_hat_max)
    """
    X = torch.zeros(100000, 8).uniform_(-1, 1)
    p = torch.zeros(100000, 1).uniform_(0, 1)
    X_in = torch.cat((X, p), dim=1)
    X = {'coords': X_in.to(torch.float32)}
    model_out = model(X)
    value = model_out['model_out']
    x = model_out['model_in']
    dv_dx = diff_operators.jacobian(value, x)[0]
    dv_dp = dv_dx[..., -1]  # gradient w.r.t p

    p_hat_2 = value - dv_dp * p
    p_hat_1 = dv_dp + p_hat_2

    p_hat_maxs = [abs(p_hat_1).max().round(), abs(p_hat_2).max().round()]

    return p_hat_maxs


def final_dual_value(p_hats, x1, x2, G, R=0.05, game='cons'):
    """
    Compute dual value at final time
    """

    g1 = np.array(G[0])
    g2 = np.array(G[1])

    final_cost_1 = np.linalg.norm(x1 - g1, axis=1).reshape(-1, 1) ** 2 - np.linalg.norm(x2 - g1, axis=1).reshape(-1, 1)
    final_cost_2 = np.linalg.norm(x1 - g2, axis=1).reshape(-1, 1) ** 2 - np.linalg.norm(x2 - g2, axis=1).reshape(-1, 1)

    final_costs = np.hstack((final_cost_1, final_cost_2))

    final_costs = p_hats - final_costs
    value = np.max(final_costs, axis=1)

    if game == 'cons':
        violation = check_violation(x1, x2, R)  # state constraint violation
        value[violation] = - PENALTY

    return value


def final_cost(X1, X2, G, p, R=0.05, game='cons'):
    """
    Compute the payoff at the final time
    :param X1: State of player 1
    :param G: Goal positions ([g1, g2, ...., gn])
    :param p: [p_1, p_2, ..., p_n] distribution over goals
    :return: scalar cost
    """

    violation = check_violation(X1, X2, R)  # state constraint violation

    assert type(p) == np.ndarray, "p must be a numpy array"

    # we just have two goals
    g1 = np.array(G[0])
    g2 = np.array(G[1])

    dist1 = np.linalg.norm(X1 - g1, axis=1).reshape(-1, 1) ** 2
    dist2 = np.linalg.norm(X1 - g2, axis=1).reshape(-1, 1) ** 2

    # player 2
    dist1_p2 = np.linalg.norm(X2 - g1, axis=1).reshape(-1, 1) ** 2
    dist2_p2 = np.linalg.norm(X2 - g2, axis=1).reshape(-1, 1) ** 2

    # cost = np.multiply(p, dist1) + np.multiply((1 - p), dist2)

    if game == 'cons':
        cost = np.multiply(p, dist1) + np.multiply((1 - p), dist2) - \
               np.multiply(p, dist1_p2) - np.multiply((1 - p),
                                                      dist2_p2) #+ 4  # + 2  # a constant to make the value always +ve
    else:
        cost = np.multiply(p, dist1) + np.multiply((1 - p), dist2) - \
               np.multiply(p, dist1_p2) - np.multiply((1 - p), dist2_p2)

    if game == 'cons':
        # payoff = np.multiply(violation, cost)
        cost[violation] = PENALTY

    return cost


def make_pairs(X1, X2, n):
    """
    Returns a matrix with all possible next states
    :param X1: states of P1
    :param X2: states of P2
    :return: X containing all pairs of (X1, X2)
    """

    # m, n = X1.shape[0], X2.shape[0]
    # m, n = 9, 9
    dim = X2.shape[1]

    # Repeat and tile to create all pairs
    X1_rep = np.repeat(X1, n, axis=0)

    X2_p = X2.reshape(-1, n, dim)
    X2_rep = np.repeat(X2_p, n, axis=0).reshape(-1, dim)

    # Stack the pairs horizontally
    result = np.hstack((X1_rep, X2_rep))

    return result


def convex_hull(points, vex=True):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """
    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    if vex:
        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
    else:
        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return lower[:] if vex else upper[:]


def cav_vex(values, type='vex', num_ps=11):
    lower = True if type == 'vex' else False
    ps = np.linspace(0, 1, num_ps)
    values = np.vstack(values).T
    cvx_vals = np.zeros((values.shape[0], num_ps))
    p = np.linspace(0, 1, num_ps)
    for i in tqdm(range(values.shape[0])):
        value = values[i]
        points = zip(ps, value)
        hull_points = convex_hull(points, vex=lower)
        hull_points = sorted(hull_points)
        x, y = zip(*hull_points)
        num_facets = len(hull_points) - 1
        for k in range(len(p)):
            if p[k] != 1:
                s_idx = [True if x[j] <= p[k] < x[j + 1] else False for j in range(num_facets)]
            else:
                s_idx = [True if x[j] < p[k] <= x[j + 1] else False for j in range(num_facets)]
            assert sum(s_idx) == 1, "p must belong to only one interval, check for bugs!"
            facets = np.array(list(zip(x, x[1:])))
            val_zips = np.array(list(zip(y, y[1:])))
            P = facets[s_idx].flatten()
            vals = val_zips[s_idx].flatten()
            x1, x2 = P
            y1, y2 = vals
            # calculate the value from the equation:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            # cvx_vals[i] = slope * p[i] + intercept
            cvx_vals[i, k] = slope * p[k] + intercept

    return cvx_vals


def get_analytical_u(K, R, Phi, x, ztheta):
    if torch.is_tensor(K):
        B = torch.tensor([[0, 0], [0, 0], [1, 0], [0, 1]]).to(torch.float32)
        u = -torch.linalg.inv(R) @ B.T @ K @ x + torch.linalg.inv(R) @ B.T @ K @ Phi @ (ztheta)
    else:
        B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        u = -np.linalg.inv(R) @ B.T @ K @ x + np.linalg.inv(R) @ B.T @ K @ Phi @ (ztheta)

    return u.T


def get_analytical_u_pos(K, R, Phi, x, ztheta):
    if torch.is_tensor(K):
        B = torch.tensor([[1, 0], [0, 1]]).to(torch.float32)
        u = -torch.linalg.inv(R) @ B.T @ K @ x + torch.linalg.inv(R) @ B.T @ K @ Phi @ (ztheta)
    else:
        B = np.array([[1, 0], [0, 1]])
        u = -np.linalg.inv(R) @ B.T @ K @ x + np.linalg.inv(R) @ B.T @ K @ Phi @ (ztheta)

    return u.T


def dPdt(P, t, A, B, Q, R, S, ):
    n = A.shape[0]
    m = B.shape[1]

    if S is None:
        S = np.zeros((n, m))

    if isinstance(B, types.FunctionType):  # if B is time varying
        B_curr = B(t)
        B = B_curr

    return -(A.T @ P + P @ A - (P @ B + S) @ np.linalg.inv(R) @ (B.T @ P + S.T) + Q)


def dPhi(Phi, t, A):
    return np.dot(A, Phi)


def d(Phi, K, B, R, z):
    ds = np.zeros((len(Phi), 1))
    if isinstance(B, types.FunctionType):
        t_span = np.linspace(0, 1, 10)
        B_temp = np.array([B(i) for i in t_span])
    else:
        B_temp = np.array([B for _ in range(len(Phi))])

    B = B_temp
    for i in range(len(Phi)):
        # ds[i] = z.T @ Phi[i, :, :] @ K[i, :, :] @ B/R @ B.T @ K[i, :, :] @ Phi[i, :, :] @ z
        ds[i] = (z.T @ Phi[i, :, :].T @ K[i, :, :].T @ B[i] @ np.linalg.inv(R) @ B[i].T @ K[i, :, :] @ Phi[i, :, :] @ z)

    return ds


import torch.nn as nn


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
