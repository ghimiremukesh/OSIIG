import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import numpy as np
import torch
import icnn_pytorch_constrained as icnn
import icnn_pytorch as icnn_uncons
import configargparse
import utils
from itertools import product
from tqdm import tqdm
import scipy.io as scio
from utils import convex_hull
from joblib import Parallel, delayed
from odeintw import odeintw

from reachability.validation_scripts.datapoint_collect import check_feasibility


# Data Collection for Training

NUM_PS = 100
pp = configargparse.ArgumentParser()
pp.add_argument('--time', type=float, default=0.1,
                help='time-step to collect data')
opt = pp.parse_args()

device = torch.device("cpu")

activation = 'relu'


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
            cvx_vals[i, k] = slope * p[k] + intercept

    return cvx_vals

def get_optimal_u(x, p, R, K, Phi, t, total_steps):
    index = int(np.round(1 - t, 2) * total_steps)
    try:
        ztheta = torch.tensor([0, 1, 0, 0]) * (2 * p - 1).to(torch.float32)
    except:
        ztheta = np.array([0, 1, 0, 0]) * (2 * p - 1)

    u = utils.get_analytical_u(K[index, :, :], R, Phi[index, :, :],
                                            x[..., :4].T, ztheta.T)

    return u

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

    R2 = np.array([[0.05, 0], [0, 0.1]])
    K2 = odeintw(dPdt, PT, tspan, args=(A, B2, Q, R2, None,))
    K2 = np.flip(K2, axis=0)

    return K1, K2, Phi_sol


def collect_data(bound, vel_bound_x, vel_bound_y_1, vel_bound_y_2, num_points, t=opt.time):
    activation = 'relu'

    num_points = num_points

    col_radius = 0.05 
    game = 'cons'

    # action bounds and normalizing constants
    ux_high = 6
    uy_high = 12
    
    dx_high = 6
    dy_high = 4


    n_actions = 10 * 10

    g1 = utils.GOAL_1
    g2 = utils.GOAL_2

    t = t
    dt = 0.1

    total_steps = int(1/dt)


    R1 = np.array([[0.05, 0], [0, 0.025]])
    R2 = np.array([[0.05, 0], [0, 0.1]])

    K1, K2, Phi = compute_uncons_system()  # get params for computing optimal u from LQR

    G = [g1, g2]

    ts = np.around(np.arange(dt, 1 + dt, dt), 2)
    t_step = int(np.where(ts == t)[0] + 1)

    NUM_PS = 100

    logging_root = 'logs/'


    torch.manual_seed(0)
    n = 7


    v1x_max = vel_bound_x
    v1y_max = vel_bound_y_1

    v2x_max = vel_bound_x
    v2y_max = vel_bound_y_2

    if v1x_max > 5:
        epsilon_x = 0.1
    else:
        epsilon_x = 0

    if v1y_max > 11:
        epsilon_y1 = 0.7
    else:
        epsilon_y1 = 0

    if v2y_max > 3.5:
        epsilon_y2 = 0.1
    else:
        epsilon_y2 = 0

    while True:
        positions = torch.zeros(n * num_points, 4).uniform_(-bound, bound)
        vel_1_x = torch.zeros(n * num_points, 1).uniform_(-(v1x_max-epsilon_x), (v1x_max-epsilon_x)).reshape(-1, 1)
        vel_1_y = torch.zeros(n * num_points, 1).uniform_(-(v1y_max-epsilon_y1), (v1y_max-epsilon_y1)).reshape(-1, 1)
        vel_1 = torch.cat((vel_1_x, vel_1_y), dim=1)

        vel_2_x = torch.zeros(n * num_points, 1).uniform_(-(v2x_max-epsilon_x), (v2x_max-epsilon_x)).reshape(-1, 1)
        vel_2_y = torch.zeros(n * num_points, 1).uniform_(-(v2y_max-epsilon_y2), (v2y_max-epsilon_y2)).reshape(-1, 1)
        vel_2 = torch.cat((vel_2_x, vel_2_y), dim=1)

        xy = torch.cat((positions[:, :2], vel_1, positions[:, 2:4], vel_2), dim=1)
        xy_f = utils.normalize_to_max(xy, ux_high, uy_high, dx_high, dy_high)
        infeasible = check_feasibility(t, xy_f).squeeze()
        xy = xy[~infeasible]
        if len(xy) >= num_points:
            xy = xy[:num_points, :]
            break
        n += 1 # sample more if fails


    if t == dt:
        vs = []
        ps = np.linspace(0, 1, NUM_PS)

        # temp = []
        for p_each in tqdm(ps):
            p_next = p_each * torch.ones_like(xy[:, 0]).reshape(-1, 1)
            x1 = xy
            x2 = torch.cat((xy[:, 4:8], xy[:, :4]), dim=1)
            u = get_optimal_u(x1.numpy(), p_next.numpy(), R1, K1, Phi, t, total_steps)
            d = get_optimal_u(x2.numpy(), p_next.numpy(), R2, K2, Phi, t, total_steps)
            x_prime = torch.from_numpy(utils.go_forward(xy, u, d, dt=dt, a=0.5)).to(torch.float32)
            x_prime_f = utils.normalize_to_max(x_prime, ux_high, uy_high, dx_high, dy_high)
            v_next = utils.final_cost(x_prime[:, :2], x_prime[:, 4:6], G, p_next.numpy(),
                                      game=game, R=col_radius).reshape(-1, ) + \
                     dt * (np.sum(np.multiply(np.diag(R1), u ** 2), axis=-1) - np.sum(np.multiply(np.diag(R2), d ** 2), axis=-1))

            infeasible = check_feasibility(t-dt, x_prime_f).squeeze()
            if ~infeasible.any():
                if v_next.max() >= 20:
                    idx = np.where(v_next >= 20)
                    infeasible[idx] = True
            if infeasible.any():
                x_inf = xy[infeasible]
                x_inf_next = utils.point_dyn(x_inf, ux_high, uy_high, dx_high, dy_high, dt=dt, n=10)
                X_inf_next = torch.from_numpy(utils.make_pairs(x_inf_next[:, :4], x_inf_next[:, 4:8], n_actions))
                X_inf_next_f = utils.normalize_to_max(X_inf_next, ux_high, uy_high, dx_high, dy_high)
                infeasible_ = check_feasibility(t - dt, X_inf_next_f.to(torch.float32)).squeeze()
                p_next_inf = p_each * torch.ones_like(X_inf_next[:, 0]).reshape(-1, 1)
                v_next_inf = utils.final_cost(X_inf_next[:, :2], X_inf_next[:, 4:6], G, p_next_inf.numpy(), game=game, R=col_radius).reshape(-1,
                                                                                                                 n_actions,
                                                                                                                 n_actions) + \
                         dt * utils.inst_cost(ux_high, uy_high, dx_high, dy_high, R1, R2, n=10).reshape(-1, n_actions, n_actions)

                v_next_inf[infeasible_.reshape(-1, n_actions, n_actions)] = utils.PENALTY

                v_next_inf = np.min(np.max(v_next_inf, 2), 1)
                v_next[infeasible] = v_next_inf

            vs.append(v_next)


        true_v = cav_vex(vs, type='vex', num_ps=NUM_PS).reshape(1, -1, 1)

        ps = torch.linspace(0, 1, 100)
        p = ps.repeat([len(xy), 1]).reshape(-1, 1)
        x = torch.vstack([xy[i].repeat([NUM_PS, 1]) for i in range(len(xy))])
        coords = torch.cat((x, p), dim=1)

        x_prev = coords
        x_prev = utils.normalize_to_max(x_prev, ux_high, uy_high, dx_high, dy_high).detach().cpu().numpy()

        gt = (np.vstack(x_prev), np.vstack(true_v))

    else:
        load_dir = os.path.join(logging_root, f'cons/t_{t_step - 1}/')

        load_dir_uncons = os.path.join(logging_root, f'uncons/t_{t_step}')

        hf = 256

        val_model = icnn.SingleBVPNet(in_features=9, out_features=1, type=activation, mode='mlp',
                                      hidden_features=hf, num_hidden_layers=5, dropout=0)

        val_model_curr = icnn_uncons.SingleBVPNet(in_features=9, out_features=1, type=activation, mode='mlp',
                                                  hidden_features=hf, num_hidden_layers=5, dropout=0)
        
        model_path = os.path.join(load_dir, 'checkpoints_dir', 'model_final.pth')
        model_path_curr = os.path.join(load_dir_uncons, 'checkpoints_dir', 'model_final.pth')

        checkpoint = torch.load(model_path, map_location=device)
        checkpoint_curr = torch.load(model_path_curr, map_location=device)
        try:
            val_model.load_state_dict(checkpoint['model'])
        except:
            val_model.load_state_dict(checkpoint)

        try:
            val_model_curr.load_state_dict(checkpoint_curr['model'])
        except:
            val_model_curr.load_state_dict(checkpoint_curr)

        val_model.eval()
        val_model_curr.eval()

        vs = []
        ps = np.linspace(0, 1, NUM_PS)


        for p_each in tqdm(ps):
            p_next = p_each * torch.ones_like(xy[:, 0]).reshape(-1, 1)
            x1 = xy
            x2 = torch.cat((xy[:, 4:8], xy[:, :4]), dim=1)
            u = get_optimal_u(x1.numpy(), p_next.numpy(), R1, K1, Phi, t, total_steps)
            d = get_optimal_u(x2.numpy(), p_next.numpy(), R2, K2, Phi, t, total_steps)
            x_prime = torch.from_numpy(utils.go_forward(xy, u, d, dt=dt, a=0.5)).to(torch.float32)
            x_prime_f = utils.normalize_to_max(x_prime, ux_high, uy_high, dx_high, dy_high)
            coords = torch.cat((xy, p_next), dim=1)
            coords = utils.normalize_to_max(coords, ux_high, uy_high, dx_high, dy_high)
            coords = {'coords': coords.to(torch.float32)}
            v_next = val_model_curr(coords)['model_out'].detach().numpy().reshape(-1, )

            infeasible = check_feasibility(t-dt, x_prime_f).squeeze()

            if infeasible.any():
                x_inf = xy[infeasible]
                x_inf_next = utils.point_dyn(x_inf, ux_high, uy_high, dx_high, dy_high, dt=dt, n=10)
                X_inf_next = torch.from_numpy(utils.make_pairs(x_inf_next[:, :4], x_inf_next[:, 4:8], n_actions))
                X_inf_next_f = utils.normalize_to_max(X_inf_next, ux_high, uy_high, dx_high, dy_high)
                infeasible_ = check_feasibility(t - dt, X_inf_next_f.to(torch.float32)).squeeze()
                
                p_next_inf = p_each * torch.ones_like(X_inf_next[:, 0]).reshape(-1, 1)
                coords_inf = torch.cat((X_inf_next, p_next_inf), dim=1)
                coords_inf = utils.normalize_to_max(coords_inf, ux_high, uy_high, dx_high, dy_high)
                coords_inf = {'coords': coords_inf.to(torch.float32)}
                v_next_inf = val_model(coords_inf)['model_out'].detach().numpy().reshape(-1, n_actions, n_actions) + \
                         dt * utils.inst_cost(ux_high, uy_high, dx_high, dy_high, R1, R2, n=10).reshape(-1, n_actions, n_actions)

                v_next_inf[infeasible_.reshape(-1, n_actions, n_actions)] = utils.PENALTY
                
            
                v_next_inf = np.min(np.max(v_next_inf, 2), 1)
                v_next[infeasible] = v_next_inf

            vs.append(v_next)

        true_v = utils.cav_vex(vs, type='vex', num_ps=NUM_PS).reshape(1, -1, 1)

        ps = torch.linspace(0, 1, 100)
        p = ps.repeat([len(xy), 1]).reshape(-1, 1)
        x = torch.vstack([xy[i].repeat([NUM_PS, 1]) for i in range(len(xy))])
        coords = torch.cat((x, p), dim=1)


        x_prev = coords
        x_prev = utils.normalize_to_max(x_prev, ux_high, uy_high, dx_high, dy_high).detach().cpu().numpy()

        gt = (np.vstack(x_prev), np.vstack(true_v))
    return gt


if __name__ == '__main__':
    t = opt.time
    bounds = np.linspace(0.1, 1, 50)
    vel_bounds_x = np.linspace(0.1, 6, 50)
    vel_bounds_y_1 = np.linspace(0.1, 12, 50)
    vel_bounds_y_2 = np.linspace(0.1, 4, 50)
    bounds = np.flip(bounds)
    vel_bounds_x = np.flip(vel_bounds_x)
    vel_bounds_y_1 = np.flip(vel_bounds_y_1)
    vel_bounds_y_2 = np.flip(vel_bounds_y_2)
    num_points = np.linspace(100, 100, 50).astype(np.int32)

    delayed_funcs = [delayed(collect_data)(bounds[i], vel_bounds_x[i], vel_bounds_y_1[i], vel_bounds_y_2[i], num_points[i], t) for i in range(50)]
    parallel_pool = Parallel(n_jobs=joblib.cpu_count())

    output = parallel_pool(delayed_funcs)


    states = [output[i][0] for i in range(50)]
    values = [output[i][1] for i in range(50)]

    save_root = f'cons/'
    
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    gt = {'states': np.vstack(states),
          'values': np.vstack(values)}

    scio.savemat(os.path.join(save_root, f'train_data_t_{t:.2f}.mat'), gt)

