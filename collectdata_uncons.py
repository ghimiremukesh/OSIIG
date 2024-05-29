import multiprocessing
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import icnn_pytorch as icnn
import configargparse
import utils
from tqdm import tqdm
import scipy.io as scio
from utils import convex_hull
from odeintw import odeintw


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
            # cvx_vals[i] = slope * p[i] + intercept
            cvx_vals[i, k] = slope * p[k] + intercept
    # p_idx = [list(ps).index(each) for each in p]
    return cvx_vals


def get_optimal_u(x, p, R, K, Phi, t, total_steps):
    index = int((1 - t) * total_steps)
    try:
        ztheta = torch.tensor([0, 1, 0, 0]) * (2 * p - 1).to(torch.float32)
    except:
        ztheta = np.array([0, 1, 0, 0]) * (2 * p - 1)

    u = utils.get_analytical_u(K[index, :, :], R, Phi[index, :, :],
                                            x[..., :4].T, ztheta.T)

    return u


if __name__ == '__main__':
    num_points = 10000
    num_players = 2
    num_states = 4  # x, y, vx, vy for each player


    game = 'uncons'

    g1 = utils.GOAL_1
    g2 = utils.GOAL_2

    t = opt.time
    dt = 0.1

    total_steps = int(1/dt)

    ## hexner's ground truth
    # define system
    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])

    B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

    Q = np.zeros((4, 4))

    R1 = np.array([[0.05, 0], [0, 0.025]])

    PT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    tspan = np.linspace(0, 1, total_steps + 1)
    tspan = np.flip(tspan)
    K1 = odeintw(utils.dPdt, PT, tspan, args=(A, B, Q, R1, None,))

    K1 = np.flip(K1, axis=0)

    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    t_span = np.linspace(0, 1, total_steps+1)
    t_span = np.flip(t_span)
    PhiT = np.eye(4)

    Phi_sol = odeintw(utils.dPhi, PhiT, t_span, args=(A,))
    Phi_sol = np.flip(Phi_sol, axis=0)

    z = np.array([[0], [1], [0], [0]])
    d1 = utils.d(Phi_sol, K1, B, R1, z)
    B2 = B

    R2 = np.array([[0.05, 0], [0, 0.1]])
    K2 = odeintw(utils.dPdt, PT, tspan, args=(A, B2, Q, R2, None,))
    K2 = np.flip(K2, axis=0)
    d2 = utils.d(Phi_sol, K2, B2, R2, z)


    G = [g1, g2]


    ts = np.around(np.arange(dt, 1 + dt, dt), 2)
    t_step = int(np.where(ts == t)[0] + 1)

    NUM_PS = 100

    logging_root = 'logs/'
    save_root = f'uncons/'

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    torch.manual_seed(0)


    v1x_max = 6
    v1y_max = 12

    v2x_max = 6
    v2y_max = 4

    epsilon_x = 0.1
    epsilon_y1 = 0.7

    positions = torch.zeros(num_points, 4).uniform_(-1, 1)
    vel_1_x = torch.zeros(num_points, 1).uniform_(-(v1x_max - epsilon_x), (v1x_max - epsilon_x)).reshape(-1, 1)
    vel_1_y = torch.zeros(num_points, 1).uniform_(-(v1y_max - epsilon_y1), (v1y_max - epsilon_y1)).reshape(-1, 1)
    vel_1 = torch.cat((vel_1_x, vel_1_y), dim=1)

    vel_2_x = torch.zeros(num_points, 1).uniform_(-(v2x_max - epsilon_x), (v2x_max - epsilon_x)).reshape(-1, 1)
    vel_2_y = torch.zeros(num_points, 1).uniform_(-(v2y_max - epsilon_x), (v2y_max - epsilon_x)).reshape(-1, 1)
    vel_2 = torch.cat((vel_2_x, vel_2_y), dim=1)

    xy = torch.zeros(num_points, 8)
    xy[:, :2] = positions[:, :2]
    xy[:, 2:4] = vel_1
    xy[:, 4:6] = positions[:, 2:4]
    xy[:, 6:8] = vel_2


    time = torch.ones(xy.shape[0], 1) * t



    if t == dt:
        t_next = t - dt
        vs = []
        ps = np.linspace(0, 1, NUM_PS)

        # temp = []
        for p_each in tqdm(ps):
            p_next = p_each * torch.ones_like(xy[:, 0]).reshape(-1, 1)
            x1 = xy
            x2 = torch.cat((xy[:, 4:8], xy[:, :4]), dim=1)
            u = get_optimal_u(x1.numpy(), p_next.numpy(), R1, K1, Phi_sol, t, total_steps)
            d = get_optimal_u(x2.numpy(), p_next.numpy(), R2, K2, Phi_sol, t, total_steps)
            x_prime = utils.go_forward(xy, u, d, dt=dt, a=0.5)
            v_next = utils.final_cost(x_prime[:, :2], x_prime[:, 4:6], G, p_next.numpy(), game=game).reshape(-1, ) + \
                     dt * (np.sum(np.multiply(np.diag(R1), u ** 2), axis=-1) - np.sum(np.multiply(np.diag(R2), d ** 2), axis=-1)).reshape(-1, )
            #
            vs.append(v_next.reshape(-1, ))


        true_v = cav_vex(vs, type='vex', num_ps=NUM_PS).reshape(1, -1, 1)



        ps = torch.linspace(0, 1, 100)
        p = ps.repeat([len(xy), 1]).reshape(-1, 1)
        x = torch.vstack([xy[i].repeat([NUM_PS, 1]) for i in range(len(xy))])
        coords = torch.cat((x, p), dim=1)


        x_prev = coords

        x_prev = utils.normalize_to_max(x_prev, v1x_max, v1y_max, v2x_max, v2y_max).detach().cpu().numpy()

        gt = {'states': np.vstack(x_prev),
              'values': np.vstack(true_v)}

        scio.savemat(os.path.join(save_root, f'train_data_t_{t:.2f}.mat'), gt)

    else:
        t_next = t - dt

        load_dir = os.path.join(logging_root, f'uncons/t_{t_step - 1}/')

        hf = 256

        val_model = icnn.SingleBVPNet(in_features=9, out_features=1, type=activation, mode='mlp',
                                      hidden_features=hf, num_hidden_layers=5, dropout=0)
        val_model.to(device)
        model_path = os.path.join(load_dir, 'checkpoints_dir', 'model_final.pth')
        checkpoint = torch.load(model_path, map_location=device)
        try:
            val_model.load_state_dict(checkpoint['model'])
        except:
            val_model.load_state_dict(checkpoint)
        val_model.eval()


        vs = []
        ps = np.linspace(0, 1, NUM_PS)


        for p_each in tqdm(ps):
            p_next = p_each * torch.ones_like(xy[:, 0]).reshape(-1, 1)
            x1 = xy
            x2 = torch.cat((xy[:, 4:8], xy[:, :4]), dim=1)
            u = get_optimal_u(x1.numpy(), p_next.numpy(), R1, K1, Phi_sol, t, total_steps)
            d = get_optimal_u(x2.numpy(), p_next.numpy(), R2, K2, Phi_sol, t, total_steps)
            x_prime = torch.from_numpy(utils.go_forward(xy, u, d, dt=dt, a=0.5))
            coords = torch.cat((x_prime, p_next), dim=1)
            coords = utils.normalize_to_max(coords, v1x_max, v1y_max, v2x_max, v2y_max)  # normalize the velocities
            coords = {'coords': coords.to(torch.float32)}
            v_next = val_model(coords)['model_out'].detach().numpy()
            v_next = v_next.reshape(-1, ) + \
                     dt * (np.sum(np.multiply(np.diag(R1), u ** 2), axis=-1) - np.sum(np.multiply(np.diag(R2), d ** 2), axis=-1)).reshape(-1, )

            vs.append(v_next.reshape(-1, ))

        true_v = utils.cav_vex(vs, type='vex', num_ps=NUM_PS).reshape(1, -1, 1)

        ps = torch.linspace(0, 1, 100)
        p = ps.repeat([len(xy), 1]).reshape(-1, 1)
        x = torch.vstack([xy[i].repeat([NUM_PS, 1]) for i in range(len(xy))])
        coords = torch.cat((x, p), dim=1)

        x_prev = coords
        x_prev = utils.normalize_to_max(x_prev, v1x_max, v1y_max, v2x_max, v2y_max).detach().cpu().numpy()

        gt = {'states': np.vstack(x_prev),
              'values': np.vstack(true_v)}

        scio.savemat(os.path.join(save_root, f'train_data_t_{t:.2f}.mat'), gt)
