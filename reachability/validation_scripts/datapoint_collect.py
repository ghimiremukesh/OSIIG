import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import modules as modules
import torch

current_dir = os.path.dirname(__file__)

def value_action(coords, model):
    # normalize the state for agent 1, agent 2
    model_in = {'coords': coords.unsqueeze(0)}
    model_output = model(model_in)
    y = model_output['model_out']

    return y


ckpt_path = os.path.join(current_dir, 'model(cr=0.05)_rev2/model_reach_avoid_sine.pth')  # r = 0.05 and Rx \neq Ry

activation = 'sine'

# Initialize and load the model
model = modules.SingleBVPNet(in_features=9, out_features=1, type=activation, mode='mlp',
                             final_layer_factor=1., hidden_features=512, num_hidden_layers=3)

checkpoint = torch.load(ckpt_path, map_location='cpu')
try:
    model_weights = checkpoint['model']
except:
    model_weights = checkpoint
model.load_state_dict(model_weights)
model.eval()

def coords_output():
    numpoints = 10000
    num_states = 8
    start_time = 0

    # coords = torch.zeros(numpoints, num_states).uniform_(0.04, 0.05)
    coords = torch.zeros(numpoints, num_states).uniform_(-1, 1)
    coords[:, 2:4] = 0
    coords[:, 6:8] = 0
    # coords[0, :] = torch.tensor([[-0.5, 0, 0, 0, 0.5, 0, 0, 0]])
    time = torch.ones(numpoints, 1) * start_time
    coords = torch.cat((time, coords), dim=1)

    y = value_action(coords, model)
    idx = torch.where(y <= 0)[1]
    coords_new = coords[idx, :]

    return coords_new

def get_distance(coords):
    x1 = coords[:, 1:3]
    x2 = coords[:, 5:7]

    distances = torch.linalg.norm(x1 - x2, dim=1)

    return distances

def check_feasibility(t, states):
    """

    :param t: time (float) (backwards)
    :param states: joint state
    :return: True if states are infeasible
    """
    t = t * torch.ones_like(states[:, 0]).reshape(-1, 1)
    coords = torch.cat((t, states), dim=1)

    y = value_action(coords, model)

    return y <= 0  # infeasible if y <= 0


