import os
import scipy.io as scio
import torch
import numpy as np

from torch.utils.data import Dataset


class OneSidedGame(Dataset):
    def __init__(self, numpoints, t, dt=0.5, u_max=1, d_max=0.8, num_ps=11, num_src_samples=1, seed=0):
        super().__init__()
        torch.manual_seed(seed)

        self.numpoints = numpoints
        self.time = t
        self.dt = dt
        self.N_src_samples = num_src_samples
        self.umax = u_max
        self.dmax = d_max
        self.num_ps = num_ps

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = torch.zeros(self.numpoints, 2).uniform_(-1, 1)  # dx and dv
        p = torch.zeros(self.numpoints, 1).uniform_(0, 1)  # belief
        # p = torch.ones(self.numpoints, 1) * 0.5  # for debugging
        # x = torch.ones(self.numpoints, 2) * 0.01  # for debugging
        coords = torch.cat((x, p), dim=1)
        time = torch.ones(self.numpoints, 1) * self.time

        coords = torch.cat((time, coords), dim=1)

        # make sure we have training samples at final time (backward)
        # coords[-self.N_src_samples:, 0] = 0 # train at the specific time

        # -p * max(0, x) - (1 - p) * max(0, -x)
        boundary_values = -coords[:, -1] * torch.maximum(torch.zeros_like(coords[:, 1]), coords[:, 1]) - (
                1 - coords[:, -1]) * \
                          torch.maximum(torch.zeros_like(coords[:, 1]), -coords[:, 1])

        if self.time == 0:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce boundary condition at final time
            dirichlet_mask = (coords[:, 0, None]) == 0

        return {'coords': coords}, {'source_boundary_values': boundary_values.reshape(-1, 1),
                                    'dirichlet_mask': dirichlet_mask}


class OneSidedGame_4d(Dataset):
    def __init__(self, numpoints, t, dt=0.5, u_max=1, d_max=0.8, num_ps=11, num_src_samples=1, seed=0):
        super().__init__()
        torch.manual_seed(seed)

        self.numpoints = numpoints
        self.time = t
        self.dt = dt
        self.N_src_samples = num_src_samples
        self.umax = u_max
        self.dmax = d_max
        self.num_ps = num_ps

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = torch.zeros(self.numpoints, 4).uniform_(-1, 1)  # x1 v1 x2 v2
        p = torch.zeros(self.numpoints, 1).uniform_(0, 1)  # belief
        # p = torch.ones(self.numpoints, 1) * 0.5  # for debugging
        # x = torch.ones(self.numpoints, 2) * 0.01  # for debugging
        coords = torch.cat((x, p), dim=1)
        time = torch.ones(self.numpoints, 1) * self.time

        coords = torch.cat((time, coords), dim=1)

        # make sure we have training samples at final time (backward)
        # coords[-self.N_src_samples:, 0] = 0 # train at the specific time

        # -p * max(0, x) - (1 - p) * max(0, -x)
        boundary_values = -coords[:, -1] * torch.maximum(torch.zeros_like(coords[:, 1]), (coords[:, 1] - coords[:, 3])) \
                          - (1 - coords[:, -1]) * torch.maximum(torch.zeros_like(coords[:, 1]),
                                                                -(coords[:, 1] - coords[:, 3]))

        if self.time == 0:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce boundary condition at final time
            dirichlet_mask = (coords[:, 0, None]) == 0

        return {'coords': coords}, {'source_boundary_values': boundary_values.reshape(-1, 1),
                                    'dirichlet_mask': dirichlet_mask}


class TrainingDataset(Dataset):
    def __init__(self, matfile):
        self.data = scio.loadmat(matfile)
        self.coords = self.data['X_train']
        self.values = self.data['V_train']
        # self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx, :]
        value = self.values[idx, :]

        return torch.tensor(coord, dtype=torch.float32), torch.tensor(value, dtype=torch.float32)


class TrainInterTime(Dataset):
    def __init__(self, matfile):
        self.data = scio.loadmat(matfile)
        self.coords = self.data['states']
        self.values = self.data['values']

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx, :]
        value = self.values[idx, :]

        return {'coords': torch.tensor(coord, dtype=torch.float32)}, \
               {'values': torch.tensor(value, dtype=torch.float32)}
        # return (torch.tensor(coord, dtype=torch.float32), torch.tensor(value, dtype=torch.float32))

class TrainInterTime_Dual(Dataset):
    def __init__(self, matfile):
        self.data = scio.loadmat(matfile)
        self.coords = self.data['states']
        self.values = self.data['values']

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx, :]
        value = self.values[idx, :]

        return {'coords': torch.tensor(coord, dtype=torch.float32).reshape(1, -1, 1)}, \
               {'values': torch.tensor(value, dtype=torch.float32).reshape(1, -1, 1)}


class TrainInterTimeGrad(Dataset):
    def __init__(self, matfile):
        self.data = scio.loadmat(matfile)
        self.coords = self.data['states']
        self.values = self.data['values']
        self.grads = self.data['gradient']

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx, :]
        value = self.values[idx, :]
        grad = self.grads[idx, :]

        return {'coords': torch.tensor(coord, dtype=torch.float32)}, \
               {'values': torch.tensor(value, dtype=torch.float32), 'gradient': torch.tensor(grad, dtype=torch.float32)}
