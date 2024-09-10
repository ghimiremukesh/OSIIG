import torch
from torch.utils.data import Dataset


class Reachability8D(Dataset):
    def __init__(self, numpoints, collisionR=0.05, pretrain=False, tMin=0.0, tMax=0.5, counter_start=0,
                 counter_end=100e3, pretrain_iters=2000, angle_alpha=1.0, num_src_samples=1000, seed=0):
        super().__init__()
        torch.manual_seed(0)


        self.pretrain = pretrain
        self.numpoints = numpoints


        self.collisionR = collisionR


        self.num_states = 8


        self.tMax = tMax
        self.tMin = tMin


        self.N_src_samples = num_src_samples


        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end
        self.counter_checkpoint = 30000
        self.counter_data = 3000
        self.num_add = 0
        self.num_vio = 8000
        self.num_end = 2000
        self.counter_next = 0


        # Set the seed
        torch.manual_seed(seed)


    def __len__(self):
        return 1


    # def __getitem__(self, idx):
    #     start_time = 0.  # time to apply  initial conditions
    #
    #     # uniformly sample domain and include coordinates where source is non-zero
    #     # coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)
    #     # coords[:10000, 4:5] = coords[:10000, 0:1] + torch.zeros(10000, 1).uniform_(0, 0.035)
    #     # coords[:10000, 5:6] = coords[:10000, 1:2] + torch.zeros(10000, 1).uniform_(0, 0.035)
    #     # coords[-10000:, 4:5] = coords[-10000:, 0:1] + torch.zeros(10000, 1).uniform_(0, 0.035)
    #     # coords[-10000:, 5:6] = coords[-10000:, 1:2] + torch.zeros(10000, 1).uniform_(0, 0.035)
    #
    #     if self.pretrain:
    #         # uniformly sample domain and include coordinates for both agents
    #         self.n_sample = self.numpoints - self.num_vio - self.num_end
    #         coords1 = torch.zeros(self.num_vio, self.num_states).uniform_(-1, 1)
    #         coords1[:, 4:5] = coords1[:, 0:1] + torch.zeros(self.num_vio, 1).uniform_(0, 0.035)
    #         coords1[:, 5:6] = coords1[:, 1:2] + torch.zeros(self.num_vio, 1).uniform_(0, 0.035)
    #         coords2 = torch.zeros(self.n_sample, self.num_states).uniform_(-1, 1)
    #         coords3 = torch.zeros(self.num_end, self.num_states).uniform_(-1, 1)
    #         coords3[:, 4:5] = coords3[:, 0:1] + torch.zeros(self.num_end, 1).uniform_(0, 0.035)
    #         coords3[:, 5:6] = coords3[:, 1:2] + torch.zeros(self.num_end, 1).uniform_(0, 0.035)
    #
    #         coords = torch.cat((coords1, coords2, coords3), dim=0)
    #
    #     else:
    #         if not self.counter % self.counter_checkpoint and (self.counter + 1):
    #             self.numpoints = self.numpoints + 60000
    #             self.num_vio = self.num_vio + 10000
    #             print(self.numpoints)
    #
    #         self.n_sample = self.numpoints - self.num_vio - self.num_end
    #
    #         if not self.counter % self.counter_data and (self.counter + 1):
    #             self.coords1 = torch.zeros(self.num_vio, self.num_states).uniform_(-1, 1)
    #             self.coords1[:, 4:5] = self.coords1[:, 0:1] + torch.zeros(self.num_vio, 1).uniform_(0, 0.035)
    #             self.coords1[:, 5:6] = self.coords1[:, 1:2] + torch.zeros(self.num_vio, 1).uniform_(0, 0.035)
    #             self.coords2 = torch.zeros(self.n_sample, self.num_states).uniform_(-1, 1)
    #             self.coords3 = torch.zeros(self.num_end, self.num_states).uniform_(-1, 1)
    #             self.coords3[:, 4:5] = self.coords3[:, 0:1] + torch.zeros(self.num_end, 1).uniform_(0, 0.035)
    #             self.coords3[:, 5:6] = self.coords3[:, 1:2] + torch.zeros(self.num_end, 1).uniform_(0, 0.035)
    #
    #         coords = torch.cat((self.coords1, self.coords2, self.coords3), dim=0)
    #
    #     if self.pretrain:
    #         # only sample in time around the initial condition
    #         time = torch.ones(self.numpoints, 1) * start_time
    #         coords = torch.cat((time, coords), dim=1)
    #
    #     else:
    #         # slowly grow time values from start time
    #         # this currently assumes start_time = 0 and max time value is tMax
    #         if not self.counter % self.counter_data and (self.counter + 1):
    #             if not self.counter % self.counter_checkpoint and (self.counter + 1):
    #                 self.counter_next = self.counter + self.counter_checkpoint
    #                 print(self.counter_next)
    #             else:
    #                 pass
    #             self.time_horizon = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax - self.tMin) * (
    #                     self.counter_next / self.full_count))
    #
    #         time = self.time_horizon
    #
    #         coords = torch.cat((time, coords), dim=1)
    #
    #         # make sure we always have training samples at the initial time
    #         coords[-self.N_src_samples:, 0] = start_time
    #
    #     # set up the initial value function
    #     # boundary_values = self.collisionR - torch.sqrt((coords[:, 1:2]-coords[:, 5:6])**2+(coords[:, 2:3]-coords[:, 6:7])**2)
    #     boundary_values = torch.sqrt((coords[:, 1:2] - coords[:, 5:6])**2 + (coords[:, 2:3] - coords[:, 6:7])**2) - self.collisionR
    #
    #     if self.pretrain:
    #         dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
    #     else:
    #         # only enforce initial conditions around start_time
    #         dirichlet_mask = (coords[:, 0, None] == start_time)
    #
    #     if self.pretrain:
    #         self.pretrain_counter += 1
    #     elif self.counter < self.full_count:
    #         self.counter += 1
    #
    #     if self.pretrain and self.pretrain_counter == self.pretrain_iters:
    #         self.pretrain = False
    #
    #     return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}


    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions


        # uniformly sample domain and include coordinates where source is non-zero
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)
        coords[:10000, 4:5] = coords[:10000, 0:1] + torch.zeros(10000, 1).uniform_(0, 0.035)  # cr=0.05
        coords[:10000, 5:6] = coords[:10000, 1:2] + torch.zeros(10000, 1).uniform_(0, 0.035)
        coords[-1000:, 4:5] = coords[-1000:, 0:1] + torch.zeros(1000, 1).uniform_(0, 0.035)
        coords[-1000:, 5:6] = coords[-1000:, 1:2] + torch.zeros(1000, 1).uniform_(0, 0.035)


        # old RA/RD setting
        # coords[:8000, 4:5] = coords[:8000, 0:1] + torch.zeros(8000, 1).uniform_(0, 0.035)  # cr=0.05
        # coords[:8000, 5:6] = coords[:8000, 1:2] + torch.zeros(8000, 1).uniform_(0, 0.035)
        # coords[-1000:, 4:5] = coords[-1000:, 0:1] + torch.zeros(1000, 1).uniform_(0, 0.035)
        # coords[-1000:, 5:6] = coords[-1000:, 1:2] + torch.zeros(1000, 1).uniform_(0, 0.035)


        # coords[:6000, 4:5] = coords[:6000, 0:1] + torch.zeros(6000, 1).uniform_(0, 0.1)  # cr=0.15
        # coords[:6000, 5:6] = coords[:6000, 1:2] + torch.zeros(6000, 1).uniform_(0, 0.1)
        # coords[-1000:, 4:5] = coords[-1000:, 0:1] + torch.zeros(1000, 1).uniform_(0, 0.1)
        # coords[-1000:, 5:6] = coords[-1000:, 1:2] + torch.zeros(1000, 1).uniform_(0, 0.1)


        # coords[:6000, 4:5] = coords[:6000, 0:1] + torch.zeros(6000, 1).uniform_(0, 0.17)  # cr=0.25
        # coords[:6000, 5:6] = coords[:6000, 1:2] + torch.zeros(6000, 1).uniform_(0, 0.17)
        # coords[-1000:, 4:5] = coords[-1000:, 0:1] + torch.zeros(1000, 1).uniform_(0, 0.17)
        # coords[-1000:, 5:6] = coords[-1000:, 1:2] + torch.zeros(1000, 1).uniform_(0, 0.17)


        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)


        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax - self.tMin) * (
                    self.counter / self.full_count))


            coords = torch.cat((time, coords), dim=1)


            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time


        # set up the initial value function
        # boundary_values = self.collisionR - torch.sqrt((coords[:, 1:2]-coords[:, 5:6])**2+(coords[:, 2:3]-coords[:, 6:7])**2)
        boundary_values = torch.sqrt((coords[:, 1:2] - coords[:, 5:6])**2 + (coords[:, 2:3] - coords[:, 6:7])**2) - self.collisionR


        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)


        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1


        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False


        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}
