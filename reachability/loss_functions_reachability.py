import torch
import diff_operators_reachability as diff_operators
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_hji_8D(dataset, minWith):
    def hji_8D(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']

        jac, _ = diff_operators.jacobian(y, x)
        dvdt = jac[..., 0, 0].squeeze()
        dvdx = jac[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11 = dvdx[:, :1]  # lambda_11
        lam12 = dvdx[:, 1:2]  # lambda_12
        lam13 = dvdx[:, 2:3] / ((6 + 6) / 2)  # lambda_13
        lam14 = dvdx[:, 3:4] / ((6 + 6) / 2)  # lambda_14
        # lam13 = dvdx[:, 2:3] / ((15 + 15) / 2)  # lambda_13
        # lam14 = dvdx[:, 3:4] / ((25 + 25) / 2)  # lambda_14

        # unnormalize the costate for agent 2
        lam21 = dvdx[:, 4:5]  # lambda_21
        lam22 = dvdx[:, 5:6]  # lambda_22
        lam23 = dvdx[:, 6:7] / ((4 + 4) / 2)  # lambda_23
        lam24 = dvdx[:, 7:8] / ((4 + 4) / 2)  # lambda_24
        # lam23 = dvdx[:, 6:7] / ((1.3 + 1.3) / 2)  # lambda_23
        # lam24 = dvdx[:, 7:8] / ((1.5 + 1.5) / 2)  # lambda_24

        # set up bounds for u1 and u2
        max_acc_u = torch.tensor([6.], dtype=torch.float32).to(device)
        min_acc_u = torch.tensor([-6.], dtype=torch.float32).to(device)
        max_acc_d = torch.tensor([4.], dtype=torch.float32).to(device)
        min_acc_d = torch.tensor([-4.], dtype=torch.float32).to(device)
        RA = 1e-4
        RD = 1e-1

        # Agent 1's action
        ux = 1 * lam13
        ux[torch.where(ux > 0)] = 1
        ux[torch.where(ux < 0)] = -1
        ux[torch.where(ux == 1)] = min_acc_u
        ux[torch.where(ux == -1)] = max_acc_u
        # ux = -0.5 * lam13 / RA

        uy = 1 * lam14
        uy[torch.where(uy > 0)] = 1
        uy[torch.where(uy < 0)] = -1
        uy[torch.where(uy == 1)] = min_acc_u
        uy[torch.where(uy == -1)] = max_acc_u
        # uy = -0.5 * lam14 / RA

        # Agent 2's action
        dx = 1 * lam23
        dx[torch.where(dx > 0)] = 1
        dx[torch.where(dx < 0)] = -1
        dx[torch.where(dx == 1)] = max_acc_d
        dx[torch.where(dx == -1)] = min_acc_d
        # dx = 0.5 * lam23 / RD

        dy = 1 * lam24
        dy[torch.where(dy > 0)] = 1
        dy[torch.where(dy < 0)] = -1
        dy[torch.where(dy == 1)] = max_acc_d
        dy[torch.where(dy == -1)] = min_acc_d
        # dy = 0.5 * lam24 / RD

        # unnormalize the state for agent 1
        sx1 = model_output['model_in'][:, :, 1:2]
        sy1 = model_output['model_in'][:, :, 2:3]
        vx1 = (model_output['model_in'][:, :, 3:4] + 1) * (6 + 6) / 2 - 6
        vy1 = (model_output['model_in'][:, :, 4:5] + 1) * (6 + 6) / 2 - 6
        # vx1 = (model_output['model_in'][:, :, 3:4] + 1) * (15 + 15) / 2 - 15
        # vy1 = (model_output['model_in'][:, :, 4:5] + 1) * (25 + 25) / 2 - 25

        # unnormalize the state for agent 2
        sx2 = model_output['model_in'][:, :, 5:6]
        sy2 = model_output['model_in'][:, :, 6:7]
        vx2 = (model_output['model_in'][:, :, 7:8] + 1) * (4 + 4) / 2 - 4
        vy2 = (model_output['model_in'][:, :, 8:9] + 1) * (4 + 4) / 2 - 4
        # vx2 = (model_output['model_in'][:, :, 7:8] + 1) * (1.3 + 1.3) / 2 - 1.3
        # vy2 = (model_output['model_in'][:, :, 8:9] + 1) * (1.5 + 1.5) / 2 - 1.5

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham = lam11.squeeze() * vx1.squeeze() + lam12.squeeze() * vy1.squeeze() + lam13.squeeze() * ux.squeeze() \
              + lam14.squeeze() * uy.squeeze() + lam21.squeeze() * vx2.squeeze() + lam22.squeeze() * vy2.squeeze() + \
              lam23.squeeze() * dx.squeeze() + lam24.squeeze() * dy.squeeze()

        # ham = lam11.squeeze() * vx1.squeeze() + lam12.squeeze() * vy1.squeeze() + lam13.squeeze() * ux.squeeze() \
        #       + lam14.squeeze() * uy.squeeze() + lam21.squeeze() * vx2.squeeze() + lam22.squeeze() * vy2.squeeze() + \
        #       lam23.squeeze() * dx.squeeze() + lam24.squeeze() * dy.squeeze() + RA*(ux.squeeze()**2+uy.squeeze()**2) \
        #       - RD*(dx.squeeze()**2+dy.squeeze()**2)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
            weight2 = 1
        else:
            # diff_constraint_hom = torch.min(source_boundary_values.squeeze() - y.squeeze(), -dvdt + ham)
            diff_constraint_hom = torch.max(y.squeeze() - source_boundary_values.squeeze(), dvdt - ham)
            weight2 = torch.abs(dirichlet).sum() / torch.abs(diff_constraint_hom).sum()

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum(),
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()/weight2}

    return hji_8D
