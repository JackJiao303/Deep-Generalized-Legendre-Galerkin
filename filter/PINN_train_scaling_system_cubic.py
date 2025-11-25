import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import time
from filter_algorithms import hx, General_Legendre_poly
import sys

# This code aims to solving 1D Forward Kolmogorov equation by using PINN.
# Here we consider 1D parabolic PDE: u_t = 1/2 * u_xx - 1/2 * h(x)^2 * S^{-1} * u

# Neural network
class Net(nn.Module):
    def __init__(self, NN):
        super(Net, self).__init__()
        Input_num = 2  # input = (t,x)
        self.input_layer = nn.Linear(Input_num, NN)
        self.hidden_layer1 = nn.Linear(NN, int(NN / 2))
        self.hidden_layer2 = nn.Linear(int(NN / 2), int(NN / 2))
        self.hidden_layer3 = nn.Linear(int(NN / 2), int(NN / 2))
        self.hidden_layer4 = nn.Linear(int(NN / 2), int(NN / 2))
        self.output_layer = nn.Linear(int(NN / 2), 1)

    def forward(self, x):
        out = torch.tanh(self.input_layer(x))
        out = torch.tanh(self.hidden_layer1(out))
        out = torch.tanh(self.hidden_layer2(out))
        out = torch.tanh(self.hidden_layer3(out))
        out = torch.tanh(self.hidden_layer4(out))
        out_final = self.output_layer(out)
        return out_final

# PDE residual
def pde(x, net):
    # solution u in variable (t, x)
    u = net(x)
    weight = torch.ones_like(u)
    u_txy = torch.autograd.grad(outputs=u, inputs=x, grad_outputs=weight,
                                create_graph=True, allow_unused=True)[0]

    var_t = x[:, 0].unsqueeze(-1)
    var_x = x[:, 1].unsqueeze(-1)

    d_t = u_txy[:, 0].unsqueeze(-1)
    d_x = u_txy[:, 1].unsqueeze(-1)

    u_xx = torch.autograd.grad(d_x, x, grad_outputs=weight,
                               create_graph=True, allow_unused=True)[0][:, 1].unsqueeze(-1)

    # PDE parameters
    C = 1.5     # scaling factor
    S = 0.03    # covariance of observation noise
    h = hx(var_x)
    q = h ** 2 / S     # q = h^T * S^(-1) * h, here h(x) = x^3

    return d_t - 1/2 * u_xx/C**2 + 1/2 * q * (C**6) * u

def FKE_training(poly_index_range, initial_lr, total_epoch, x_scope, batch_size, Time_itv, L2cost, weights, Display_interval, if_save_basis):
    for poly_index in range(poly_index_range[0], poly_index_range[1]):

        print('poly_index: %d' % poly_index)
        start_time = time.time()

        # construct a FNN
        net = Net(40)  # Initial forward NN

        optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

        # records
        Loss_log = np.zeros((total_epoch))
        minLoss = 1.0  # minimal loss value

        print('Training process start ...')
        for epoch in range(total_epoch):
            optimizer.zero_grad()

            # *****************************
            # sampling of different region
            # *****************************
            # Initial boundary
            num_init1 = 700
            num_init2 = 100
            #x1 = 0.3
            x1 = 0.08        # [-1, 1] is segmented as [-1, -x1] u [-x1, x1] u [x1, 1]

            pt_t_init1 = torch.tensor(np.zeros((num_init1, 1)), dtype=torch.float32)
            pt_x_init1 = torch.tensor(np.linspace(-x_scope, -x1, num_init1)[:, np.newaxis], dtype=torch.float32)
            pt_t_init2 = torch.tensor(np.zeros((num_init2, 1)), dtype=torch.float32)
            pt_x_init2 = torch.tensor(np.linspace(-x1, x1, num_init2)[:, np.newaxis], dtype=torch.float32)
            pt_t_init3 = torch.tensor(np.zeros((num_init1, 1)), dtype=torch.float32)
            pt_x_init3 = torch.tensor(np.linspace(x1, x_scope, num_init1)[:, np.newaxis], dtype=torch.float32)

            # Generalized Legendre polynomial as initial distribution
            pt_u_init1 = General_Legendre_poly(pt_x_init1, poly_index)
            pt_u_init2 = General_Legendre_poly(pt_x_init2, poly_index)
            pt_u_init3 = General_Legendre_poly(pt_x_init3, poly_index)

            # *****************************
            # Inner region
            num_in = batch_size
            pt_t_in_var = torch.tensor(np.random.uniform(low=Time_itv[0], high=Time_itv[1], size=(num_in, 1)), dtype=torch.float32, requires_grad=True)
            pt_x_in_var = torch.tensor(np.random.uniform(low=-x_scope, high=x_scope, size=(num_in, 1)), dtype=torch.float32, requires_grad=True)
            pt_u_in_zeros = torch.tensor(np.zeros((num_in, 1)), dtype=torch.float32, requires_grad=False)

            # *****************************
            # Spatial boundary: x=-1 or x=1
            t1 = 0.03       # time interval segmented: [0, 1] = [0, t1] u [t1, Time_high]
            num_near = 800
            pt_t_near = torch.tensor(np.linspace(Time_itv[0], 0.03, num_near)[:, np.newaxis], dtype=torch.float32)
            pt_x_near_1 = torch.tensor(np.ones((num_near, 1)), dtype=torch.float32)
            pt_x_near_2 = torch.tensor((-1)*np.ones((num_near, 1)), dtype=torch.float32)

            num_far = 100
            pt_t_far = torch.tensor(np.linspace(0.03, Time_itv[1], num_far)[:, np.newaxis], dtype=torch.float32)
            pt_x_far_1 = torch.tensor(np.ones((num_far, 1)), dtype=torch.float32)
            pt_x_far_2 = torch.tensor((-1) * np.ones((num_far, 1)), dtype=torch.float32)

            pt_u_bdry_near = torch.tensor(np.zeros((num_near, 1)), dtype=torch.float32, requires_grad=False)
            pt_u_bdry_far = torch.tensor(np.zeros((num_far, 1)), dtype=torch.float32, requires_grad=False)

            # Calculate Loss Value
            # Initial loss
            net_init1 = net(torch.cat([pt_t_init1, pt_x_init1], 1))  # u(x,t) on initial condition
            mse_init1 = L2cost(net_init1, pt_u_init1)  # Loss of initial condition
            net_init2 = net(torch.cat([pt_t_init2, pt_x_init2], 1))  # u(x,t) on initial condition
            mse_init2 = L2cost(net_init2, pt_u_init2)  # Loss of initial condition
            net_init3 = net(torch.cat([pt_t_init3, pt_x_init3], 1))  # u(x,t) on initial condition
            mse_init3 = L2cost(net_init3, pt_u_init3)  # Loss of initial condition

            loss_init = mse_init1 + mse_init2 + mse_init3

            # boundary loss
            net_bdry_near_1 = net(torch.cat([pt_t_near, pt_x_near_1], 1))
            mes_bdry_near_1 = L2cost(net_bdry_near_1, pt_u_bdry_near)
            net_bdry_near_2 = net(torch.cat([pt_t_near, pt_x_near_2], 1))
            mes_bdry_near_2 = L2cost(net_bdry_near_2, pt_u_bdry_near)

            net_bdry_far_1 = net(torch.cat([pt_t_far, pt_x_far_1], 1))
            mes_bdry_far_1 = L2cost(net_bdry_far_1, pt_u_bdry_far)
            net_bdry_far_2 = net(torch.cat([pt_t_far, pt_x_far_2], 1))
            mes_bdry_far_2 = L2cost(net_bdry_far_2, pt_u_bdry_far)

            loss_bdry = mes_bdry_near_1 + mes_bdry_near_2 + mes_bdry_far_1 + mes_bdry_far_2

            # PDE loss
            f_out = pde(torch.cat([pt_t_in_var, pt_x_in_var], 1), net)
            loss_pde = L2cost(f_out, pt_u_in_zeros)

            loss = weights[0] * loss_init + weights[1] * loss_bdry + weights[2] * loss_pde

            loss.backward()
            optimizer.step()                # update network parameters
            Loss_log[epoch] = loss.data     # record loss value

            if loss.data.numpy() < minLoss:
                minLoss = loss.data.numpy()  # update minimal loss

            with torch.autograd.no_grad():
                if epoch % Display_interval == 0:
                    print('Epoch: ', epoch)
                    print("Loss_init: %.3e, Loss_bdry: %.3e, Loss_pde: %.3e, Loss_total: %.3e" %
                          (loss_init.data.numpy(), loss_bdry.data.numpy(), loss_pde.data.numpy(), loss.data.numpy()))
                    scheduler.step()

        # Save well-trained neural network parameters
        if if_save_basis:
            print('save neural basis.')
            torch.save(net.state_dict(), 'neural_basis/Trained_NN_param_L'+str(poly_index)+'.pt')

    print('Train NN finished.')
    print('Minimal Loss of Legendre basis order ' + str(poly_index) + ': %.3e' % minLoss)
    end_time = time.time()
    run_time = end_time - start_time
    print('Computational time: %.3f' % run_time)

    return Loss_log

def figure_plot(t_axis, y_axis):
    fig = plt.figure()
    plt.plot(t_axis, y_axis, linestyle='-', marker='.', color='y')
    plt.xlabel('Epochs')
    plt.ylabel('L2 error')
    plt.yscale('log')
    plt.show()


def main():
    #******************
    # global parameters
    #******************
    # Neural Network
    initial_lr = 1e-2
    L2cost = torch.nn.MSELoss(reduction='mean')  # Mean squared error
    total_epoch = 2000
    batch_size = 1000
    Display_interval = 1000  # Display interval of loss value
    weights = [2, 2, 1]  # Loss coefficients: initial, boundary, residual loss

    # Computational region
    x_scope = 1.0  # scope of sampling of state argument: [-scope, scope]
    Time_itv = [0, 0.3]

    poly_index_range = [5, 6]   # training basis
    if_save_basis = False
    # Training
    Loss_log = FKE_training(poly_index_range, initial_lr, total_epoch, x_scope, batch_size, Time_itv, L2cost, weights, Display_interval, if_save_basis)

    t_axis = np.arange(0, total_epoch)
    # Learning curve plotting
    figure_plot(t_axis, Loss_log)

    return 0


if __name__ == "__main__":
    sys.exit(main())

