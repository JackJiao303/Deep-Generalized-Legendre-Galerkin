import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import math
import sys
from filter_algorithms import get_Phi, get_Phi_tilde, next_solution_NBM, basis_function_value, hx

# Neuron Network architecture
class Net(nn.Module):
    def __init__(self, NN):
        super(Net, self).__init__()
        Input_num = 2
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

mse_cost_function = torch.nn.MSELoss(reduction='mean')

def nonlinear_filter(Time, dt, mu0, sigma0, Obv_itv, delta_t, delta_x, C, S, m):
    eff_num = 0
    while eff_num == 0:
        # System evolution
        T_piece_num = int(Time/dt)
        t_axis = np.arange(0, Time, dt)

        Xt = np.zeros((T_piece_num, 1))
        Zt = np.zeros((T_piece_num, 1))

        # Monte-Carlo process
        Xt[0] = np.random.normal(mu0, sigma0, 1)
        for i in range(T_piece_num-1):
            Xt[i+1] = Xt[i] + np.random.normal(0, math.sqrt(dt), 1)
            Zt[i+1] = Zt[i] + hx(Xt[i]) * dt + np.random.normal(0, math.sqrt(S * dt), 1)

        if np.max(np.abs(Xt)) > C:
            continue
        else: eff_num += 1
        #***********************************************
        # observation data collected
        obs_num = int(Time/Obv_itv)
        obs_itv = int(Obv_itv/dt)       # picking an observation each # real value
        t_piece_num = int(Obv_itv/delta_t)

        # observation history
        Y_t = Zt[::obs_itv]      # observation history
        t_obs = t_axis[::obs_itv]       # observation instants

        #***********************************************
        # dglg implement

        delta_c = delta_x * C

        x = np.arange(-1, 1, delta_x)
        x_C = np.arange(-C, C, delta_c)

        x_piece_num = int(2/delta_x)

        u_dglg = np.zeros((x_piece_num, obs_num))        # each column domain on the bigger interval [-C, C]
        u_dglg[:, 0] = norm.pdf(x_C, mu0, sigma0)

        Phi = get_Phi(m, delta_x)
        Phi_tilde = get_Phi_tilde(m, delta_x, S, C)

        # Load trained neural basis
        net_list = []
        for Poly_order in range(m):
            param_net = torch.load('neural_basis/Trained_NN_param_L'+str(Poly_order)+'.pt')
            net = Net(40)
            net.load_state_dict(param_net)
            net_list.append(net)

        basis_value = basis_function_value(m, delta_x, Obv_itv, net_list)

        # implement filter
        print('start filtering...')
        start_time = time.time()

        for index in range(obs_num-1):
            u_init = u_dglg[:, index]
            u_final = next_solution_NBM(m, delta_x, Obv_itv, Phi, Phi_tilde, basis_value, u_init)
            delta_Y = Y_t[index + 1] - Y_t[index]
            h_tilde = hx(x * C)
            u_dglg[:, index + 1] = u_final[:, 0] * np.exp(h_tilde / S * delta_Y)

        end_time = time.time()
        run_time_dglg = end_time - start_time
        print('Finished. Cost time: %.2f' % run_time_dglg)

        est_dglg = np.zeros((obs_num, 1))
        for i in range(obs_num):
            est_dglg[i] = np.dot(u_dglg[:, i], x_C) / np.sum(u_dglg[:, i])

        return t_axis, t_obs, Xt, est_dglg

def figure_plot(t_axis, t_obs, Xt, est_dglg, Time):
    fig = plt.figure()
    plt.rcParams.update({'font.size': 14})
    plt.plot(t_axis, Xt, label='Real', color='blue')
    plt.plot(t_obs, est_dglg, label='DGLG', color='purple')
    plt.legend(fontsize='small')
    plt.xlabel('t')
    plt.ylabel('Xt')
    plt.title('State tracking')
    plt.xlim(0, Time)
    plt.show()

def main():
    # system Global parameters
    Time = 3  # simulation time
    dt = 0.001  # simulated time step
    mu0 = 0.1  # initial condition
    sigma0 = 0.05  # initial standard deviation
    C = 1.5  # scaling factor
    S = 0.03  # covariance of observation noise

    #  observation parameters
    Obv_itv = 0.01  # default 0.01
    delta_t = 0.001  # step size of Legendre Spectral Method

    # dglg parameters
    m = 7  # polynomial basis order
    delta_x = 0.02  # spatial grid

    t_axis, t_obs, Xt, est_dglg = nonlinear_filter(Time, dt, mu0, sigma0, Obv_itv, delta_t, delta_x, C, S, m)
    figure_plot(t_axis, t_obs, Xt, est_dglg, Time)

    return 0

if __name__ == "__main__":
    sys.exit(main())

