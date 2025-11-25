import torch
import numpy as np
import math

# this file contains system functions

# observation function
def hx(x):
    return 1/2 * (x ** 3)

# Legendre polynomial
def Legendre_poly(x, n):
    # Recursive definition of Legendre polynomial
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        return (2*n-1)/n * x * Legendre_poly(x, n-1) - (n-1)/n * Legendre_poly(x, n-2)

# Generalized Legendre polynomial
def General_Legendre_poly(x, n):
    # generalized legendre polynomial
    return 1/math.sqrt(4*n+6) * (Legendre_poly(x, n) - Legendre_poly(x, n+2))


def get_Phi(m, delta_x):
    x = np.arange(-1, 1, delta_x)
    # ***********************************************
    # Calculate Phi
    Phi = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            Phi[i, j] = np.dot(General_Legendre_poly(x, i), General_Legendre_poly(x, j)) * delta_x

    return Phi

def get_Phi_tilde(m, delta_x, S, C):
    x = np.arange(-1, 1, delta_x)
    x_piece_num = int(2 / delta_x)
    # ***********************************************
    # Calculate Phi_tilde
    L0_phi = np.zeros((x_piece_num, m))

    for i in range(m):
        # Calculate L0(phi_k)
        v = np.zeros((x_piece_num, 1))
        for j in range(1, x_piece_num - 1):
            phi = General_Legendre_poly(x[j], i)
            dd_phi = (General_Legendre_poly(x[j + 1], i) - 2 * General_Legendre_poly(x[j], i)
                      + General_Legendre_poly(x[j - 1], i)) / pow(delta_x, 2)

            v[j] = 1/2 * dd_phi / pow(C, 2) - 1/2 * pow(hx(x[j] * C), 2) / S  * phi
        v[0] = v[1]
        v[x_piece_num - 1] = v[x_piece_num - 2]

        L0_phi[:, i] = v[:, 0]

    Phi_tilde = np.zeros((m, m))
    for j in range(m):
        for k in range(m):
            Phi_tilde[j, k] = np.dot(General_Legendre_poly(x, j), L0_phi[:, k]) * delta_x

    return Phi_tilde


def next_solution_NBM(m, delta_x, Total_time, Phi, Phi_tilde, basis_value, u_init):
    # parameter:
    # u_init: initial distribution
    # delta_t: time step to solve
    # delta_x: position partition size
    # m: order of Basis function to be used
    # Total_time: total time to be solved
    # C: scaling factor

    x = np.arange(-1, 1, delta_x)
    x_piece_num = int(2/delta_x)

    # ***********************************************
    # decompose initial condition
    ksi = np.zeros((m, 1))
    for k in range(m):
        ksi[k] = np.dot(u_init, General_Legendre_poly(x, k)) * delta_x

    alpha0 = np.linalg.inv(Phi) @ ksi       # initial weight

    # Forward Evolution

    u_final = basis_value @ alpha0
    return u_final


def basis_function_value(m, delta_x, Total_time, net_list):
    x = np.arange(-1, 1, delta_x)
    x_piece_num = int(2 / delta_x)
    basis_function_value = np.zeros((x_piece_num, m))
    for poly_order in range(m):
        net = net_list[poly_order]
        for i in range(x_piece_num):
            point = torch.tensor([Total_time, x[i]], dtype=torch.float32)
            basis_function_value[i, poly_order] = net(point)

    return basis_function_value
