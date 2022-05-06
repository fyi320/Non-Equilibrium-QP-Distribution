# main

import matplotlib.pyplot as plt
import numpy as np
from NQP_func import *


########################### Parameter Definition and Initialization ##################################
Delta = 180    # ueV
Tc = 1.17   # Kelvin
Tb = 0.1 * Tc   # bath temperature
N0 = 1.74E4   # ueV^-1 um^-3
tau0 = 438 * 1E-9   # s
tau0_ph = 0.26 * 1E-9   # s
tau_l = tau0_ph
Ep = 16   # ueV
Pabs = 2E-15 * 6.24E24    # ueV s^-1 um^-3
N = 1000    # Total number of grid
alpha = np.zeros(2*N + 1)
chi = 0.9
Energy = np.zeros(N)
therm_f = np.zeros(N)
Omega = np.zeros(N)

for i in range(N):
    Energy[i] = (Delta + i) / Delta
    Omega[i] = (i) / Delta

for i in range(2*N + 1):
    if i < N:
        alpha[i] = Fermi(Delta+i, 2*Tb)
    elif i == N:
        alpha[i] = Pabs / Power(Ep, Delta, 1, N, alpha[:N], N0)
    else:
        alpha[i] = Bose(i-N, Tb)
        
        
########################### Main Loop ##################################
err = np.zeros(2*N + 1)
err_qp_phi = 1
err_phi_b = 1
it = 0
max_it = 20   # max iteration
while (err_qp_phi > 1e-5 or err_phi_b > 1e-5) and it < max_it:
    # Calculate alpha(l+1)
    J = np.zeros([2*N + 1,2*N + 1])
    for i in range(2*N + 1):
        for j in range(2*N + 1):
            J[i,j] = Jacobian(i, j, N, alpha, Ep, Delta, tau0, Tc, N0, tau0_ph, tau_l, Tb)
    J_inv = (np.matrix(J)).getI()
    alpha_m = np.matrix(alpha) - chi*(J_inv*np.matrix(err).getT()).getT()
    alpha = np.array(alpha_m).flatten()

    # Calculate err(l)
    for i in range(2*N + 1):
        if i < N:
            err[i] = time_deriv_f(Delta+i, Ep, Delta, alpha[N], N, tau0, Tc, alpha[:N], alpha[N+1:2*N+1])
        elif i == N:
            err[i] = Power(Ep, Delta, alpha[N], N, alpha[:N], N0) - Pabs
        else:
            err[i] = time_deriv_n(i-N, Delta, N, tau0_ph, tau_l, Tb, alpha[:N], alpha[N+1:2*N+1])
    err_qp_phi = abs((Pabs-Power_qp_phi(Ep, Delta, alpha[N], N, alpha[:N], N0, err))/Pabs)
    err_phi_b = abs((Pabs-Power_phi_b(alpha[N+1:2*N+1], Delta, N, Tc, N0, tau0, tau0_ph, tau_l, Tb))/Pabs)
    it += 1


########################### Calculate Thermal Equilibrium Distribution ##################################
Nnqp = calc_Nnqp(N, N0, alpha[:N], Delta)
for ss in range(1, 2000):
    T = ss / 100 * Tb
    Ntqp = calc_Ntqp(N, N0, T, Delta)
    if abs((Ntqp-Nnqp)/Nnqp) < 0.1:
        for i in range(N):
            therm_f[i] = Fermi(Delta+i, T)
        break
    
    
########################### Plot Result ##################################
fig, ax = plt.subplots(figsize=(16,10))
ax.plot(Energy, 1e3*alpha[:N], "b", label="neq")
ax.plot(Energy, 1e3*therm_f, "r", label="eq")
ax.set_xlabel('$E \, / \, \Delta$', fontsize=18)
ax.set_ylabel('$10^3 \\times f(E)$', fontsize=18)
plt.xlim([1, 2])
ax.set_title('');
fig.savefig('Fig3.jpg')

fig, ax = plt.subplots(figsize=(16,10))
ax.plot(Energy, alpha[:N], "b", label="neq")
ax.plot(Energy, therm_f, "r", label="eq")
ax.set_xlabel('$E \, / \, \Delta$', fontsize=18)
ax.set_ylabel('log(f)', fontsize=18)
ax.set_yscale('log')
plt.xlim([1, 4])
#plt.ylim([1e-10, 1e-8])
ax.set_title('');
fig.savefig('Fig3_inset1.jpg')

fig, ax = plt.subplots(figsize=(16,10))
ax.plot(Omega, Power_phi_b_comp(alpha[N+1:2*N+1], Delta, N, Tc, N0, tau0, tau0_ph, tau_l, Tb)/6.24E6, "k", label="neq")
ax.set_xlabel('$\Omega \, / \, \Delta$', fontsize=18)
ax.set_ylabel('$P(\Omega)_{\phi-b} \, (aW \, / \, \mu eV \, \mu m^3)$', fontsize=18)
ax.set_yscale('log')
plt.xlim([0, 4])
ax.set_title('');
fig.savefig('Fig3_inset2.jpg')


########################### Examine Convergence ##################################
print (abs(err[N]/Pabs), err_qp_phi, err_phi_b)