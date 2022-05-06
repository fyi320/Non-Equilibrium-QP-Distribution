# function definition (import as .py)

import numpy as np

########################### Function Declaration ##################################
def Fermi(E, T):
    return 1 / (np.exp(E*1e-6/(8.617333262145e-5*T)) + 1)

def Bose(E, T):
    return 1 / (np.exp(E*1e-6/(8.617333262145e-5*T)) - 1)

def rho(E, Delta):
    if E <= Delta:
        return 0 
    return E / np.sqrt(E**2 - Delta**2)

def I_qp(E, Ep, Delta, B, N, f):
    if E+Ep-Delta >= N:
        return 2*B*(-rho(E-Ep, Delta)*(1+Delta**2/(E*(E-Ep)))*(f[E-Delta]-f[E-Ep-Delta]))
    elif E-Ep-Delta < 0:
        return 2*B*(rho(E+Ep, Delta)*(1+Delta**2/(E*(E+Ep)))*(f[E+Ep-Delta]-f[E-Delta]))
    else:
        return 2*B*(rho(E+Ep, Delta)*(1+Delta**2/(E*(E+Ep)))*(f[E+Ep-Delta]-f[E-Delta])-
                  rho(E-Ep, Delta)*(1+Delta**2/(E*(E-Ep)))*(f[E-Delta]-f[E-Ep-Delta]))

def time_deriv_n(Omega, Delta, N, tau0_ph, tau_l, Tb, f, n):
    result = 0
    for i in range(N):
        E = Delta + i
        if i+Omega < N:
            result += (-2/np.pi/tau0_ph/Delta)*rho(E, Delta)*rho(E+Omega, Delta)*(1-Delta**2/(E*(E+Omega)))*(f[i]*(1-f[i+Omega])*n[Omega-1]-(1-f[i])*f[i+Omega]*(n[Omega-1]+1))
    if Omega > 2*Delta:
        for j in range(Omega-2*Delta):
            E = Delta + j
            result += (-1/np.pi/tau0_ph/Delta)*rho(E, Delta)*rho(Omega-E, Delta)*(1+Delta**2/(E*(Omega-E)))*((1-f[j])*(1-f[Omega-2*Delta-j])*n[Omega-1]-f[j]*f[Omega-2*Delta-j]*(n[Omega-1]+1))
    result += (Bose(Omega,Tb)-n[Omega-1])/tau_l
    return result

def time_deriv_f(E, Ep, Delta, B, N, tau0, Tc, f, n):
    result = 0
    kb = 8.617333262145e-5 * 1e6
    for i in range(N):
        Omega = i+1
        Ei = int(E-Delta)
        if Ei+Omega < N:
            result += (-1/tau0/(kb*Tc)**3)*(Omega**2)*rho(E+Omega, Delta)*(1-Delta**2/(E*(E+Omega)))*(f[Ei]*(1-f[Ei+Omega])*n[i]-(1-f[Ei])*f[Ei+Omega]*(n[i]+1))
    if E > Delta:
        for j in range(E-Delta):
            Omega = j+1
            Ej = int(E-Delta)
            result += (-1/tau0/(kb*Tc)**3)*(Omega**2)*rho(E-Omega, Delta)*(1-Delta**2/(E*(E-Omega)))*(f[Ej]*(1-f[Ej-Omega])*(n[j]+1)-(1-f[Ej])*f[Ej-Omega]*n[j])
    if N > (E+Delta):
        for k in range(E+Delta,N):
            Omega = k+1
            Ek = int(E-Delta)
            result += (-1/tau0/(kb*Tc)**3)*(Omega**2)*rho(Omega-E, Delta)*(1+Delta**2/(E*(Omega-E)))*(f[Ek]*f[k-E-Delta]*(n[k]+1)-(1-f[Ek])*(1-f[k-E-Delta])*n[k])
    result += I_qp(E, Ep, Delta, B, N, f)
    return result

def Jacobian(i, j, N, alpha, Ep, Delta, tau0, Tc, N0, tau0_ph, tau_l, Tb):
    result = 0
    f = alpha[:N]
    B = alpha[N]
    n = alpha[N+1:2*N+1]
    kb = 8.617333262145e-5 * 1e6
    if i < N:
        if j == i:
            for k in range(N):
                Omega = k+1
                if i+Omega < N:
                    result += (-1/tau0/(kb*Tc)**3)*(Omega**2)*rho(Delta+i+Omega, Delta)*(1-Delta**2/((Delta+i)*((Delta+i)+Omega)))*((1-f[i+Omega])*n[k]+f[i+Omega]*(n[k]+1))
            for k in range(i+1):
                Omega = k+1
                result += (-1/tau0/(kb*Tc)**3)*(Omega**2)*rho((Delta+i)-Omega, Delta)*(1-Delta**2/((Delta+i)*((Delta+i)-Omega)))*(f[i-Omega]*n[k]+(1-f[i-Omega])*(n[k]+1))
            if N > ((Delta+i)+Delta):
                for k in range((Delta+i)+Delta,N):
                    Omega = k+1
                    result += (-1/tau0/(kb*Tc)**3)*(Omega**2)*rho(Omega-(Delta+i), Delta)*(1+Delta**2/((Delta+i)*(Omega-(Delta+i))))*(f[k-i-2*Delta]*(n[k]+1)+(1-f[k-i-2*Delta])*n[k])
        if j == i:
            result += 2*B*(-rho((Delta+i)+Ep, Delta)*(1+Delta**2/((Delta+i)*((Delta+i)+Ep))))
        if j == i:
            result += 2*B*(-rho((Delta+i)-Ep, Delta)*(1+Delta**2/((Delta+i)*((Delta+i)-Ep))))
        if j == i+Ep:
            result += 2*B*rho((Delta+i)+Ep, Delta)*(1+Delta**2/((Delta+i)*((Delta+i)+Ep)))
        if j == i-Ep:
            result += 2*B*rho((Delta+i)-Ep, Delta)*(1+Delta**2/((Delta+i)*((Delta+i)-Ep)))
        if i < j < N:
            Omega = j-i
            result += (-1/tau0/(kb*Tc)**3)*(Omega**2)*rho(Delta+i+Omega, Delta)*(1-Delta**2/((Delta+i)*((Delta+i)+Omega)))*(-f[i]*n[Omega-1]-(1-f[i])*(n[Omega-1]+1))
        if j < i:
            Omega = i-j
            result += (-1/tau0/(kb*Tc)**3)*(Omega**2)*rho(Delta+i-Omega, Delta)*(1-Delta**2/((Delta+i)*((Delta+i)-Omega)))*(-f[i]*(n[Omega-1]+1)-(1-f[i])*(n[Omega-1]))
        if j < N:
            Omega = i+j+2*Delta
            if Omega <= N:
                result += (-1/tau0/(kb*Tc)**3)*(Omega**2)*rho(Omega-(Delta+i), Delta)*(1+Delta**2/((Delta+i)*(Omega-(Delta+i))))*(f[i]*(n[Omega-1]+1)+(1-f[i])*n[Omega-1])
        if j == N:
            result += I_qp(Delta+i, Ep, Delta, 1, N, f)
        if N < j < 2*N+1:
            Omega = j - N
            if i+Omega < N:
                result += (-1/tau0/(kb*Tc)**3)*(Omega**2)*rho(Delta+i+Omega, Delta)*(1-Delta**2/((Delta+i)*((Delta+i)+Omega)))*(f[i]*(1-f[i+Omega])-(1-f[i])*f[i+Omega])
        if N < j < 2*N+1 and j - N - 1 < i:
            Omega = j - N
            result += (-1/tau0/(kb*Tc)**3)*(Omega**2)*rho(Delta+i-Omega, Delta)*(1-Delta**2/((Delta+i)*((Delta+i)-Omega)))*(f[i]*(1-f[i-Omega])-(1-f[i])*f[i-Omega])
        if N < j < 2*N+1 and j - N > 2*Delta+i:
            Omega = j - N
            result += (-1/tau0/(kb*Tc)**3)*(Omega**2)*rho(Omega-(Delta+i), Delta)*(1+Delta**2/((Delta+i)*(Omega-(Delta+i))))*(f[i]*f[Omega-i-2*Delta-1]-(1-f[i])*(1-f[Omega-i-2*Delta-1]))
    elif i == N:
        if j < N:
            E = Delta + j
            if E+Ep-Delta < N:
                result += 4*N0*2*B*(-rho(E+Ep, Delta)*(1+Delta**2/(E*(E+Ep)))*E*rho(E,Delta)+rho(E, Delta)*(1+Delta**2/(E*(E+Ep)))*(E+Ep)*rho(E+Ep,Delta))
            if E-Ep-Delta >= 0:
                result += 4*N0*2*B*(-rho(E-Ep, Delta)*(1+Delta**2/(E*(E-Ep)))*E*rho(E,Delta)+rho(E, Delta)*(1+Delta**2/(E*(E-Ep)))*(E-Ep)*rho(E-Ep,Delta))
        elif j == N:
            result = Power(Ep, Delta, 1, N, f, N0)
        else:
            result = 0
    else:
        Omega = i - N
        if j < N:
            E = Delta + j
            if j+Omega < N:
                result += (-2/np.pi/tau0_ph/Delta)*rho(E, Delta)*rho(E+Omega, Delta)*(1-Delta**2/(E*(E+Omega)))*((1-f[j+Omega])*n[Omega-1]+f[j+Omega]*(n[Omega-1]+1))
            if j-Omega >= 0:
                result += (-2/np.pi/tau0_ph/Delta)*rho(E, Delta)*rho(E-Omega, Delta)*(1-Delta**2/(E*(E-Omega)))*(-(1-f[j-Omega])*(n[Omega-1]+1)-f[j-Omega]*(n[Omega-1]))
        if j < Omega - 2*Delta:
            E = Delta + j
            result += 2*(-1/np.pi/tau0_ph/Delta)*rho(E, Delta)*rho(Omega-E, Delta)*(1+Delta**2/(E*(Omega-E)))*(-(1-f[Omega-2*Delta-j])*n[Omega-1]-f[Omega-2*Delta-j]*(n[Omega-1]+1))
        if j == N:
            result = 0
        if j == i:
            for k in range(N):
                E = Delta + k
                if k+Omega < N:
                    result += (-2/np.pi/tau0_ph/Delta)*rho(E, Delta)*rho(E+Omega, Delta)*(1-Delta**2/(E*(E+Omega)))*(f[k]*(1-f[k+Omega])-(1-f[k])*f[k+Omega])
            if Omega > 2*Delta:
                for k in range(Omega-2*Delta):
                    E = Delta + k
                    result += (-1/np.pi/tau0_ph/Delta)*rho(E, Delta)*rho(Omega-E, Delta)*(1+Delta**2/(E*(Omega-E)))*((1-f[k])*(1-f[Omega-2*Delta-k])-f[k]*f[Omega-2*Delta-k])
            result += -1/tau_l
    return result

def Power(Ep, Delta, B, N, f, N0):
    result = 0
    for i in range(N):
        E = Delta + i
        result += 4*N0*E*rho(E,Delta)*I_qp(E, Ep, Delta, B, N, f)
    return result

def Power_qp_phi(Ep, Delta, B, N, f, N0, err):
    result = 0
    for i in range(N):
        E = Delta + i
        result += 4*N0*E*rho(E,Delta)*(I_qp(E, Ep, Delta, B, N, f)-err[i])
    return result

def Power_phi_b(n, Delta, N, Tc, N0, tau0, tau0_ph, tau_l, Tb):
    result = 0
    kb = 8.617333262145e-5 * 1e6
    N_ion_over_OmegaD_cube = 2*np.pi*N0*tau0_ph*Delta / (9*tau0*(kb*Tc)**3)
    for i in range(N):
        Omega = i + 1
        result += 9*N_ion_over_OmegaD_cube*(Omega**3)*(n[i]-Bose(Omega,Tb))/tau_l
    return result

def Power_phi_b_comp(n, Delta, N, Tc, N0, tau0, tau0_ph, tau_l, Tb):
    result = np.zeros(N)
    kb = 8.617333262145e-5 * 1e6
    N_ion_over_OmegaD_cube = 2*np.pi*N0*tau0_ph*Delta / (9*tau0*(kb*Tc)**3)
    for i in range(N):
        Omega = i + 1
        result[i] = 9*N_ion_over_OmegaD_cube*(Omega**3)*(n[i]-Bose(Omega,Tb))/tau_l
    return result

def calc_Nnqp(N, N0, f, Delta):
    result = 0
    for i in range(N):
        E = Delta + i
        result += 4*N0*E*rho(E,Delta)*f[i]
    return result

def calc_Ntqp(N, N0, T, Delta):
    result = 0
    for i in range(N):
        E = Delta + i
        result += 4*N0*E*rho(E,Delta)*Fermi(E, T)
    return result