"""
 DAO-CP: Data Adaptive Online CP Decomposition (ECML PKDD 2021)

Authors:
- Sangjun Son      (lucetre@snu.ac.kr), Seoul National University
- Yongchan Park (wjdakf3948@snu.ac.kr), Seoul National University
- Minyong Cho   (chominyong@gmail.com), Seoul National University
- U Kang             (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import numpy as np
import tensorly as tl

from dao_cp.tensor import *
from tensorly.decomposition.candecomp_parafac import initialize_factors, unfolding_dot_khatri_rao, KruskalTensor

def data_adaptive_online_cp(factors_old, X_old, X_new, rank, n_iter=1, mu=1, verbose=False):
    
    weights = tl.ones(rank)
    if verbose:
        X = tl.tensor(np.concatenate((X_old, X_new)))
    n_dim = tl.ndim(X_old)
    U = factors_old.copy()
    
    H = get_Hadamard(U[1:])
    G = H
    
    ATA0 = tl.dot(tl.transpose(U[0]), U[0])
    ATA1 = tl.dot(tl.transpose(U[1]), U[1])
    
    for i in range(n_iter):        
        # temporal mode for A1
        mttkrp_parts = []
        for r in range(rank):
            component = tl.tenalg.multi_mode_dot(X_new, [f[:, r] for f in U], skip=0)
            mttkrp_parts.append(component)
        mttkrp = np.stack(mttkrp_parts, axis=1)
        
        A1 = tl.transpose(tl.solve(tl.transpose(H), tl.transpose(mttkrp)))
        ATA1 = tl.dot(tl.transpose(A1), A1)
        
        
        # non-temporal mode
        for mode in range(1, n_dim):
            
            U1 = U.copy()
            U1[0] = A1
                    
            G = G / tl.dot(tl.transpose(factors_old[mode]), U[mode])
            W = G * tl.dot(tl.transpose(factors_old[0]), U[0])
            mttkrp0 = mu * tl.dot(factors_old[mode], W)
            mttkrp1 = unfolding_dot_khatri_rao(X_new, (None, U1), mode)
            
            H = H / tl.dot(tl.transpose(U[mode]), U[mode])
            V = H * (mu * ATA0 + ATA1)
            U[mode] = tl.transpose(tl.solve(tl.transpose(V), tl.transpose(mttkrp0 + mttkrp1)))
            G = G * tl.dot(tl.transpose(factors_old[mode]), U[mode])
            H = H * tl.dot(tl.transpose(U[mode]), U[mode])
            

        # temporal mode for A0        
        mttkrp = tl.dot(factors_old[0], G)
        U[0] = tl.transpose(tl.solve(tl.transpose(H), tl.transpose(mttkrp)))
        ATA0 = tl.dot(tl.transpose(U[0]), U[0])

        if verbose:
            U1 = U.copy()
            U1[0] = np.concatenate((U[0], A1))
            X_est = construct_tensor(U1)
            compare_tensors(X, X_est)

    U[0] = np.concatenate((U[0].copy(), A1))
    return KruskalTensor((weights, U))
