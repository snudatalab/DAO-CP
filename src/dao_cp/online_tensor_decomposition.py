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
import time
import sys
import numpy as np
import tensorly as tl

from warnings import warn
from tensorly.decomposition import parafac

from dao_cp.welford import *
from dao_cp.tensor import *
from dao_cp.methods.ocp import *
from dao_cp.methods.dtd import *
from dao_cp.methods.dao import *

def online_tensor_decomposition(dataset, X, X_stream, rank, n_iter=1, ul=-1, ll=-1, verbose=False, methods=['dao', 'dtd', 'ocp', 'fcp']):
    results = {}
    start = time.time()
    (weights, factors_old) = parafac(X_stream[0], rank, init='random')
    init_time = time.time()-start
#    print('making init decomposition result:', init_time)
    
    for method in methods:

        print('-------------------------------------')
        mem_usage = sys.getsizeof(X_stream[0])
        if method in ['dao', 'dtd']:
            print(f'>> {method}: rank-{rank} n_iter-{n_iter}')
        elif method in ['ocp', 'fcp']:
            print(f'>> {method}: rank-{rank}')
        
        factors = factors_old
        X_old = X_stream[0]
        n_dim = tl.ndim(X_old)
        if not method in ['dao', 'dtd', 'ocp', 'fcp']:
            raise ValueError('The method does not exist.')  
        if method == 'fcp':
            mem_usage = sys.getsizeof(X)
            ktensor = parafac(X, rank, init='random')
            (weights, factors) = ktensor
            mem_usage += sys.getsizeof(factors)
            X_est = construct_tensor(factors)
            err_norm = tl.norm(X - X_est)
            global_rt = time.time()-start
            global_fit = 1 - (err_norm/tl.norm(X))
            print('Global Fitness         :', format(global_fit*100, '.4f'), '%')
            print('Global Running Time    :', format(global_rt, '.4f'), 'sec')
            print('Memory Usage           :', mem_usage, 'bytes')
#            results[method] = [global_fit, 0, global_rt, 0, mem_usage, X_est]
            results[method] = [ktensor]
            continue

        ktensors = []
        verbose_list = []
        split_points = []
        refine_points = []
        fitness = []
        running_time = []
        begin = time.time() - init_time

        welford = Welford()
        X_est = construct_tensor(factors)
        err_norm = tl.norm(X_old - X_est)
        welford(err_norm * 1.2)


        if method == 'ocp':
            start = time.time()
            K = get_KhatriRao_except0(factors)
            H = get_Hadamard(factors)

            P = np.empty((n_dim), dtype=object)
            Q = np.empty((n_dim), dtype=object)

            for mode in range(1, n_dim):
                P[mode] = tl.dot(tl.unfold(X_old, mode), tl.tenalg.khatri_rao((factors[0], K[mode])))
                Q[mode] = H / tl.dot(tl.transpose(factors[mode]), factors[mode])
            #print('init_time:', time.time()-start)
            mem_usage += sys.getsizeof(K)
            mem_usage += sys.getsizeof(H)
            mem_usage += sys.getsizeof(P)
            mem_usage += sys.getsizeof(Q)
        
        iter_mem_usage = 0
        for i, X_new in enumerate(X_stream[1:]):
            i_mem = sys.getsizeof(X_new)
            start = time.time()
            if method == 'dao':
                (weights, factors0) = data_adaptive_online_cp(factors.copy(), X_old, X_new, rank, n_iter=n_iter, mu=0.8, verbose=False)
            elif method == 'ocp':
                ((weights, factors0), P0, Q0) = online_cp(factors.copy(), X_old, X_new, rank, P, Q, verbose=False)
            elif method == 'dtd':
                (weights, factors0) = dtd(factors.copy(), X_old, X_new, rank, n_iter=n_iter, mu=1, verbose=False)

            U = factors0.copy()
            U[0] = U[0][-X_new.shape[0]-1:-1]
            i_mem += sys.getsizeof(U)
            dX_est = construct_tensor(U)

            err_norm = tl.norm(X_new - dX_est)
            z_score = get_z_score(err_norm, welford.mean, welford.std)

            if method == 'dao' and ul > 0 and z_score > ul:
                weights = tl.ones(rank)
                ktensors.append(KruskalTensor((weights, factors.copy())))
                #print('=== SPLIT({}, {}) ==='.format(z_score, err_norm))
                split_points.append(i+1)

                X_old = X_stream[i+1]

                (weights, factors0) = parafac(X_old, rank, init='random')
                elapsed_time = time.time()-start
                #print('making init decomposition result:', time.time()-start)
                verbose_list.append([i+1, elapsed_time, err_norm, z_score])

                i_mem += sys.getsizeof(factors0)
                start = time.time()
                X_est = construct_tensor(factors0)
                err_norm = tl.norm(X_old - X_est)
                welford = Welford()
                welford(err_norm * 1.2)

                z_score = get_z_score(err_norm, welford.mean, welford.std)
                factors = factors0.copy()
                welford(err_norm)
                elapsed_time = time.time()-start
                #print('{}th_iter:'.format(i+1), elapsed_time, err_norm, z_score)
                verbose_list.append([i+1, elapsed_time, err_norm, z_score])
                fitness.append(err_norm/tl.norm(X_new))
                running_time.append(elapsed_time)
                continue
            elif method == 'dao' and ll > 0 and z_score > ll:
                #print('=== REFINE({}, {}) ==='.format(z_score, err_norm))
                refine_points.append(i+1)
                elapsed_time = time.time()-start
                verbose_list.append([i+1, elapsed_time, err_norm, z_score])

                (weights, factors) = data_adaptive_online_cp(factors, X_old, X_new, rank, n_iter=n_iter*2, mu=0.5, verbose=False)
                
                i_mem += sys.getsizeof(factors)
                U = factors.copy()
                U[0] = U[0][-X_new.shape[0]-1:-1]
                dX_est = construct_tensor(U)
                err_norm = tl.norm(X_new - dX_est)
                welford(err_norm)
            else:
                if method == 'ocp':
                    P = P0
                    Q = Q0
                factors = factors0.copy()
                welford(err_norm)
            
            elapsed_time = time.time()-start
            #print('{}th_iter:'.format(i+1), elapsed_time, err_norm, z_score)
            verbose_list.append([i+1, elapsed_time, err_norm, z_score])
            fitness.append(err_norm/tl.norm(X_new))
            running_time.append(elapsed_time)
            X_old = tl.concatenate((X_old, X_new))
            iter_mem_usage = max(iter_mem_usage, i_mem)
            if verbose:
                X_est = construct_tensor(factors)
                compare_tensors(X_old, X_est)

        mem_usage += iter_mem_usage
        
        weights = tl.ones(rank)
        ktensors.append(KruskalTensor((weights, factors)))
        mem_usage += sys.getsizeof(ktensors)
        
#     return (ktensors, np.asarray(verbose_list))
        global_rt = time.time() - begin

        tensor_est = construct_tensor(ktensors[0][1])
        for (weights, factors) in ktensors[1:]:
            tensor_est = tl.tensor(tl.concatenate((tensor_est, construct_tensor(factors))))
        global_error_norm = compare_tensors(X, tensor_est)
#        print('Elapsed Time:', time.time() - begin)
#         print_tensor(np.asarray((X, tensor_est))[:,0,0,0,:10])
        if method == 'dao':
            print(f'SPLIT: {len(split_points)}, REFINE: {len(refine_points)}')
            
        if method != 'fcp':
            verbose_list = np.asarray(verbose_list, dtype=float)
            fitness = np.asarray(fitness, dtype=float)
            running_time = np.asarray(running_time, dtype=float)

            tot_norm = tl.norm(X)
            local_fit = 1 - np.mean(fitness)
            local_rt = np.mean(running_time)
            global_fit = 1 - (global_error_norm / tot_norm)
            print('Global Fitness         :', format(global_fit*100, '.4f'), '%')
            print('Avg Local Fitness      :', format(local_fit*100, '.4f'), '%')
            print('Global Running Time    :', format(global_rt, '.4f'), 'sec')
            print('Avg Local Running Time :', format(local_rt, '.4f'), 'sec')
            print('Memory Usage           :', mem_usage, 'bytes')
#            results[method] = [global_fit, local_fit, global_rt, local_rt, mem_usage, verbose_list, (split_points, refine_points), tensor_est]
            results[method] = ktensors
            
    return results