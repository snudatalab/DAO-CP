"""
 DAO-CP: Data Adaptive Online CP Decomposition (PLOS ONE 2021)

Authors:
- Sangjun Son      (lucetre@snu.ac.kr), Seoul National University
- Yongchan Park (wjdakf3948@snu.ac.kr), Seoul National University
- Minyong Cho   (chominyong@gmail.com), Seoul National University
- U Kang             (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import sys

from dao_cp.tensor import *
from dao_cp.dataset import *
from dao_cp.online_tensor_decomposition import *

""" Parameter settings for each dataset """
model_params = {
    'synthetic': (1, 1.2, 1.1, -10, 30),
    'video': (1, 6, 2, -5, 30),
    'stock': (1, 6, 5, -3, 20),
    'hall': (1, 0.5, 0.1, -10, 30),
    'korea': (1, 2, 1.3, -100, 40),
}

def main(data):
    """ Load data. """
    print(f'Start loading \'{data}\' dataset...')
    org_tensor = get_dataset(data)

    """ Save the original tensor. """
    filename = f'out/{data}-org.npy'
    np.save(f'../{filename}', org_tensor)
    print(f'Original tensor of shape {org_tensor.shape} saved as \'{filename}\'!')
    
    """ Get parameter settings and start online decomposition. """
    (n_iter, ul, ll, start_to_stream, rank) = model_params[data]
    tensor_stream = create_tensor_stream(org_tensor, start_to_stream)
    results = online_tensor_decomposition(data, org_tensor, tensor_stream, rank, n_iter=n_iter, ul=ul, ll=ll, verbose=False)
    
    print('=========================================================================')
    for method in results:
        print(method)
        ktensors = results[method]
        for ktensor in ktensors:
            print(f'    {ktensor}')


        """ Construct the estimated tensor by concatenating split tensor slices. """
        tensor_est = construct_tensor(ktensors[0][1])
        for (weights, factors) in ktensors[1:]:
            tensor_est = tl.tensor(tl.concatenate((tensor_est, construct_tensor(factors))))
        filename = f'out/{data}-{method}.npy'
        np.save(f'../{filename}', tensor_est)
        print(f'Estimated tensor of shape {tensor_est.shape} saved as \'{filename}\'!')
        print('=========================================================================')
    
if __name__ == "__main__":
    dataset_name = sys.argv[1]
    if dataset_name in ['synthetic', 'video', 'stock', 'hall', 'korea']:
        main(dataset_name)
    else:
        raise Exception(f'No datasets named \'{dataset_name}\'')