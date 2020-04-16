# -*- coding: utf-8 -*-
from scipy import optimize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def LLKLIEP(theta, kP, kQ):
    """
    Calculate log-likelihood
    
    parameters:
    theta:
        Markov network parameters
    kP, kQ:
        Basis function of numerator and denominator
    
    return:
    l:
        log-likelihood
    """
    theta = theta[:,None].T
    l = -np.mean(theta.dot(kP)) + np.log(np.mean(np.exp(theta.dot(kQ))))
    
    return l
        
    
def LLKLIEP_jac(theta, kP, kQ):
    """
    return:
    g:
        gradient of LLKELIEP
    """
    N_q = np.sum(np.exp(theta.dot(kQ)))
    g_q = np.exp(theta.dot(kQ)) / N_q
    g = -np.mean(kP,1) + kQ.dot(g_q.T)
    
    return g
    
def calc_karnel(data, sigma=0.5, kernel='RBF'):
    """
    Basis function calculation
    
    parameters:
    data:
    sigma:
        Bandwidth with RBF kernel.
    kernel:
        Basis function specification. Select 'RBF' or 'lin'.
    
    return:
    phi:
        Basis function calculation result.
    """
    (dim, num) = data.shape
    #Combination of each dimension that requires calculation of basis functions
    
    calc_area = np.tri(dim, dim, 0)
    calc_area = (calc_area==1)
    calc_area = calc_area[:,:,None]
    area = np.tile(calc_area,(1, 1, num))

    data = data[:,:,None]
    data = np.rollaxis(data, 2, 1)
    base = np.tile(data,(1, data.shape[0], 1))

    trans = np.rollaxis(base.T, 0, 3)        
    
    calc = np.empty([dim,dim,num])
    if kernel == 'RBF':
        calc[area] = np.exp(-((base[area]-trans[area])**2 / (2*sigma**2)))
    elif kernel == 'lin':
        calc[area] = base[area] * trans[area]
    else:
        calc[area] = base[area] * trans[area]
    
    phi = np.reshape(calc[area], (int(dim*(dim+1)*0.5), num))
    
    return phi

if __name__ == '__main__':
    #Artificial data
    #Multivariate normal distribution
    dim = 30
    diff_edge = 6
    size = 500
    
    P_mean, Q_mean = np.full(dim, 0), np.full(dim, 0)
    
    P_cov, Q_cov = np.eye(dim), np.eye(dim)
    index = [x for x in range(dim)]
    diff_edge_elm = np.array([np.random.choice(index, 2, replace=False) for _ in range(diff_edge)])
    Q_cov[diff_edge_elm[:,0], diff_edge_elm[:,1]] = 0.5
    Q_cov[diff_edge_elm[:,1], diff_edge_elm[:,0]] = 0.5
    
    P = np.random.multivariate_normal(P_mean, P_cov, size)
    Q = np.random.multivariate_normal(Q_mean, Q_cov, size)
    
    #calc kernel
    kP = calc_karnel(P.T, kernel='lin')
    kQ = calc_karnel(Q.T, kernel='lin')
    
    #optimize
    arg = (kP,kQ)
    '''
    print(optimize.check_grad(LLKLIEP, LLKLIEP_jac, \
          np.zeros(int(dim*(dim+1)*0.5)), kP, kQ))
    '''
    result = optimize.minimize(LLKLIEP, np.zeros(int(dim*(dim+1)*0.5)), \
                               jac=LLKLIEP_jac, args=arg, method='L-BFGS-B')
    theta = result['x']
    
    #Transform estimated parameters into matrix
    #Create triangular matrix
    calc_area = np.ones((dim, dim))
    calc_area = np.triu(calc_area, 1)
    calc_area = np.reshape(calc_area.T, (np.size(calc_area), 1))
    calc_area = (calc_area==1)
    theta = np.reshape(theta, [theta.shape[0]])
    Delta = np.zeros((1, dim*dim)).T
    
    area_bool = np.eye(dim)
    area_bool = area_bool == 1
    tr = np.tri(dim, dim)
    tr = (tr == 1)
    tr_area = area_bool[tr]
    
    Delta[calc_area] = theta[tr_area==False]
    
    Delta = np.reshape(Delta, (dim, dim))
    Delta = Delta + Delta.T
    
    #plot
    sns.heatmap(P_cov-Q_cov, square=True, cbar=False)
    plt.savefig('ground_truth.png')
    plt.clf()
    sns.heatmap(Delta, square=True, cbar=False)
    plt.savefig('KLIEP.png')
    
    