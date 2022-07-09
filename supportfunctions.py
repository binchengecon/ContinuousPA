import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import norm
import pdb
import warnings
import time
import pickle
import SolveLinSys
# from numba import njit

def finiteDiff(data, dim, order, dlt, cap = None):  
    # compute the central difference derivatives for given input and dimensions
    res = np.zeros(data.shape)
    l = len(data.shape)
    if l == 3:
        if order == 1:                    # first order derivatives
            
            if dim == 0:                  # to first dimension

                res[1:-1,:,:] = (1 / (2 * dlt)) * (data[2:,:,:] - data[:-2,:,:])
                res[-1,:,:] = (1 / dlt) * (data[-1,:,:] - data[-2,:,:])
                res[0,:,:] = (1 / dlt) * (data[1,:,:] - data[0,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:] = (1 / (2 * dlt)) * (data[:,2:,:] - data[:,:-2,:])
                res[:,-1,:] = (1 / dlt) * (data[:,-1,:] - data[:,-2,:])
                res[:,0,:] = (1 / dlt) * (data[:,1,:] - data[:,0,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1] = (1 / (2 * dlt)) * (data[:,:,2:] - data[:,:,:-2])
                res[:,:,-1] = (1 / dlt) * (data[:,:,-1] - data[:,:,-2])
                res[:,:,0] = (1 / dlt) * (data[:,:,1] - data[:,:,0])

            else:
                raise ValueError('wrong dim')
                
        elif order == 2:
            
            if dim == 0:                  # to first dimension

                res[1:-1,:,:] = (1 / dlt ** 2) * (data[2:,:,:] + data[:-2,:,:] - 2 * data[1:-1,:,:])
                res[-1,:,:] = (1 / dlt ** 2) * (data[-1,:,:] + data[-3,:,:] - 2 * data[-2,:,:])
                res[0,:,:] = (1 / dlt ** 2) * (data[2,:,:] + data[0,:,:] - 2 * data[1,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:] = (1 / dlt ** 2) * (data[:,2:,:] + data[:,:-2,:] - 2 * data[:,1:-1,:])
                res[:,-1,:] = (1 / dlt ** 2) * (data[:,-1,:] + data[:,-3,:] - 2 * data[:,-2,:])
                res[:,0,:] = (1 / dlt ** 2) * (data[:,2,:] + data[:,0,:] - 2 * data[:,1,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1] = (1 / dlt ** 2) * (data[:,:,2:] + data[:,:,:-2] - 2 * data[:,:,1:-1])
                res[:,:,-1] = (1 / dlt ** 2) * (data[:,:,-1] + data[:,:,-3] - 2 * data[:,:,-2])
                res[:,:,0] = (1 / dlt ** 2) * (data[:,:,2] + data[:,:,0] - 2 * data[:,:,1])

            else:
                raise ValueError('wrong dim')
            
        else:
            raise ValueError('wrong order')
    elif l == 2:
        if order == 1:                    # first order derivatives
            
            if dim == 0:                  # to first dimension

                res[1:-1,:] = (1 / (2 * dlt)) * (data[2:,:] - data[:-2,:])
                res[-1,:] = (1 / dlt) * (data[-1,:] - data[-2,:])
                res[0,:] = (1 / dlt) * (data[1,:] - data[0,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1] = (1 / (2 * dlt)) * (data[:,2:] - data[:,:-2])
                res[:,-1] = (1 / dlt) * (data[:,-1] - data[:,-2])
                res[:,0] = (1 / dlt) * (data[:,1] - data[:,0])

            else:
                raise ValueError('wrong dim')
                
        elif order == 2:
            
            if dim == 0:                  # to first dimension

                res[1:-1,:] = (1 / dlt ** 2) * (data[2:,:] + data[:-2,:] - 2 * data[1:-1,:])
                res[-1,:] = (1 / dlt ** 2) * (data[-1,:] + data[-3,:] - 2 * data[-2,:])
                res[0,:] = (1 / dlt ** 2) * (data[2,:] + data[0,:] - 2 * data[1,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1] = (1 / dlt ** 2) * (data[:,2:] + data[:,:-2] - 2 * data[:,1:-1])
                res[:,-1] = (1 / dlt ** 2) * (data[:,-1] + data[:,-3] - 2 * data[:,-2])
                res[:,0] = (1 / dlt ** 2) * (data[:,2] + data[:,0] - 2 * data[:,1])

            else:
                raise ValueError('wrong dim')
            
        else:
            raise ValueError('wrong order')

    elif l == 4:
        if order == 1:                    # first order derivatives
            
            if dim == 0:                  # to first dimension

                res[1:-1,:,:,:] = (1 / (2 * dlt)) * (data[2:,:,:,:] - data[:-2,:,:,:])
                res[-1,:,:,:] = (1 / dlt) * (data[-1,:,:,:] - data[-2,:,:,:])
                res[0,:,:,:] = (1 / dlt) * (data[1,:,:,:] - data[0,:,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:,:] = (1 / (2 * dlt)) * (data[:,2:,:,:] - data[:,:-2,:,:])
                res[:,-1,:,:] = (1 / dlt) * (data[:,-1,:,:] - data[:,-2,:,:])
                res[:,0,:,:] = (1 / dlt) * (data[:,1,:,:] - data[:,0,:,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1,:] = (1 / (2 * dlt)) * (data[:,:,2:,:] - data[:,:,:-2,:])
                res[:,:,-1,:] = (1 / dlt) * (data[:,:,-1,:] - data[:,:,-2,:])
                res[:,:,0,:] = (1 / dlt) * (data[:,:,1,:] - data[:,:,0,:])
            
            elif dim == 3:

                res[:,:,:, 1:-1] = (1 / (2 * dlt)) * (data[:,:,:,2:] - data[:,:,:,:-2])
                res[:,:,:,-1] = (1 / dlt) * (data[:,:,:,-1]- data[:, :, :, -2])
                res[:,:,:,0] = (1 / dlt) * (data[:,:,:,1] - data[:,:,:,0])
                

            else:
                raise ValueError('wrong dim')
                
        elif order == 2:
            
            if dim == 0:                  # to first dimension

                res[1:-1,:,:,:] = (1 / dlt ** 2) * (data[2:,:,:,:] + data[:-2,:,:,:] - 2 * data[1:-1,:,:,:])
                res[-1,:,:,:] = (1 / dlt ** 2) * (data[-1,:,:,:] + data[-3,:,:, :] - 2 * data[-2,:,:,:])
                res[0,:,:,:] = (1 / dlt ** 2) * (data[2,:,:,:] + data[0,:,:,:] - 2 * data[1,:,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:,:] = (1 / dlt ** 2) * (data[:,2:,:,:] + data[:,:-2,:,:] - 2 * data[:,1:-1,:,:])
                res[:,-1,:,:] = (1 / dlt ** 2) * (data[:,-1,:,:] + data[:,-3,:,:] - 2 * data[:,-2,:,:])
                res[:,0,:,:] = (1 / dlt ** 2) * (data[:,2,:,:] + data[:,0,:,:] - 2 * data[:,1,:,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1,:] = (1 / dlt ** 2) * (data[:,:,2:,:] + data[:,:,:-2,:] - 2 * data[:,:,1:-1,:])
                res[:,:,-1,:] = (1 / dlt ** 2) * (data[:,:,-1,:] + data[:,:,-3,:] - 2 * data[:,:,-2,:])
                res[:,:,0,:] = (1 / dlt ** 2) * (data[:,:,2,:] + data[:,:,0,:] - 2 * data[:,:,1,:])

            elif dim == 3:                # to third dimension

                res[:,:,:,1:-1] = (1 / dlt ** 2) * (data[:,:,:,2:] + data[:,:,:,:-2] - 2 * data[:,:,:,1:-1])
                res[:,:,:,-1] = (1 / dlt ** 2) * (data[:,:,:,-1] + data[:,:,:,-3] - 2 * data[:,:,:,-2])
                res[:,:,:,0] = (1 / dlt ** 2) * (data[:,:,:,2] + data[:,:,:,0] - 2 * data[:,:,:,1])
            else:
                raise ValueError('wrong dim')
            
        else:
            raise ValueError('wrong order')
    else:
        raise ValueError("Dimension NOT supported")
        
    if cap is not None:
        res[res < cap] = cap
    return res

def finiteDiff_3D(data, dim, order, dlt, cap = None):  
    # compute the central difference derivatives for given input and dimensions
    res = np.zeros(data.shape)
    l = len(data.shape)
    if l == 3:
        if order == 1:                    # first order derivatives
            
            if dim == 0:                  # to first dimension

                res[1:-1,:,:] = (1 / (2 * dlt)) * (data[2:,:,:] - data[:-2,:,:])
                res[-1,:,:] = (1 / dlt) * (data[-1,:,:] - data[-2,:,:])
                res[0,:,:] = (1 / dlt) * (data[1,:,:] - data[0,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:] = (1 / (2 * dlt)) * (data[:,2:,:] - data[:,:-2,:])
                res[:,-1,:] = (1 / dlt) * (data[:,-1,:] - data[:,-2,:])
                res[:,0,:] = (1 / dlt) * (data[:,1,:] - data[:,0,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1] = (1 / (2 * dlt)) * (data[:,:,2:] - data[:,:,:-2])
                res[:,:,-1] = (1 / dlt) * (data[:,:,-1] - data[:,:,-2])
                res[:,:,0] = (1 / dlt) * (data[:,:,1] - data[:,:,0])

            else:
                raise ValueError('wrong dim')
                
        elif order == 2:
            
            if dim == 0:                  # to first dimension

                res[1:-1,:,:] = (1 / dlt ** 2) * (data[2:,:,:] + data[:-2,:,:] - 2 * data[1:-1,:,:])
                res[-1,:,:] = (1 / dlt ** 2) * (data[-1,:,:] + data[-3,:,:] - 2 * data[-2,:,:])
                res[0,:,:] = (1 / dlt ** 2) * (data[2,:,:] + data[0,:,:] - 2 * data[1,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:] = (1 / dlt ** 2) * (data[:,2:,:] + data[:,:-2,:] - 2 * data[:,1:-1,:])
                res[:,-1,:] = (1 / dlt ** 2) * (data[:,-1,:] + data[:,-3,:] - 2 * data[:,-2,:])
                res[:,0,:] = (1 / dlt ** 2) * (data[:,2,:] + data[:,0,:] - 2 * data[:,1,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1] = (1 / dlt ** 2) * (data[:,:,2:] + data[:,:,:-2] - 2 * data[:,:,1:-1])
                res[:,:,-1] = (1 / dlt ** 2) * (data[:,:,-1] + data[:,:,-3] - 2 * data[:,:,-2])
                res[:,:,0] = (1 / dlt ** 2) * (data[:,:,2] + data[:,:,0] - 2 * data[:,:,1])

            else:
                raise ValueError('wrong dim')
            
        else:
            raise ValueError('wrong order')
    elif l == 2:
        if order == 1:                    # first order derivatives
            
            if dim == 0:                  # to first dimension

                res[1:-1,:] = (1 / (2 * dlt)) * (data[2:,:] - data[:-2,:])
                res[-1,:] = (1 / dlt) * (data[-1,:] - data[-2,:])
                res[0,:] = (1 / dlt) * (data[1,:] - data[0,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1] = (1 / (2 * dlt)) * (data[:,2:] - data[:,:-2])
                res[:,-1] = (1 / dlt) * (data[:,-1] - data[:,-2])
                res[:,0] = (1 / dlt) * (data[:,1] - data[:,0])

            else:
                raise ValueError('wrong dim')
                
        elif order == 2:
            
            if dim == 0:                  # to first dimension

                res[1:-1,:] = (1 / dlt ** 2) * (data[2:,:] + data[:-2,:] - 2 * data[1:-1,:])
                res[-1,:] = (1 / dlt ** 2) * (data[-1,:] + data[-3,:] - 2 * data[-2,:])
                res[0,:] = (1 / dlt ** 2) * (data[2,:] + data[0,:] - 2 * data[1,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1] = (1 / dlt ** 2) * (data[:,2:] + data[:,:-2] - 2 * data[:,1:-1])
                res[:,-1] = (1 / dlt ** 2) * (data[:,-1] + data[:,-3] - 2 * data[:,-2])
                res[:,0] = (1 / dlt ** 2) * (data[:,2] + data[:,0] - 2 * data[:,1])

            else:
                raise ValueError('wrong dim')
            
        else:
            raise ValueError('wrong order')

    elif l == 1:
        if order == 1:                    # first order derivatives
            
            if dim == 0:                  # to first dimension

                res[1:-1] = (1 / (2 * dlt)) * (data[2:] - data[:-2])
                res[-1] = (1 / dlt) * (data[-1] - data[-2])
                res[0] = (1 / dlt) * (data[1] - data[0])

            else:
                raise ValueError('wrong dim')
                
        elif order == 2:
            
            if dim == 0:                  # to first dimension

                res[1:-1] = (1 / dlt ** 2) * (data[2:] + data[:-2] - 2 * data[1:-1])
                res[-1] = (1 / dlt ** 2) * (data[-1] + data[-3] - 2 * data[-2])
                res[0] = (1 / dlt ** 2) * (data[2] + data[0] - 2 * data[1])

            else:
                raise ValueError('wrong dim')
            
        else:
            raise ValueError('wrong order')            
    else:
        raise ValueError("Dimension NOT supported")
        
    if cap is not None:
        res[res < cap] = cap
    return res


def DiffOne(data, dim, order, dlt, cap = None):  
    # compute the central difference derivatives for given input and dimensions
    res = np.zeros(data.shape)
    l = len(data.shape)
    if l == 3:
        if order == 1:                    # first order derivatives
            
            if dim == 0:                  # to first dimension

                res[1:-1,:,:] = (1 / dlt) * (data[1:,:,:] - data[:-1,:,:])
                res[-1,:,:] = (1 / dlt) * (data[-1,:,:] - data[-2,:,:])
                res[0,:,:] = (1 / dlt) * (data[1,:,:] - data[0,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:] = (1 / dlt) * (data[:,1:,:] - data[:,:-1,:])
                res[:,-1,:] = (1 / dlt) * (data[:,-1,:] - data[:,-2,:])
                res[:,0,:] = (1 / dlt) * (data[:,1,:] - data[:,0,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1] = (1 / dlt) * (data[:,:,1:] - data[:,:,:-1])
                res[:,:,-1] = (1 / dlt) * (data[:,:,-1] - data[:,:,-2])
                res[:,:,0] = (1 / dlt) * (data[:,:,1] - data[:,:,0])

            else:
                raise ValueError('wrong dim')
                
        elif order == 2:
            
            if dim == 0:                  # to first dimension

                res[1:-1,:,:] = (1 / dlt ** 2) * (data[2:,:,:] + data[:-2,:,:] - 2 * data[1:-1,:,:])
                res[-1,:,:] = (1 / dlt ** 2) * (data[-1,:,:] + data[-3,:,:] - 2 * data[-2,:,:])
                res[0,:,:] = (1 / dlt ** 2) * (data[2,:,:] + data[0,:,:] - 2 * data[1,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:] = (1 / dlt ** 2) * (data[:,2:,:] + data[:,:-2,:] - 2 * data[:,1:-1,:])
                res[:,-1,:] = (1 / dlt ** 2) * (data[:,-1,:] + data[:,-3,:] - 2 * data[:,-2,:])
                res[:,0,:] = (1 / dlt ** 2) * (data[:,2,:] + data[:,0,:] - 2 * data[:,1,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1] = (1 / dlt ** 2) * (data[:,:,2:] + data[:,:,:-2] - 2 * data[:,:,1:-1])
                res[:,:,-1] = (1 / dlt ** 2) * (data[:,:,-1] + data[:,:,-3] - 2 * data[:,:,-2])
                res[:,:,0] = (1 / dlt ** 2) * (data[:,:,2] + data[:,:,0] - 2 * data[:,:,1])

            else:
                raise ValueError('wrong dim')
            
        else:
            raise ValueError('wrong order')
    elif l == 2:
        if order == 1:                    # first order derivatives
            
            if dim == 0:                  # to first dimension

                res[1:-1,:] = (1 / dlt) * (data[1:,:] - data[:-1,:])
                res[-1,:] = (1 / dlt) * (data[-1,:] - data[-2,:])
                res[0,:] = (1 / dlt) * (data[1,:] - data[0,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1] = (1 / dlt) * (data[:,1:] - data[:,:-1])
                res[:,-1] = (1 / dlt) * (data[:,-1] - data[:,-2])
                res[:,0] = (1 / dlt) * (data[:,1] - data[:,0])

            else:
                raise ValueError('wrong dim')
                
        elif order == 2:
            
            if dim == 0:                  # to first dimension

                res[1:-1,:] = (1 / dlt ** 2) * (data[2:,:] + data[:-2,:] - 2 * data[1:-1,:])
                res[-1,:] = (1 / dlt ** 2) * (data[-1,:] + data[-3,:] - 2 * data[-2,:])
                res[0,:] = (1 / dlt ** 2) * (data[2,:] + data[0,:] - 2 * data[1,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1] = (1 / dlt ** 2) * (data[:,2:] + data[:,:-2] - 2 * data[:,1:-1])
                res[:,-1] = (1 / dlt ** 2) * (data[:,-1] + data[:,-3] - 2 * data[:,-2])
                res[:,0] = (1 / dlt ** 2) * (data[:,2] + data[:,0] - 2 * data[:,1])

            else:
                raise ValueError('wrong dim')
            
        else:
            raise ValueError('wrong order')

    elif l == 4:
        if order == 1:                    # first order derivatives
            
            if dim == 0:                  # to first dimension

                res[1:-1,:,:,:] = (1 / dlt) * (data[1:-1,:,:,:] - data[:-2,:,:,:])
                res[-1,:,:,:] = (1 / dlt) * (data[-1,:,:,:] - data[-2,:,:,:])
                res[0,:,:,:] = (1 / dlt) * (data[1,:,:,:] - data[0,:,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:,:] = (1 / dlt) * (data[:,1:-1,:,:] - data[:,:-2,:,:])
                res[:,-1,:,:] = (1 / dlt) * (data[:,-1,:,:] - data[:,-2,:,:])
                res[:,0,:,:] = (1 / dlt) * (data[:,1,:,:] - data[:,0,:,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1,:] = (1 /dlt) * (data[:,:,1:-1,:] - data[:,:,:-2,:])
                res[:,:,-1,:] = (1 / dlt) * (data[:,:,-1,:] - data[:,:,-2,:])
                res[:,:,0,:] = (1 / dlt) * (data[:,:,1,:] - data[:,:,0,:])
            
            elif dim == 3:

                res[:,:,:, 1:-1] = (1 / dlt) * (data[:,:,:,1:-1] - data[:,:,:,:-2])
                res[:,:,:,-1] = (1 / dlt) * (data[:,:,:,-1]- data[:, :, :, -2])
                res[:,:,:,0] = (1 / dlt) * (data[:,:,:,1] - data[:,:,:,0])
                

            else:
                raise ValueError('wrong dim')
                
        elif order == 2:
            
            if dim == 0:                  # to first dimension

                res[1:-1,:,:,:] = (1 / dlt ** 2) * (data[2:,:,:,:] + data[:-2,:,:,:] - 2 * data[1:-1,:,:,:])
                res[-1,:,:,:] = (1 / dlt ** 2) * (data[-1,:,:,:] + data[-3,:,:, :] - 2 * data[-2,:,:,:])
                res[0,:,:,:] = (1 / dlt ** 2) * (data[2,:,:,:] + data[0,:,:,:] - 2 * data[1,:,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:,:] = (1 / dlt ** 2) * (data[:,2:,:,:] + data[:,:-2,:,:] - 2 * data[:,1:-1,:,:])
                res[:,-1,:,:] = (1 / dlt ** 2) * (data[:,-1,:,:] + data[:,-3,:,:] - 2 * data[:,-2,:,:])
                res[:,0,:,:] = (1 / dlt ** 2) * (data[:,2,:,:] + data[:,0,:,:] - 2 * data[:,1,:,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1,:] = (1 / dlt ** 2) * (data[:,:,2:,:] + data[:,:,:-2,:] - 2 * data[:,:,1:-1,:])
                res[:,:,-1,:] = (1 / dlt ** 2) * (data[:,:,-1,:] + data[:,:,-3,:] - 2 * data[:,:,-2,:])
                res[:,:,0,:] = (1 / dlt ** 2) * (data[:,:,2,:] + data[:,:,0,:] - 2 * data[:,:,1,:])

            elif dim == 3:                # to third dimension

                res[:,:,:,1:-1] = (1 / dlt ** 2) * (data[:,:,:,2:] + data[:,:,:,:-2] - 2 * data[:,:,:,1:-1])
                res[:,:,:,-1] = (1 / dlt ** 2) * (data[:,:,:,-1] + data[:,:,:,-3] - 2 * data[:,:,:,-2])
                res[:,:,:,0] = (1 / dlt ** 2) * (data[:,:,:,2] + data[:,:,:,0] - 2 * data[:,:,:,1])
            else:
                raise ValueError('wrong dim')
            
        else:
            raise ValueError('wrong order')
    else:
        raise ValueError("Dimension NOT supported")
        
    if cap is not None:
        res[res < cap] = cap
    return res




def PDESolver(stateSpace, A, B_r, B_f, B_k, C_rr, C_ff, C_kk, D, v0, ε = 1, tol = -10, smartguess = False, solverType = 'False Transient'):

    if solverType == 'False Transient':
        A = A.reshape(-1,1,order = 'F')
        B = np.hstack([B_r.reshape(-1,1,order = 'F'),B_f.reshape(-1,1,order = 'F'),B_k.reshape(-1,1,order = 'F')])
        C = np.hstack([C_rr.reshape(-1,1,order = 'F'), C_ff.reshape(-1,1,order = 'F'), C_kk.reshape(-1,1,order = 'F')])
        D = D.reshape(-1,1,order = 'F')
        v0 = v0.reshape(-1,1,order = 'F')
        out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0, ε, tol)

        return out

    elif solverType == 'Feyman Kac':
        
        if smartguess:
            iters = 1
        else:
            iters = 400000
            
        A = A.reshape(-1, 1, order='F')
        B = np.hstack([B_r.reshape(-1, 1, order='F'), B_f.reshape(-1, 1, order='F'), B_k.reshape(-1, 1, order='F')])
        C = np.hstack([C_rr.reshape(-1, 1, order='F'), C_ff.reshape(-1, 1, order='F'), C_kk.reshape(-1, 1, order='F')])
        D = D.reshape(-1, 1, order='F')
        v0 = v0.reshape(-1, 1, order='F')
        out = SolveLinSys.solveFK(stateSpace, A, B, C, D, v0, iters)

        return out

def PDESolver_4D(stateSpace, A, B_1, B_2, B_3, B_4, C_11, C_22, C_33, C_44,  D, v0, ε = 1, tol = -10, smartguess = False, solverType = 'False Transient'):

    if solverType == 'False Transient':
        A = A.reshape(-1,1,order = 'F')
        B = np.hstack([B_1.reshape(-1,1,order = 'F'),B_2.reshape(-1,1,order = 'F'), B_3.reshape(-1,1,order = 'F'), B_4.reshape(-1,1,order='F')])
        C = np.hstack([C_11.reshape(-1,1,order = 'F'), C_22.reshape(-1,1,order = 'F'), C_33.reshape(-1,1,order = 'F'), C_44.reshape(-1,1,order='F')])
        D = D.reshape(-1,1,order = 'F')
        v0 = v0.reshape(-1,1,order = 'F')
        out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0, ε, tol)

        return out

    elif solverType == 'Feyman Kac':
        
        if smartguess:
            iters = 1
        else:
            iters = 400000
            
        A = A.reshape(-1, 1, order='F')
        B = np.hstack([B_1.reshape(-1, 1, order='F'), B_2.reshape(-1, 1, order='F'), B_3.reshape(-1, 1, order='F'), B_4.reshape(-1,1,order='F')])
        C = np.hstack([C_11.reshape(-1, 1, order='F'), C_22.reshape(-1, 1, order='F'), C_33.reshape(-1, 1, order='F'), C_44.reshape(-1,1,order='F')])
        D = D.reshape(-1, 1, order='F')
        v0 = v0.reshape(-1, 1, order='F')
        out = SolveLinSys.solveFK(stateSpace, A, B, C, D, v0, iters)

        return out
