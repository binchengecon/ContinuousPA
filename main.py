from configparser import MAX_INTERPOLATION_DEPTH
import os
import sys

import petsc4py
#petsc4py.init(sys.argv)
from petsc4py import PETSc
import petsclinearsystem
from supportfunctions import *
import SolveLinSys

import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy import optimize
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

mpl.rc_file_defaults()

def util(c):
    return c**(1/2.)

def h(a):
    return a**2./2.+2.*a/5.

def gamma(a):
    return h(a)/a

def maxover_ac(W,dFdW,dFdWW):

    def function(variable):
        a,c= variable
        # print(a,c)

        temp = r*(a-c) + dFdW*r*(W-util(c)+h(a))+dFdWW*r**2*gamma(a)**2*sigma**2/2

        return -temp
    
    x0 = [0.1,0.]
    bound_a = (0,1)
    bound_c = (0,np.inf)
    result = optimize.minimize(function,x0,method='L-BFGS-B',bounds=(bound_a,bound_c))
    # result = optimize.minimize(function,x0)
    a,c = result.x
    return a,c


def finiteDiff(data, dim, order, dlt, cap = None):  
    # compute the central difference derivatives for given input and dimensions
    res = np.zeros(data.shape)
    l = len(data.shape)

    if l == 1:
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

def smooth_pasting(order):

    if order==0:

        temp = -W_mat[-1]**2
    
    elif order==1:

        temp= -2*W_mat[-1]

    return temp

def false_transient_one_iteration_cpp(stateSpace, A, B1, C1, D, v0, epsilon):
    A = A.reshape(-1, 1, order='F')
    B = np.hstack([B1.reshape(-1, 1, order='F'), ])
    C = np.hstack([C1.reshape(-1, 1, order='F'), ])
    D = D.reshape(-1, 1, order='F')
    out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0.reshape(-1, 1, order='F'), epsilon, -10)
    return out[2].reshape(v0.shape, order = "F")


r=0.1
sigma=1.

W_min = 0.
W_max = 1.
hW = 0.01

W = np.arange(W_min,W_max,hW)
nW = len(W)


W_mat, = np.meshgrid(W, indexing = 'ij')

StateSpace = np.hstack([W_mat.reshape(-1,1,order = 'F')])


error = 1.
count = 0

tol = 1e-7
max_iter = 10000
fraction = 0.1
epsilon=0.1

F0 = -W_mat**2


a = np.zeros_like(W_mat)    
c = np.zeros_like(W_mat)    


a_new = np.zeros_like(W_mat)    
c_new = np.zeros_like(W_mat)    


while error>tol and count <max_iter:
    F0[0]=0
    F0[-1]=smooth_pasting(0)
    dFdW = finiteDiff(F0,0,1,hW)
    dFdW[-1]=smooth_pasting(1)
    dFdWW = finiteDiff(F0,0,2,hW)

    for k in range(len(W_mat)):

        a_new[k],c_new[k]=maxover_ac(W[k],dFdW[k],dFdWW[k])


    a = a_new*fraction + a*(1-fraction)
    c = c_new*fraction + c*(1-fraction)

    A = np.ones_like(W_mat)*(-r)
    B_W = r*(W_mat-c**(1/2)+a**2/2+2*a/5)
    C_WW = r**2*sigma**2*(a/2+2/5)**2/2
    D = r*(a-c)


    F = false_transient_one_iteration_cpp(StateSpace, A, B_W, C_WW, D, F0, epsilon)

    rhs_error = A * F0 + B_W * dFdW +  C_WW * dFdWW  + D
    rhs_error = np.max(abs(rhs_error))
    lhs_error = np.max(abs((F - F0)/epsilon))

    error = lhs_error
    F0 = F
    count += 1

    # if print_iteration:
    print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

Data_List = list(zip(W_mat,F,a,c,B_W))

file_header = ['W','F','a','c','B_W']

with open('Result.csv','w+') as f:
    writer  = csv.writer(f)
    writer.writerow(file_header)
    writer.writerows(Data_List)

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 18}
        
plt.rc('font', **font)  # pass in the font dict as kwargs

figwidth = 10

fig, axs = plt.subplot_mosaic(
    [["left column", "right top"],
     ["left column", "right mid"],
     ["left column", "right down"]],figsize=(2  * figwidth, 2 *figwidth)
)

axs["left column"].plot(W_mat,F)
axs["left column"].set_title("Profit")
axs["left column"].grid(linestyle=':')

axs["right top"].plot(W_mat,a)
axs["right top"].set_title("Effort a(W)")
axs["right top"].grid(linestyle=':')


axs["right mid"].plot(W_mat,c)
axs["right mid"].set_title("Consumption c(W)")
axs["right mid"].grid(linestyle=':')

axs["right down"].plot(W_mat,B_W)
axs["right down"].set_title("Drift of W")
axs["right down"].grid(linestyle=':')

pdf_pages = PdfPages("Result.pdf")

pdf_pages.savefig(fig)
plt.close()
pdf_pages.close()  