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


file = open('Result.csv','r')
reader = csv.reader(file,delimiter=',')
file_header= next(reader)
file_varnum = len(file_header)
data = np.array(list(reader)).astype(float)
file_length = len(data[:,1])

figwidth = 10

fig, axs = plt.subplot_mosaic(
    [["left column", "right top"],
     ["left column", "right mid"],
     ["left column", "right down"]]
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