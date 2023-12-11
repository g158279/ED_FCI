# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 09:23:20 2023

@author: Jianpeng Liu & Zhongqing Guo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import math
from mpi4py import MPI
import time
import sys
import os
import warnings
from numpy import (
    load, savez, savez_compressed,
    array, linspace, arange, zeros, ones, eye, full, identity, matrix, matmul,
    hstack, vstack, stack, concatenate, sort, argsort, where, all, any, expand_dims,
    tensordot, einsum, dot, vdot, inner, kron, cross,
    trace, transpose, conj, real, imag, diag, sum, prod, diagonal, fill_diagonal, roll,
    around, abs, angle, pi, sqrt, exp, log, sin, cos, tan, heaviside,
    min, max, nonzero,
)
from numpy.linalg import eigh, eigvalsh, det, inv, norm, eig, eigvals
from scipy.sparse.linalg import eigs,eigsh
# from scipy.sparse import coo_array
from matplotlib.pyplot import subplots, figure, plot, imshow, scatter
from matplotlib import cm
from numba import njit,objmode,types

comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()
# size=1
# rank=0
if rank==0:
    time3=time.time()

###### input parameters ######
Ne = int(sys.argv[1])
flux1 = 0.
flux2 = float(sys.argv[2])
# flux2_HF=int(flux2*3+0.001)
# theta0=0.77 #deg
theta0 = 0.77# 0.77 #deg
nD = 9# 11 # plane-wave cut-off
num_k0=4 # k0 of ED
num_k1=5 # k1 of ED
num_k0_HF=20 # k0 of HF
num_k1_HF=20 # k1 of HF
num_kpt_HF=num_k0_HF*num_k1_HF
# num_k1 = int(sys.argv[3]) #18 # k mesh
ncut = 3 #3 # band cut-off, for RG and HF+RPA 

# eps_r = 4 # dielectric constant
# Uinput = array([-0.08277066, -0.03852567, -0.00132553, 0.03699317, 0.08562869])# ksqr=0, no strain

# eps_r = 5 # dielectric constant
# Uinput = array([-0.09454625, -0.04466344, -0.00111442,  0.04338836,  0.09693575])# ksqr=0, no strain

# eps_r = 6 # dielectric constant
# Uinput = array([-0.1033046,  -0.04927666, -0.00096347,  0.04818289,  0.10536184])# ksqr=0, no strain

Dfield=float(sys.argv[3])
eps_r=float(sys.argv[4])
Uinput = np.load(f'plot/Uinput_Dfield{Dfield:.3f}_eps{eps_r:.1f}.npy')

# Dfield=0.097
# eps_r=5
# Uinput = np.load(f'Uinput_Dfield{Dfield:.3f}_eps{eps_r:.1f}.npy')

# eps_r = 7 # dielectric constant
# Uinput = array([-0.11006308, -0.05286125, -0.00084962,  0.0519026,   0.11187135])# ksqr=0, no strain
# for i in os.listdir(os.getcwd()+'/plot'):
#     if i.split('_')[0]==f'Dfield{Dfield:.3f}' and i.split('_')[-1].split('.')[-1]=='npy':
#         init=i.split('_')[-1].split('.')[0]

param_up_name=f"plot/Dfield{Dfield:.3f}_param_up_pGr-hBN_RG_nostrain_theta{theta0:.3f}_nk{num_k1_HF}_nD{nD}_ncut{ncut}_eps{eps_r:.1f}.npy"
param_dn_name=f"plot/Dfield{Dfield:.3f}_param_dn_pGr-hBN_RG_nostrain_theta{theta0:.3f}_nk{num_k1_HF}_nD{nD}_ncut{ncut}_eps{eps_r:.1f}.npy"
#param_up_name=f"plot/Dfield{Dfield:.3f}_param_up_pGr-hBN_RG_nostrain_theta{theta0:.3f}_nk{num_k1_HF}_nD{nD}_ncut{ncut}_eps{eps_r:.1f}_flux{flux2:.2f}_init130.npy"
#param_dn_name=f"plot/Dfield{Dfield:.3f}_param_dn_pGr-hBN_RG_nostrain_theta{theta0:.3f}_nk{num_k1_HF}_nD{nD}_ncut{ncut}_eps{eps_r:.1f}_flux{flux2:.2f}_init130.npy"
param_up=np.load(param_up_name)
param_dn=np.load(param_dn_name)

# indRG = 0
dc=1
indRG = 1#0 # RG
# Given filling #
#filling = 4 * 1.00000001 * 1/num_kpt * 7
filling = 1. # spinless formulation, filling=2 means conduction flat band is full filled for two valleys. 

N1=5
num_kpt=num_k0*num_k1
num_G=nD**2
nsub=2*N1
nbnd=num_G*nsub

### Useful notations ###

# Pauli matrices
sigma=zeros((4,2,2),complex)
sigma[0]=np.array([[1,0],[0,1]])
sigma[1]=np.array([[0,1],[1,0]])
sigma[2]=np.array([[0,-1j],[1j,0]])
sigma[3]=np.array([[1,0],[0,-1]])

# C3 C6 rotation
Rot3=array([[np.cos(2*pi/3),-np.sin(2*pi/3),0],
           [np.sin(2*pi/3),np.cos(2*pi/3),0],
           [0,0,1]])
Rot6=array([[np.cos(1*pi/3),-np.sin(1*pi/3),0],
           [np.sin(1*pi/3),np.cos(1*pi/3),0],
           [0,0,1]])

# theta rotation
theta=theta0/180*pi #rad
Rot=array([[np.cos(theta),-np.sin(theta),0],
               [np.sin(theta),np.cos(theta),0],
               [0,0,1]])
Rotm=array([[np.cos(-theta),-np.sin(-theta),0],
                [np.sin(-theta),np.cos(-theta),0],
                [0,0,1]])





### Latitice set-up ###

# Gr lattice
ag=2.46 #angstrom
ag1=array([ag,0,0],float) 
ag2=array([ag/2,ag*sqrt(3)/2,0],float)
ag3=array([0,0,1],float) # unit z
Sg0=(ag**2)*sqrt(3.0)/2
gg1=2*pi*cross(ag2,ag3)/Sg0
gg2=2*pi*cross(ag3,ag1)/Sg0
gg3=gg1+gg2

# hBN lattice
ab=2.46*55/54 #2.504 #angstrom
ab1=np.dot(Rot,array([ab,0,0],float)) 
ab2=np.dot(Rot,array([ab/2,ab*sqrt(3)/2,0],float))
ab3=array([0,0,1],float)
Sb0=(ab**2)*sqrt(3.0)/2
gb1=2*pi*cross(ab2,ab3)/Sb0
gb2=2*pi*cross(ab3,ab1)/Sb0
gb3=gb1+gb2

# Moire lattice
deltg1=gg1-gb1
deltg2=np.dot(Rot3,deltg1)
Sd0=(norm(deltg1)**2)*sqrt(3.0)/2
delt1=2*pi*cross(deltg2,ab3)/Sd0
delt2=2*pi*cross(ab3,deltg1)/Sd0
L1=delt1[0:2]
L2=delt2[0:2]
G1=deltg1[0:2]
G2=deltg2[0:2]
bvec=np.array([G1,G2])
Ls = norm(delt1)
Omega0=(Ls**2)*sqrt(3.0)/2.0

# Commensuration (for check)
numdg1=(delt1[0]-delt1[1]/sqrt(3))/ag
numdg2=delt1[1]*2/sqrt(3)/ag
d1=np.dot(Rotm,delt1)
d2=np.dot(Rotm,delt2)
numdb1=(d1[0]-d1[1]/sqrt(3))/ab
numdb2=d1[1]*2/sqrt(3)/ab
gf1=round(numdg1)*ag1+round(numdg2)*ag2
bf1=round(numdb1)*ab1+round(numdb2)*ab2
Ls_com=norm(gf1)


# Constants
muB=5.7883818012*10**(-5)
d0=3.35
Qz=2*pi/(2*d0)
vF0 = 2.1354*2.46#5.253084
gamma0 = 0.34
vperp0 = vF0*0.051/0.80


### Moire potential: Ud-induced change is neglected for now ###

# Decomposed version
V0r=0.0289
V1r=0.0210
phir=-0.29
Veff=-1/2*V1r*np.array([[np.exp(1j*(phir+2*pi/3)),0],[0,np.exp(1j*(phir+2*pi/3))]])
Meff=-1j*V1r*sqrt(3)/2*np.array([[np.exp(1j*(phir+2*pi/3)),0],[0,-np.exp(1j*(phir+2*pi/3))]])
Aeff1=V1r*np.array([cos(4*pi/3),sin(4*pi/3)])*np.exp(1j*(phir+2*pi/3))
Aeff2=V1r*np.array([cos(6*pi/3),sin(6*pi/3)])*np.exp(1j*(phir+2*pi/3))
Aeff3=V1r*np.array([cos(8*pi/3),sin(8*pi/3)])*np.exp(1j*(phir+2*pi/3))

# RG

alpha = 2.74/eps_r # alpha_0 = e^2 / (4*pi*epsilon_0 * hbar * vF0) all in SI units

# Ec/Ec' = Gvec_gr / (G_moire*ncut) = Ls/(ag*ncut)

if indRG == 1:
    vF =  vF0 * (1 + alpha/4 * np.log(Ls/(ag*ncut) ))
    Aeff1 = Aeff1 * (1 + alpha/4 * np.log(Ls/(ag*ncut) ))
    Aeff2 = Aeff2 * (1 + alpha/4 * np.log(Ls/(ag*ncut) ))
    Aeff3 = Aeff3 * (1 + alpha/4 * np.log(Ls/(ag*ncut) ))
    Meff = Meff * (1 + alpha/4 * np.log(Ls/(ag*ncut) ))**2
    gamma = gamma0 * (1 + alpha/4 * np.log(Ls/(ag*ncut) ))
    vperp = vperp0
else:
    vF = vF0
    gamma = gamma0
    vperp = vperp0

Uk=zeros((4,2,2),complex)
Ukp=zeros((4,2,2),complex)
Uk[0]=V0r*sigma[0]
Uk[1]=Veff+Meff+Aeff1[0]*sigma[1]+Aeff1[1]*sigma[2]
Uk[2]=Veff+Meff+Aeff2[0]*sigma[1]+Aeff2[1]*sigma[2]
Uk[3]=Veff+Meff+Aeff3[0]*sigma[1]+Aeff3[1]*sigma[2]
Ukp[0]=V0r*sigma[0]
Ukp[1]=conj(Veff+Meff+Aeff1[0]*(sigma[1])+Aeff1[1]*sigma[2])
Ukp[2]=conj(Veff+Meff+Aeff2[0]*(sigma[1])+Aeff2[1]*sigma[2])
Ukp[3]=conj(Veff+Meff+Aeff3[0]*(sigma[1])+Aeff3[1]*sigma[2])



### Hamiltonian ###

# G vectors and its index
Gvec=zeros((num_G,2),float)
for iG0 in range(0,nD):
    for iG1 in range(0,nD):
        Gvec[iG0*nD+iG1,0:2]=(iG0-(nD-1)/2)*bvec[0,0:2]+(iG1-(nD-1)/2)*bvec[1,0:2]

Gm=4*pi/(sqrt(3)*Ls)



def make_hamk(kpt,Uonsite):
    hamk = zeros((nsub*num_G,nsub*num_G),complex)   # Gvec, sublattice, layer
    
    for iG in range(0,num_G):
        i0 = iG//nD
        i1 = iG-i0*nD
        kx=kpt[0]
        ky=kpt[1]
        Ut=np.array([[vperp*((kpt[0]+Gvec[iG,0])+1j*(kpt[1]+Gvec[iG,1])),gamma],[vperp*((kpt[0]+Gvec[iG,0])-1j*(kpt[1]+Gvec[iG,1])),vperp*((kpt[0]+Gvec[iG,0])+1j*(kpt[1]+Gvec[iG,1]))]])               
        for l1 in range(0,N1):
            # set center gr sheet as the reference for electrostatic potential 
            hamk[nsub*iG+l1*2:nsub*iG+2*l1+2,nsub*iG+l1*2:nsub*iG+2*l1+2]+= -vF*(kx+Gvec[iG,0])*(sigma[1])-vF*(ky+Gvec[iG,1])*sigma[2]+Uonsite[l1]*sigma[0]
        if N1>1:
            for l1 in range(0,N1-1):
                hamk[nsub*iG+2*(l1+1):nsub*iG+2*(l1+1)+2,nsub*iG+2*l1:nsub*iG+2*l1+2]+=Ut
                hamk[nsub*iG+2*l1:nsub*iG+2*l1+2,nsub*iG+2*(l1+1):nsub*iG+2*(l1+1)+2]+=conj(transpose(Ut))
        
        # layer 0
        jG0=iG
        hamk[nsub*jG0:nsub*jG0+2,nsub*iG:nsub*iG+2]+=Uk[0]
        if i0<(nD-1):
            jG1 = (i0+1)*nD+i1
            hamk[nsub*jG1:nsub*jG1+2,nsub*iG:nsub*iG+2]+=Uk[1]    #jG1>>iG
            hamk[nsub*iG:nsub*iG+2,nsub*jG1:nsub*jG1+2]+=conj(transpose(Uk[1]))
        if i1<(nD-1):   #G1
            jG1 = i0*nD + i1 + 1
            #twist
            hamk[nsub*jG1:nsub*jG1+2,nsub*iG:nsub*iG+2]+=Uk[2]     #jG1>>iG
            hamk[nsub*iG:nsub*iG+2,nsub*jG1:nsub*jG1+2]+=conj(transpose(Uk[2]))

        if i0>0 and i1>0:        #G3=G1+G2
            jG2=(i0-1)*nD+i1 -1
            #twist
            hamk[nsub*jG2:nsub*jG2+2,nsub*iG:nsub*iG+2]+=Uk[3]      #jG2>>iG
            hamk[nsub*iG:nsub*iG+2,nsub*jG2:nsub*jG2+2]+=conj(transpose(Uk[3]))
        iG=jG0
    return hamk


def make_hamkP(kpt,Uonsite):
    hamk = zeros((nsub*num_G,nsub*num_G),complex)   
    for iG in range(0,num_G):
        i0 = iG//nD
        i1 = iG-i0*nD
        kx=kpt[0]
        ky=kpt[1]
        Ut=np.array([[vperp*(-(kpt[0]+Gvec[iG,0])+1j*(kpt[1]+Gvec[iG,1])),gamma],[vperp*(-(kpt[0]+Gvec[iG,0])-1j*(kpt[1]+Gvec[iG,1])),vperp*(-(kpt[0]+Gvec[iG,0])+1j*(kpt[1]+Gvec[iG,1]))]])      
        for l1 in range(0,N1):        
            hamk[nsub*iG+l1*2:nsub*iG+2*l1+2,nsub*iG+l1*2:nsub*iG+2*l1+2]+= -vF*(kx+Gvec[iG,0])*(-sigma[1])-vF*(ky+Gvec[iG,1])*sigma[2]+ Uonsite[l1]*sigma[0]
        if N1>1:
            for l1 in range(0,N1-1):
                hamk[nsub*iG+2*(l1+1):nsub*iG+2*(l1+1)+2,nsub*iG+2*l1:nsub*iG+2*l1+2]+=Ut
                hamk[nsub*iG+2*l1:nsub*iG+2*l1+2,nsub*iG+2*(l1+1):nsub*iG+2*(l1+1)+2]+=conj(transpose(Ut))
        
        # layer 0
        jG0=iG
        hamk[nsub*jG0:nsub*jG0+2,nsub*iG:nsub*iG+2]+=Ukp[0]
        if i0>0:
            jG1 = (i0-1)*nD+i1
            hamk[nsub*jG1:nsub*jG1+2,nsub*iG:nsub*iG+2]+=Ukp[1]     #jG1>>iG
            hamk[nsub*iG:nsub*iG+2,nsub*jG1:nsub*jG1+2]+=conj(transpose(Ukp[1]))
        if i1>0:   #G1
            jG1 = i0*nD + i1 - 1
            #twist
            hamk[nsub*jG1:nsub*jG1+2,nsub*iG:nsub*iG+2]+=Ukp[2]     #jG1>>iG
            hamk[nsub*iG:nsub*iG+2,nsub*jG1:nsub*jG1+2]+=conj(transpose(Ukp[2]))

        if i0<(nD-1) and i1<(nD-1):        #G6=G1+G2
            jG2=(i0+1)*nD+i1 +1
            #twist
            hamk[nsub*jG2:nsub*jG2+2,nsub*iG:nsub*iG+2]+=Ukp[3]      #jG2>>iG
            hamk[nsub*iG:nsub*iG+2,nsub*jG2:nsub*jG2+2]+=conj(transpose(Ukp[3]))
    return hamk

nQ0=nD
nQ1=nD
Qvec=Gvec            

indexGq=zeros((num_G,nQ0*nQ1),int)
indexGqm=zeros((num_G,nQ0*nQ1),int)

@njit
def find_GQindex():
    indexGq=zeros((num_G,nQ0*nQ1),'int')
    indexGqm=zeros((num_G,nQ0*nQ1),'int')
    for iQ in range(0,nQ0*nQ1):
        for iG in range(0,num_G):
            indexGq[iG,iQ]=1000
            indexGqm[iG,iQ]=1000
            GQvec=Gvec[iG,:]+Qvec[iQ,:]
            GQvecm=Gvec[iG,:]-Qvec[iQ,:]
            for iGp in range(0,num_G):
                dGQ=GQvec-Gvec[iGp,:]
                if (dGQ[0]**2+dGQ[1]**2)<0.00000001:
                    indexGq[iG,iQ]=iGp
                    
                dGQm=GQvecm-Gvec[iGp,:]
                if (dGQm[0]**2+dGQm[1]**2)<0.00000001:
                    indexGqm[iG,iQ]=iGp 
    
    return indexGq, indexGqm
                
indexGq, indexGqm = find_GQindex()

ncenter = int(num_G*nsub/2)
ntarget = 2*ncut
nbmin = ncenter - ncut 
nbmax = ncenter + ncut

G1vec=bvec[0,0:2]
G2vec=bvec[1,0:2]

kvec = zeros((num_kpt,2),float)
qvec = zeros((num_kpt,2),float)
qvecindex=zeros((num_kpt,2),int)
kvecindex=zeros((num_kpt,2),int)
for ik0 in range(0,num_k0):
    for ik1 in range(0,num_k1):
        kvec[ik0*num_k1+ik1,0:2]=((ik0+flux1)/num_k0)*G1vec + ((ik1+flux2)/num_k1)*G2vec
        qvec[ik0*num_k1+ik1,0:2]=((ik0)/num_k0)*G1vec + ((ik1)/num_k1)*G2vec
        qvecindex[ik0*num_k1+ik1,0]=ik0#((ik0-num_k0C)/num_k0)
        qvecindex[ik0*num_k1+ik1,1]=ik1
        kvecindex[ik0*num_k1+ik1,0]=ik0#+flux1#((ik0-num_k0C)/num_k0)
        kvecindex[ik0*num_k1+ik1,1]=ik1#+flux2

@njit
def kqmap():
    kpq_index=zeros((num_kpt,num_kpt,3),'int')
    kmq_index=zeros((num_kpt,num_kpt,3),'int')
    for ik in range(0,num_kpt):
        for iq in range(0,num_kpt):
            for ikp in range(0,num_kpt):
                diffkq0=(kvecindex[ik,0]+qvecindex[iq,0]-kvecindex[ikp,0])%num_k0
                diffkq1=(kvecindex[ik,1]+qvecindex[iq,1]-kvecindex[ikp,1])%num_k1
                if (diffkq0**2+diffkq1**2)<0.000000001:
                    kpq_index[ik,iq,0]=ikp
                    kpq_index[ik,iq,1]=(kvecindex[ik,0]+qvecindex[iq,0]-kvecindex[ikp,0])//num_k0
                    kpq_index[ik,iq,2]=(kvecindex[ik,1]+qvecindex[iq,1]-kvecindex[ikp,1])//num_k1

                diffkqm0=(kvecindex[ik,0]-qvecindex[iq,0]-kvecindex[ikp,0])%num_k0
                diffkqm1=(kvecindex[ik,1]-qvecindex[iq,1]-kvecindex[ikp,1])%num_k1
                if (diffkqm0**2+diffkqm1**2)<0.000000001:
                    kmq_index[ik,iq,0]=ikp
                    kmq_index[ik,iq,1]=(kvecindex[ik,0]-qvecindex[iq,0]-kvecindex[ikp,0])//num_k0
                    kmq_index[ik,iq,2]=(kvecindex[ik,1]-qvecindex[iq,1]-kvecindex[ikp,1])//num_k1

    return kpq_index, kmq_index

kpq_index, kmq_index = kqmap()        

if rank==0:
    time1=time.time()
    print('We start solving the no-double-counting HF wavefunctions for a single band')

# generate the non-interacting wavefunctions in planewave basis
psi0=zeros((nsub*num_G,ntarget,num_kpt),'complex') # non-interacting wavefunction for K valley
psi0p=zeros((nsub*num_G,ntarget,num_kpt),'complex') # non-interacting wavefunction for K' valley
Ek0=zeros((ntarget,num_kpt),'float') ## flat bands for K valley
Ek0p=zeros((ntarget,num_kpt),'float') ## flat bands for K' valley
for ik in range(0,num_kpt):
    if rank==0:
        print(f'{ik+1}/{num_kpt}')
    kpt=kvec[ik,0:2]
    hamk0t=make_hamk(kpt,Uinput)
    hamkP0t=make_hamkP(kpt,Uinput)
    w0,us0=eigh(hamk0t)
    wP0,usP0=eigh(hamkP0t)
    psi0[:,:,ik]=us0[:,nbmin:nbmax]
    psi0p[:,:,ik]=usP0[:,nbmin:nbmax]     
    Ek0[:,ik]=w0[nbmin:nbmax]
    Ek0p[:,ik]=wP0[nbmin:nbmax]

def EV_HF2ED(Ek0,Ek0p,psi0,psi0p,param_up,param_dn,dc):
    param=zeros((2*2*ntarget,2*2*ntarget,num_kpt),'complex')  
    param_ED=zeros((2*2*ntarget,2*2*ntarget,num_kpt),'complex')  

    param[0:2*ntarget,0:2*ntarget]=param_up
    param[2*ntarget:4*ntarget,2*ntarget:4*ntarget]=param_dn
    # # generate wavefunctions in band basis
    # for ik in range(0,num_kpt):
    #     hamkband_up=zeros((2*ntarget,2*ntarget),'complex')
    #     hamkband_dn=zeros((2*ntarget,2*ntarget),'complex')
    #     for n in range(0,ntarget):
    #         hamkband_up[n,n]+=Ek0[n,ik]
    #         hamkband_dn[n,n]+=Ek0[n,ik]
    #     for n in range(ntarget,2*ntarget):
    #         hamkband_up[n,n]+=Ek0p[n-ntarget,ik]
    #         hamkband_dn[n,n]+=Ek0p[n-ntarget,ik]
        
    #     hamkband=zeros((2*2*ntarget,2*2*ntarget),complex)
    #     hamkband_up+=param_up[:,:,ik]
    #     hamkband_dn+=param_dn[:,:,ik]
    #     hamkband[0:2*ntarget,0:2*ntarget]=hamkband_up[:,:]
    #     hamkband[2*ntarget:4*ntarget,2*ntarget:4*ntarget]=hamkband_dn[:,:]
    #     _, Ub = eigh(hamkband)
    #     param_ED_no_double_counting=(Ub.conj().T@param[...,ik]@Ub) # transfom into interacting band basis
    #     if dc==0:
    #         param_ED_no_double_counting[2*ntarget,2*ntarget]=0. # eliminate the target intraband HF interaction
    #     param_ED[...,ik]=Ub@param_ED_no_double_counting@Ub.conj().T
    param_ED=param
    # generate wavefunctions in band basis after eliminating the target intraband HF interaction
    E_ED=zeros(num_kpt,'float')   # kinetic energy of target band
    V_ED=zeros((nbnd,num_kpt),'complex')  
    C0_k=zeros((nsub*num_G,2*2*ntarget,num_kpt),'complex') # non-interacting wavefunction including both K and K' valleys
    C0_k[:,0*ntarget:1*ntarget,:]=psi0
    C0_k[:,1*ntarget:2*ntarget,:]=psi0p
    C0_k[:,2*ntarget:3*ntarget,:]=psi0
    C0_k[:,3*ntarget:4*ntarget,:]=psi0p
    for ik in range(0,num_kpt):
        hamkband_up_ED=zeros((2*ntarget,2*ntarget),'complex')
        hamkband_dn_ED=zeros((2*ntarget,2*ntarget),'complex')
        for n in range(0,ntarget):
            hamkband_up_ED[n,n]+=Ek0[n,ik]
            hamkband_dn_ED[n,n]+=Ek0[n,ik]
        for n in range(ntarget,2*ntarget):
            hamkband_up_ED[n,n]+=Ek0p[n-ntarget,ik]
            hamkband_dn_ED[n,n]+=Ek0p[n-ntarget,ik]
        
        hamkband_ED=zeros((2*2*ntarget,2*2*ntarget),'complex')
        hamkband_ED[0:2*ntarget,0:2*ntarget]=hamkband_up_ED[:,:]
        hamkband_ED[2*ntarget:4*ntarget,2*ntarget:4*ntarget]=hamkband_dn_ED[:,:]
        hamkband_ED+=param_ED[:,:,ik]
        wb_ED,ub_ED = eigh(hamkband_ED)
        E_ED[ik]=wb_ED[2*ntarget]
        for iG in range(nbnd):
            for n0 in range(2*2*ntarget):
                V_ED[iG,ik]+=ub_ED[n0,2*ntarget]*C0_k[iG,n0,ik]
    return E_ED,V_ED,param_ED


param_up=param_up.reshape((2*ntarget,2*ntarget,num_k0_HF,num_k1_HF))[:,:,::(num_k0_HF//num_k0),::(num_k1_HF//num_k1)]
param_dn=param_dn.reshape((2*ntarget,2*ntarget,num_k0_HF,num_k1_HF))[:,:,::(num_k0_HF//num_k0),::(num_k1_HF//num_k1)]
param_up=param_up.reshape((2*ntarget,2*ntarget,num_k0*num_k1))
param_dn=param_dn.reshape((2*ntarget,2*ntarget,num_k0*num_k1))
# param_up=np.ascontiguousarray(transpose(param_up,[2,0,1]))
# param_dn=np.ascontiguousarray(transpose(param_dn,[2,0,1]))
# single band in planewave basis from no-double-counting HF
Ek0,psi0,param_ED=EV_HF2ED(Ek0,Ek0p,psi0,psi0p,param_up,param_dn,dc)

psi0Q=zeros((nbnd,num_kpt,nD**2),'complex')  ## non-interacting wavefunction for K valley, with G replaced by G+Q
psi0Qm=zeros((nbnd,num_kpt,nD**2),'complex')  ## non-interacting wavefunction for K valley, with G replaced by G-Q
for iQ in range(0,nD**2):
    for iG in range(0,num_G):
        for ia in range(0,nsub):
            iGq=indexGq[iG,iQ]
            iGqm=indexGqm[iG,iQ]
            if (iGq<500):
                psi0Q[iG*nsub+ia,:,iQ]=psi0[iGq*nsub+ia,:]
            if (iGqm<500):
                psi0Qm[iG*nsub+ia,:,iQ]=psi0[iGqm*nsub+ia,:]

#print(kpq_index[:,:,0])
#print(kmq_index[:,:,0])   
if rank==0:
    print(f'Maximum of Ek0: {max(Ek0[:])}')
    print(f'Minimum of Ek0: {min(Ek0[:])}')
        
@njit
def kqmap_wavefunction():
    psi0_q=zeros((nbnd,num_kpt,num_kpt),'complex')   # wavefunction at k+q for K valley
    psi0_qm=zeros((nbnd,num_kpt,num_kpt),'complex')  # wavefunction at k-q for K valley
    for ik in range(0,num_kpt):
        for iq in range(0,num_kpt):
            for iG in range(0,num_G):
                GQvec=Gvec[iG,:]+kpq_index[ik,iq,1]*G1vec+kpq_index[ik,iq,2]*G2vec
                iGq=1000
                for iGp in range(0,num_G):
                    if ((GQvec[0]-Gvec[iGp,0])**2+(GQvec[1]-Gvec[iGp,1])**2)<0.00000001:
                        iGq=iGp
                if iGq<900:
                    for ia in range(0,nsub):
                        psi0_q[iG*nsub+ia,ik,iq]=psi0[iGq*nsub+ia,kpq_index[ik,iq,0]]

                GQmvec=Gvec[iG,:]+kmq_index[ik,iq,1]*G1vec+kmq_index[ik,iq,2]*G2vec
                iGqm=1000
                for iGp in range(0,num_G):
                    if ((GQmvec[0]-Gvec[iGp,0])**2+(GQmvec[1]-Gvec[iGp,1])**2)<0.00000001:
                        iGqm=iGp
                if iGqm<900:
                    for ia in range(0,nsub):
                        psi0_qm[iG*nsub+ia,ik,iq]=psi0[iGqm*nsub+ia,kmq_index[ik,iq,0]]                         

    return psi0_q, psi0_qm   

psi0_q, psi0_qm=kqmap_wavefunction()


if rank==0:
    time2=time.time()
    print('Non-interacting wavefunctions solving time elapse is', time2-time1) 
    print('-----------------------------------------------------------------------------')


Uvalue = 14.4*2*pi/Omega0 # e^2/(4*pi*eps0) * 2*pi / Omega0
kappa = 1/400
ds = 400 # A

@njit
def Coulomb_U_layer(norm_q,l1,l2): 
    if norm_q > 1e-7 * 2.*pi/Ls / num_k1 :
        if l1 == l2 :
            return 1./sqrt(norm_q**2 + kappa**2)   
        else:
            return 1./norm_q * np.exp(-norm_q*d0*np.absolute(l1-l2))   
    else:
        return 0.


@njit
def Coulomb_proj(sitei,sitej,iq):
    psi0_q_conj_i=conj(psi0_q[:,sitei,iq])
    psi0Qm_i=np.ascontiguousarray(psi0Qm[:,sitei,:].T)
    psi0_qm_conj_j=conj(psi0_qm[:,sitej,iq])
    psi0Q_j=np.ascontiguousarray(psi0Q[:,sitej,:].T)
    Veff=.0j
    for iQ in range(num_G):
        qvec_norm=norm(qvec[iq,0:2]+Qvec[iQ,0:2])
        Veff_iQ=.0j
        Veff_iQ_i=zeros(N1,'complex')
        Veff_iQ_j=zeros(N1,'complex')
        for l1 in range(N1):
            for iG in range(num_G):
                for n in range(2):
                    Veff_iQ_i[l1]+=psi0_q_conj_i[iG*nsub+l1*2+n]*psi0Qm_i[iQ,iG*nsub+l1*2+n]
                    Veff_iQ_j[l1]+=psi0_qm_conj_j[iG*nsub+l1*2+n]*psi0Q_j[iQ,iG*nsub+l1*2+n]
        if qvec_norm>1e-12:
            for l1 in range(N1):
                for l2 in range(N1):
                    Veff_iQ+=(Veff_iQ_i[l1]*Veff_iQ_j[l2])*Coulomb_U_layer(qvec_norm,l1,l2)
        else:
            Veff_iQ+=0.
        Veff+=Veff_iQ
    return Veff*Uvalue/eps_r

@njit
def Coulomb_projm(ik,ikp,iq):
    psi0_qm_conj_k=conj(psi0_qm[:,ik,iq])
    psi0Q_k=np.ascontiguousarray(psi0Q[:,ik,:].T)
    psi0Qm_kp=np.ascontiguousarray(psi0Qm[:,ikp,:].T)
    psi0_q_conj_kp=conj(psi0_q[:,ikp,iq])
    Veffm=.0j
    for iQ in range(num_G):
        qvec_norm=norm(-qvec[iq,0:2]-Qvec[iQ,0:2])
        Veffm_iQ=.0j
        Veffm_iQ_k=zeros(N1,'complex')
        Veffm_iQ_kp=zeros(N1,'complex')
        for l1 in range(N1):
            for iG in range(num_G):
                for n in range(2):
                    Veffm_iQ_k[l1]+=psi0_qm_conj_k[iG*nsub+l1*2+n]*psi0Q_k[iQ,iG*nsub+l1*2+n]
                    Veffm_iQ_kp[l1]+=psi0_q_conj_kp[iG*nsub+l1*2+n]*psi0Qm_kp[iQ,iG*nsub+l1*2+n]
        if qvec_norm>1e-12:
            for l1 in range(N1):
                for l2 in range(N1):
                    Veffm_iQ+=(Veffm_iQ_k[l1]*Veffm_iQ_kp[l2])*Coulomb_U_layer(qvec_norm,l1,l2)
        else:
            Veffm_iQ+=0.
        Veffm+=Veffm_iQ
    return Veffm*Uvalue/eps_r

#@njit
#def Coulomb_proj(sitei,sitej,iq):
#    psi0_q_conj_i=conj(psi0_q[:,sitei,iq])
#    psi0Qm_i=np.ascontiguousarray(psi0Qm[:,sitei,:].T)
#    psi0_qm_conj_j=conj(psi0_qm[:,sitej,iq])
#    psi0Q_j=np.ascontiguousarray(psi0Q[:,sitej,:].T)
#    Veff=.0j
#    for iQ in range(num_G):
#        qvec_norm=norm(qvec[iq,0:2]+Qvec[iQ,0:2])
#        Veff_iQ=.0j
#        Veff_iQ_i=np.dot(psi0_q_conj_i,psi0Qm_i[iQ])
#        Veff_iQ_j=np.dot(psi0_qm_conj_j,psi0Q_j[iQ])
#        if qvec_norm>1e-7:
#            #Veff_iQ+=Veff_iQ_i*Veff_iQ_j*np.tanh(qvec_norm*ds)/qvec_norm
#            Veff_iQ=(Veff_iQ_i*Veff_iQ_j)/qvec_norm
#        else:
#            Veff_iQ=Veff_iQ_i*Veff_iQ_j*0.0#ds#0.0
#        Veff+=Veff_iQ
#    return Veff*Uvalue/eps_r
#
#@njit
#def Coulomb_projm(ik,ikp,iq):
#    psi0_qm_conj_k=conj(psi0_qm[:,ik,iq])
#    psi0Q_k=np.ascontiguousarray(psi0Q[:,ik,:].T)
#    psi0Qm_kp=np.ascontiguousarray(psi0Qm[:,ikp,:].T)
#    psi0_q_conj_kp=conj(psi0_q[:,ikp,iq])
#    Veffm=.0j
#    for iQ in range(num_G):
#        qvec_norm=norm(-qvec[iq,0:2]-Qvec[iQ,0:2])
#        Veffm_iQ=.0j
#        Veffm_iQ_k=np.dot(psi0_qm_conj_k,psi0Q_k[iQ])
#        Veffm_iQ_kp=np.dot(psi0_q_conj_kp,psi0Qm_kp[iQ])
#        if qvec_norm>1e-7:
#            Veffm_iQ=(Veffm_iQ_k*Veffm_iQ_kp)/qvec_norm#*np.tanh(qvec_norm*ds)/qvec_norm
#        else:
#            Veffm_iQ=Veffm_iQ_k*Veffm_iQ_kp*0.0#*0.0#ds
#        Veffm+=Veffm_iQ
#    return Veffm*Uvalue/eps_r

@njit
def get_Coulomb(i):
    Coulomb=zeros((num_kpt,num_kpt),'complex')
    Coulomb_m=zeros((num_kpt,num_kpt),'complex')
    for j in range(num_kpt):
        for k in range(num_kpt):
            Coulomb[j,k]=Coulomb_proj(i,j,k)
            Coulomb_m[j,k]=Coulomb_projm(i,j,k)
    return Coulomb,Coulomb_m

def get_Coulomb_mpi():
    ########################### MPI ###########################
    # If the size of array to be parallelized *can not be divided* by the number of cores,
    # the array will be diveded into subsets with 2 types of size:
    # {num_more} subsets have {subset_size+1} elements, lefted are the subsets with {subset_size} elements
    subset_size,num_more=divmod(num_kpt,size)
    k_subsets=[range(num_kpt)[i*(subset_size+1):(i+1)*(subset_size+1)] if i < num_more else range(num_kpt)[i*subset_size+num_more:(i+1)*subset_size+num_more] for i in range(size)] # divide kvec into size subsets
    k_subset=comm.scatter(k_subsets,root=0)
    ###########################################################

    Coulomb=zeros((len(k_subset),num_kpt,num_kpt),'complex')
    Coulomb_m=zeros((len(k_subset),num_kpt,num_kpt),'complex')
    for n,i in enumerate(k_subset):
        if i==[]:
            pass
        else:
            Coulomb[n],Coulomb_m[n]=get_Coulomb(i)

    ########################### MPI ###########################
    Coulomb_gather=comm.gather(Coulomb,root=0)
    Coulomb_m_gather=comm.gather(Coulomb_m,root=0)
    if rank==0:
        Coulomb=concatenate(Coulomb_gather)
        Coulomb_m=concatenate(Coulomb_m_gather)
    else:
        Coulomb=None
        Coulomb_m=None
    Coulomb=comm.bcast(Coulomb,root=0)
    Coulomb_m=comm.bcast(Coulomb_m,root=0)
    ###########################################################
    return Coulomb,Coulomb_m

# @njit
# def get_Coulomb():
#     Coulomb=zeros((num_kpt,num_kpt,num_kpt),'complex')
#     Coulomb_m=zeros((num_kpt,num_kpt,num_kpt),'complex')
#     for i in range(num_kpt):
#         for j in range(num_kpt):
#             for k in range(num_kpt):
#                 Coulomb[i,j,k]=Coulomb_proj(i,j,k)
#                 Coulomb_m[i,j,k]=Coulomb_projm(i,j,k)
#     return Coulomb,Coulomb_m


time1=time.time()

Nsite=num_kpt
Nstate=math.comb(Nsite,Ne)
if rank==0:
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('The dimension of Hilbert space is ', Nstate)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

@njit
def bitfield(n,Nsite):
    bits=zeros(Nsite,np.int8)
    for i in range(Nsite-1,-1,-1):
        bit=n//(2**i)
        bits[i]=bit
        if bit:
            n-=2**i
    return bits[::-1]

@njit
def invert_bit(nb,i,j):
    nt=nb
    nia=np.bitwise_xor(nt[i],1)
    nja=np.bitwise_xor(nt[j],1)
    nt[i]=nia
    nt[j]=nja
    return nt

@njit
def get_Hilbert(Ne,Nsite,Nstate):
    Nhilbert=np.zeros((Nstate,Nsite),np.int8)  #all state
    Nfilled=np.zeros((Nstate,Ne),'int') # index occup
    Ihilbert=np.zeros(2**Nsite,'int')
    ind=0
    for i in range(2**Nsite):
        if i%1000000==0 and rank==0:
            print(f'{i}/{2**Nsite-1}')
        # Iindex.append(i)
        ib=bitfield(i,Nsite)
        Ihilbert[i]=-2
        if sum(ib)==Ne:
            Ihilbert[i]=ind
            Nhilbert[ind,:]=ib   
            Nfilled[ind,:]=np.where(ib==1)[0]
            # Nempty[ind,:]=np.where(ib==0)[0]
            ind+=1

    return Nhilbert,Nfilled,Ihilbert
    # return Nhilbert,Nfilled#,Nempty,Iindex,Ihilbert
if rank==0:
    print('Initializing Hilbert space')
    time1=time.time()
Nhilbert,Nfilled,Ihilbert=get_Hilbert(Ne,Nsite,Nstate)
# print(Nhilbert)
# print(Nfilled)
# print(Ihilbert)
if rank==0:
    time2=time.time()
    print('Hilbert space initialization time elapse is', time2-time1) 
    print('-----------------------------------------------------------------------------')

@njit
def bits2int(bits): # used to replace "int(''.join(str(xs) for xs in invsiteback),2)" due to @njit
    int=0
    bits_inv=bits[::-1]
    for xs in range(bits.shape[0]):
        if bits_inv[xs]==1:
            int+=2**xs
    return int

@njit
def Hamiltonian_momentum(ik,Ne,Coulomb,Coulomb_m):
            
    num_momentum_ik=0
    Nhilbert_ik=[]
    for n in range(0,Nstate):
        ikt0=0
        ikt1=0
        for ie in range(0,Ne):
            isite0=Nfilled[n,ie]//num_k1
            isite1=Nfilled[n,ie]-isite0*num_k1
            ikt0+=isite0
            ikt1+=isite1
        ik0=ikt0%num_k0
        ik1=ikt1%num_k1
        total_momentum=ik0*num_k1+ik1
        #total_momentum=sum(Nfilled[n,:])
        if total_momentum==ik:
            num_momentum_ik+=1
            Nhilbert_ik.append(n)
    Nhilbert_ik=array(Nhilbert_ik)
    H=[]
    rowindex=[]
    columnindex=[]
    
    
    for nk in range(0,num_momentum_ik):   ## loop over the Hilbert space indices within ik block
        if nk%10000==0 and rank==0:
            print(f'{nk}/{num_momentum_ik}')
        n=Nhilbert_ik[nk]                 # n is the original basis state index in ik block
        for i in range(Ne):   
            #print(nk)
            #print(Ek0[Nfilled[n,i]])
            # H.append(1.*Ek0[Nfilled[n,i]])  #Hamiltonian element in ik block
            H.append(-1.*Ek0[Nfilled[n,i]])  #Hamiltonian element in ik block
            rowindex.append(nk)              # row, column indices within ik block  
            columnindex.append(nk)
            for j in range(i+1,Ne):             
                for iq in range(0, Nsite): 
            
                    sitei=Nfilled[n,i]          # sitei, sitej point in state n in ik block   
                    sitej=Nfilled[n,j]
                    nbite=Nhilbert[n,:].copy()
                    nbite0=Nhilbert[n,:].copy()
                    invsite=nbite
                    nia=np.bitwise_xor(nbite[sitei],1)
                    nja=np.bitwise_xor(nbite[sitej],1)
                    invsite[sitei]=nia
                    invsite[sitej]=nja                    
                    signi=(-1)**i
                    signj=(-1)**(j-1)
                    iqp=kpq_index[sitei,iq,0]#iqx*num_k1+iqy   #sitei+q
                    jqm=kmq_index[sitej,iq,0]#jqx*num_k1+jqy    #sitej-q   
                    invsite0=invsite.copy()
                    #if (dqx%num_k0!=0 or dqy%num_k1!=0):
                    if (invsite0[iqp]==0 and invsite0[jqm]==0 and iqp!=jqm):
                        #invsiteback=invert_bit(invsite,iqp,jqm)
                        remoccup=np.where(invsite0==1)[0]
                        signjqm=0
                        for jt in range(0,Ne-2):
                            if remoccup[jt]<jqm:
                                signjqm=signjqm+1
                        signjqm=(-1)**signjqm
                        
                        
                        invsiteback1=invsite                                                
                        njat=np.bitwise_xor(invsite[jqm],1)
                        invsiteback1[jqm]=njat
                        
                        invsiteback1_0=invsiteback1.copy()
                        
                        signiqp=0
                        remoccupt=np.where(invsiteback1_0==1)[0]
                        for it in range(0,Ne-1):
                            if remoccupt[it]<iqp:
                                signiqp=signiqp+1
                        
                        signiqp=(-1)**signiqp
                                                
                        invsiteback=invsiteback1
                        niat=np.bitwise_xor(invsiteback1[iqp],1)
                        invsiteback[iqp]=niat

                        
                        nt=Ihilbert[bits2int(invsiteback)]  
                        nkt=(np.where(Nhilbert_ik==nt)[0])  ## find the index in ik block which equals to nt
                        nkt=nkt[0]
                        rowindex.append(nkt)
                        columnindex.append(nk)
                        overallsign=signiqp*signjqm*signi*signj
                        H.append(overallsign*((Coulomb[sitei,sitej,iq])*.5/Nsite+(Coulomb_m[sitej,sitei,iq])*.5/Nsite))
                      
    return H, rowindex, columnindex, Nhilbert_ik

@njit
def momentum(Vg0,Nhilbert_ik):
    Kx=0.0
    Ky=0.0
    siteoccup=np.zeros(Nsite,'float')
    for nk in range(0,Nhilbert_ik.shape[0]):
        for ie in range(0,Ne):
            kxtmp=kvecindex[Nfilled[Nhilbert_ik[nk],ie],0]#//num_k1
            kytmp=kvecindex[Nfilled[Nhilbert_ik[nk],ie],1]#-kxtmp*num_k1
            #print(kxtmp,kytmp)
            Kx+=((abs(Vg0[nk])**2)*kxtmp)
            Ky+=((abs(Vg0[nk])**2)*kytmp)
        #for j in range(0,Nsite):
        #    siteoccup
    for j in range(0,Nsite):
        for nk in range(0,Nhilbert_ik.shape[0]):
            siteoccup[j]+=(abs(Vg0[nk])**2)*Nhilbert[Nhilbert_ik[nk],j]
    Kx=(Kx%num_k0)/norm(Vg0)
    Ky=(Ky%num_k1)/norm(Vg0)
    #print(norm(Vg0))    
    return Kx, Ky, siteoccup

def ED_mpi(Ne,Coulomb,Coulomb_m):
    ########################### MPI ###########################
    # If the size of array to be parallelized *can not be divided* by the number of cores,
    # the array will be diveded into subsets with 2 types of size:
    # {num_more} subsets have {subset_size+1} elements, lefted are the subsets with {subset_size} elements
    subset_size,num_more=divmod(num_kpt,size)
    k_subsets=[range(num_kpt)[i*(subset_size+1):(i+1)*(subset_size+1)] if i < num_more else range(num_kpt)[i*subset_size+num_more:(i+1)*subset_size+num_more] for i in range(size)] # divide kvec into size subsets
    k_subset=comm.scatter(k_subsets,root=0)
    ###########################################################

    Enk=zeros((len(k_subset),10),'float')
    Kx=zeros((len(k_subset),10),'float')
    Ky=zeros((len(k_subset),10),'float')
    siteoccup=zeros((len(k_subset),10,num_kpt),'float')
    for n,ik in enumerate(k_subset):
        if ik==[]:
            pass
        else:
            if rank==0:
                print('Calculating Hamiltonian and indices')
                time1=time.time()
            Hk,rowindexk,columnindexk,Nhilbert_ik=Hamiltonian_momentum(ik,Ne,Coulomb,Coulomb_m)
            if rank==0:
                time2=time.time()
                print('Hamiltonian and indices calculation time elapse is ', time2-time1) 
                print('-----------------------------------------------------------------------------')

            if rank==0:
                print('Constructing sparse Hamiltonian')
                time1=time.time()
            Nstatek=Nhilbert_ik.shape[0]
            # Hamk=sp.sparse.coo_matrix((Hk,(rowindexk,columnindexk)),shape=(Nstatek, Nstatek),  dtype=complex).todense()
            Hamk=sp.sparse.coo_matrix((Hk,(rowindexk,columnindexk)),shape=(Nstatek, Nstatek),  dtype=complex).tocsr()
            if rank==0:
                time2=time.time()
                print('Sparse Hamiltonian construction time elapse is ', time2-time1) 
                print('-----------------------------------------------------------------------------')
            
            if rank==0:
                print('Diagonalizing Hamiltonian')
                time1=time.time()
            # Enkt, Vnkt= eigh(Hamk)
            Enkt, Vnkt= eigs(Hamk,k=10,which='SR')
            Enk[n]=Enkt.real
            if rank==0:
                time2=time.time()
                print('Hamiltonian diagonalization time elapse is ', time2-time1)    
                print('-----------------------------------------------------------------------------')

            if rank==0:
                print('Calculating momentum')
                time1=time.time()
            for i in range(10):
                Kx[n,i],Ky[n,i],siteoccup[n,i]=momentum(Vnkt[:,i],Nhilbert_ik)
            if rank==0:
                time2=time.time()
                print('Momentum calculation time elapse is', time2-time1)  
                print('-----------------------------------------------------------------------------')

    ########################### MPI ###########################
    Enk_gather=comm.gather(Enk,root=0)
    Kx_gather=comm.gather(Kx,root=0)
    Ky_gather=comm.gather(Ky,root=0)
    siteoccup_gather=comm.gather(siteoccup,root=0)
    if rank==0:
        Enk=concatenate(Enk_gather)
        Kx=concatenate(Kx_gather)
        Ky=concatenate(Ky_gather)
        siteoccup=concatenate(siteoccup_gather)
    else:
        Enk=None
        Kx=None
        Ky=None
        siteoccup=None
    Enk=comm.bcast(Enk,root=0)
    Kx=comm.bcast(Kx,root=0)
    Ky=comm.bcast(Ky,root=0)
    siteoccup=comm.bcast(siteoccup,root=0)
    ###########################################################
    return Enk,Kx,Ky,siteoccup

if __name__=='__main__':
    # #plot_band(0.0,0.0)
    # #Chern1,Chern2,Chernt, Eindgap, Edgap, Berry1, Berry2=chern_number(0.0,20)

    if rank==0:
        print('Calculating Coulomb parameters')
        time1=time.time()
    # Coulomb,Coulomb_m=get_Coulomb()
    Coulomb,Coulomb_m=get_Coulomb_mpi()
    if rank==0:
        time2=time.time()
        print('Coulomb parameters calculation time elapse is', time2-time1) 
        print('-----------------------------------------------------------------------------')

    Enk,Kx,Ky,siteoccup=ED_mpi(Ne,Coulomb,Coulomb_m)
    if rank==0:
        Enk_1st=Enk[:,0]
        Enk_argsort=argsort(Enk.reshape(num_kpt*10))
        Kx=Kx.reshape(num_kpt*10)[Enk_argsort]
        Ky=Ky.reshape(num_kpt*10)[Enk_argsort]
        siteoccup=siteoccup.reshape((num_kpt*10,num_kpt))[Enk_argsort]
        En=sort(Enk.flatten())
        np.savez(f'Ne{Ne}_Dfield{Dfield:.3f}_eps{eps_r:.1f}_flux{flux2:.2f}_theta{theta0:.3f}.npz',Ek0=Ek0,Enk=Enk,En=En,Kx=Kx,Ky=Ky,siteoccup=siteoccup)
        for i in range(10):
            print('[Kx, Ky] = ', [Kx[i], Ky[i]])
        print(siteoccup[0])
        print(sum(siteoccup[0]))
        print(siteoccup[1])
        print(sum(siteoccup[1]))
        print(siteoccup[2])
        print(sum(siteoccup[2]))
        print(siteoccup[3])
        print(sum(siteoccup[3]))
        print(siteoccup[4])
        print(sum(siteoccup[4]))
        print(En)
        print(f'FCI ground state gap is {En[3]-En[2]} eV')
        print(f'Ek0 sum over = {sum(Ek0)} eV')
        fig,ax=plt.subplots(figsize=(4,4))
        ax.set_ylabel('$E-E_{GS}$ (meV)')
        ax.set_xlabel('$k_0N_1+k_1$')
        # ax.set_xlim(1.5,4.0)
        ax.plot((Enk-Enk.min())*1e3,'ro')
        ax.set_title(f'Ne={Ne} Dfield={Dfield:.3f} $\epsilon_r$={eps_r:.1f}')
        fig.tight_layout()
        # fig.savefig(f'Ne{Ne}_theta0{theta0:.3f}_flux1{flux1:.1f}_flux2{flux2:.1f}_kin{1}_RG{indRG}.pdf')
        # fig.savefig(f'ED_dc{dc}_RG{indRG}_HFfilling{filling:.2f}_EDfilling{Ne/num_kpt:.2f}_flux2{flux2:.1f}.pdf')
        fig.savefig(f'Ne{Ne}_Dfield{Dfield:.3f}_eps{eps_r:.1f}_flux{flux2:.2f}_theta{theta0:.3f}.pdf')

        time4=time.time()
        print('Total time elapse is', time4-time3) 
    # if rank==0:
    #     print(sum(Coulomb),sum(Coulomb_m))
    #     print(sum(abs(Coulomb)),sum(abs(Coulomb_m)))
    #     print(Coulomb_proj(1,3,2))
    #     print(Coulomb_projm(1,3,2))
    #     print(sum(Ek0),sum(psi0),sum(abs(psi0)))

