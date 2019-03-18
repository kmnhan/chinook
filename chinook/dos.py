# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:02:49 2018

@author: rday

Calculation of Density of States

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import chinook.tetrahedra as tetrahedra


################# Density of States following the Blochl Prescription #######################
###############https://journals.aps.org/prb/pdf/10.1103/PhysRevB.49.16223####################
    
def dos_tetra(TB,NE,NK):
    '''
    Generate a tetrahedra mesh of k-points which span the BZ with even distribution
    Diagonalize over this mesh and then compute the resulting density of states as
    prescribed in the above paper. 
    The result is plotted, and DOS returned
    
    *args*:
        - **TB**: tight-binding model object
        
        - **NE**: int, number of energy points
        
        - **NK**: int or list of 3 int -- number of k-points in mesh
        
    *return*:
        - **Elin**: linear energy array of float, spanning the range of the eigenspectrum
        
        - **DOS**: numpy array of float, same length as Elin, density of states
    
    ***
    '''
    kpts,tetra = tetrahedra.mesh_tetra(TB.avec,NK)
    print('k tetrahedra defined')
    TB.Kobj.kpts = kpts
    TB.solve_H()
    print('Diagonalization complete')
    Elin = np.linspace(TB.Eband.min(),TB.Eband.max(),NE)
    
    DOS = np.zeros(len(Elin))
    for ki in range(len(tetra)):
        sys.stdout.write('\r'+progress_bar(ki+1,len(tetra)))
        for bi in range(len(TB.basis)): #iterate over all bands
            DOS += band_contribution(TB.Eband[tetra[ki]][:,bi],Elin,len(tetra))
    print('DOS calculation complete')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Elin,DOS)               
    return Elin,DOS

def band_contribution(eigenvals,w_domain,volume):
    
    '''
    Compute the contribution over a single tetrahedron, from a 
    single band, to the density of states
    
    *args*:
        - **eigenvals**: numpy array of float, energy values at corners
        
        - **w_domain**: numpy array of float, energy domain
        
        - **volume**: int, number of tetrahedra in the total mesh
    
    *return*:
        - **DOS**: numpy array of float, same length as w_domain
    
    ***
    '''
    eig_sort = sorted(eigenvals)
    args = (*eig_sort,1,volume)
    DOS = dos_func(w_domain,args)
    
    return DOS

########################Partial Density of States##############################

    
def proj_avg(eivecs,proj_matrix):
    '''
    Calculate the expectation value of the projection operator, for each of the
    eigenvectors, at each of the vertices, and then sum over the vertices. We
    use *numpy.einsum* to perform matrix multiplication and contraction.
    
    *args*:
        - **eivecs**: numpy array of complex float, 4xNxM, with M number of eigenvectors,
        N basis dimension
        
        - **proj_matrix**: numpy array of complex float, NxN in size
        
    *return*:
        - numpy array of M float, indicating the average projection over the 4 
        corners of the tetrahedron
      
    ***
    '''
    return np.real(0.25*np.einsum('ijk,ijk->k',np.conj(eivecs),np.einsum('ij,kjl->kil',proj_matrix,eivecs)))

def proj_mat(proj,lenbasis):
    '''
    Define projection matrix for fast evaluation of the partial density of states
    weighting. As the projector here is diagonal, and represents a Hermitian 
    matrix, it is by definition a real matrix operator.
    
    *args*:
        - **proj**: numpy array, either 1D (indices of projection), or 2D (indices of
        projection and weight of projection)
        
        - **lenbasis**: int, size of the orbital basis
    
    *return*:
        - numpy array of float, lenbasis x lenbasis
    
    ***
    '''
    projector = np.identity(lenbasis,dtype=complex)
    
    proj_vect = np.zeros(lenbasis,dtype=complex)

    if len(np.shape(proj))==1:
        proj_vect[proj] = 1/(len(proj))**0.5
    
    elif len(np.shape(proj))==2:
        proj_vect[proj[:,0]] = proj[:,1]
        proj_vect/=np.sqrt(np.einsum('i,i',np.conj(proj_vect),proj_vect))
    
    projector*=np.real(proj_vect)
    
    return projector
        
        
    


def pdos_tetra(TB,NE,NK,proj):
    
    '''
    Partial density of states calculation. Follows same tetrahedra method, 
    weighting the contribution of a given tetrahedra by the average projection
    onto the indicated user-defined projection. The average here taken as the sum
    over projection at the 4 vertices of the tetrahedra.
    
    *args*:
        - **TB**: tight-binding model object
        
        - **NE**: int, number of energy bins
        
        - **NK**: int, or iterable of 3 int, indicating the number of k-points
        along each of the axes of the Brillouin zone
        
        - **proj**: numpy array of float, 1D or 2D, c.f. *proj_mat*.
        
    *return*:
        - **Elin**: numpy array of float, with length **NE**, spanning the
        range of the tight-binding bandstructure
        
        - **pDOS**: numpy array of float, len **NE**, projected density of states
        
        - **DOS**: numpy array of float, len **NE**, full density of states
    ***
    '''
    
    projection_matrix = proj_mat(proj,len(TB.basis))
    
    kpts,tetra = tetrahedra.mesh_tetra(TB.avec,NK)
    print('k tetrahedra defined')
    TB.Kobj.kpts = kpts
    TB.solve_H()
    print('Diagonalization complete')
    Elin = np.linspace(TB.Eband.min(),TB.Eband.max(),NE)
    
    DOS = np.zeros(len(Elin))
    pDOS = np.zeros(len(Elin))
    for ki in range(len(tetra)):
        eivecs = TB.Evec[tetra[ki],:,:]
        projection_avg = proj_avg(eivecs,projection_matrix)
        sys.stdout.write('\r'+progress_bar(ki+1,len(tetra)))
        for bi in range(len(TB.basis)): #iterate over all bands
            DOS_tetra = band_contribution(TB.Eband[tetra[ki]][:,bi],Elin,len(tetra))
            pDOS += DOS_tetra*projection_avg[bi]
            DOS += DOS_tetra
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Elin,DOS)
    ax.plot(Elin,pDOS)
    
    return Elin,pDOS,DOS
    
   
##############################-------D(E)---------#############################
def dos_func(energy,epars):
    '''
    Piecewise function for calculation of density of states
    
    *args*:
        - **energy**: numpy array of float (energy domain)
        
        - **epars**: tuple of parameters: e[0],e[1],e[2],e[3],V_T,V_G being the ranked band energies for the tetrahedron, 
        as well as the volume of both the tetrahedron and the Brillouin zone, all float
    
    *return*:
        - numpy array of float giving DOS contribution from this tetrahedron
    '''
    return np.piecewise(energy,[energy<epars[0],(epars[0]<=energy)*(energy<epars[1]),(epars[1]<=energy)*(energy<epars[2]),(epars[2]<=energy)*(energy<epars[3]),energy>=epars[3]],[e_out,e_12,e_23,e_34,e_out],epars)


def e_out(energy,epars):
    return np.zeros(len(energy))

def e_12(energy,epars):
    return epars[4]/epars[5]*3*(energy-epars[0])**2/(epars[1]-epars[0])/(epars[2]-epars[0])/(epars[3]-epars[0])

def e_23(energy,epars):
    e21,e31,e41,e42,e32 = epars[1]-epars[0],epars[2]-epars[0],epars[3]-epars[0],epars[3]-epars[1],epars[2]-epars[1]
    e2 = energy-epars[1]
    return epars[4]/epars[5]/e31/e41*(3*e21+6*e2-3*(e31+e42)/e32/e42*e2**2)

def e_34(energy,epars):
    return epars[4]/epars[5]*3*(epars[3]-energy)**2/(epars[3]-epars[0])/(epars[3]-epars[1])/(epars[3]-epars[2])
##############################-------D(E)---------#############################
    

##############################-------n(E)---------#############################
def EF_find(TB,occ,dE,NK):
    '''
    Use the tetrahedron-integration method to establish the Fermi-level, for a given
    electron occupation.
    
    *args*:
        - **TB**: instance of tight-binding model object from *TB_lib*

        - **occ**: float, desired electronic occupation
        
        - **dE**: estimate of energy precision desired for evaluation of the 
        Fermi-level (in eV)
        
        - **NK**: int or iterable of 3 int, number of k points in mesh.
        
    *return*:
        **EF**: float, Fermi Energy for the desired occupation, to within dE of actual
        value.
    
    ***
    '''
    e_domain,n_elec = n_tetra(TB,dE,NK)
    EF = e_domain[np.where(abs(n_elec-occ)==abs(n_elec-occ).min())[0][0]]
    return EF


def n_tetra(TB,dE,NK,plot=True):
    '''
    This function, also from the algorithm of Blochl, gives the integrated DOS
    at every given energy (so from bottom of bandstructure up to its top. This makes
    for very convenient and precise evaluation of the Fermi level, given an electron
    number)
    
    *args*:
        - **TB**: tight-binding model object
        
        - **dE**: float, energy spacing (meV)
        
        - **NK**: int, iterable of 3 int. number of k-points in mesh
        
        - **plot**: bool, to plot or not to plot the calculated array
    
    *return*:
        - **Elin**: linear energy array of float, spanning the range of the eigenspectrum
        
        - **n_elect**: numpy array of float, same length as **Elin**, integrated DOS 
        at each energy, i.e. total number of electrons occupied at each energy
        
    ***
    '''
    kpts,tetra = tetrahedra.mesh_tetra(TB.avec,NK)
    TB.Kobj.kpts = kpts
    TB.solve_H()
    Elin = np.arange(TB.Eband.min(),TB.Eband.max(),dE)
    n_elect = np.zeros(len(Elin))
    for ki in range(len(tetra)):
        E_tmp = TB.Eband[tetra[ki]]
        for bi in range(len(TB.basis)): #iterate over all bands
            Eband = sorted(E_tmp[:,bi])
            args = (*Eband,1,len(tetra))
            n_elect += n_func(Elin,args)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(Elin,n_elect)               
    return Elin,n_elect    



def n_func(energy,epars):
    '''
    Piecewise function for evaluating contribution of tetrahedra to electronic
    occupation number
    
    *args*:
        - **energy**: numpy array of float, energy domain
        
         - **epars**: tuple of parameters: e[0],e[1],e[2],e[3],V_T,V_G being the ranked band energies for the tetrahedron, 
        as well as the volume of both the tetrahedron and the Brillouin zone, all float
       
    *return*:
        - numpy array of float, same length as **energy**, providing contribution of
        tetrahedra to the occupation function
        
    ***
    '''
    
    
    return np.piecewise(energy,[energy<epars[0],(epars[0]<=energy)*(energy<epars[1]),(epars[1]<=energy)*(energy<epars[2]),(epars[2]<=energy)*(energy<epars[3]),energy>=epars[3]],[n1,n12,n23,n34,n4],epars)

def n1(energy,epars):
    return np.zeros(len(energy))

def n12(energy,epars):
    return epars[4]/epars[5]*(energy-epars[0])**3/(epars[1]-epars[0])/(epars[2]-epars[0])/(epars[3]-epars[0])

def n23(energy,epars):
    e21,e31,e41,e42,e32 = epars[1]-epars[0],epars[2]-epars[0],epars[3]-epars[0],epars[3]-epars[1],epars[2]-epars[1]
    e2 = energy-epars[1]
    return epars[4]/epars[5]*(1/(e31*e41))*(e21**2+3*e21*(e2)+3*e2**2-(e31+e42)/(e32*e42)*(e2**3))

def n34(energy,epars):
    return epars[4]/epars[5]*(1-(epars[3]-energy)**3/(epars[3]-epars[0])/(epars[3]-epars[1])/(epars[3]-epars[2]))

def n4(energy,epars):
    return epars[4]/epars[5]

##############################-------n(E)---------#############################
    
    
def progress_bar(N,Nmax):
    frac = N/Nmax
    st = ''.join(['|' for i in range(int(frac*30))])
    st = '{:30s}'.format(st)+'{:3d}%'.format(int(frac*100))
    return st
    
    
