# -*- coding: utf-8 -*-
#Created on Fri Aug 31 16:02:49 2018

#@author: ryanday
#MIT License

#Copyright (c) 2018 Ryan Patrick Day

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



import contextlib
import sys

import joblib
import matplotlib.pyplot as plt
import numba
import numpy as np
from numba_progress import ProgressBar as tqdm_numba
from scipy.special import erf

import chinook.tetrahedra as tetrahedra


def dos_broad(TB,NK,NE=None,dE=None,origin = np.zeros(3), ax=None, **plot_kw):
    '''
    Energy-broadened discrete density of states calculation.
    The Hamiltonian is diagonalized over the kmesh defined by NK and
    states are summed, as energy-broadened Gaussian peaks, rather than
    delta functions. 
    
    *args*:

        - **TB**: tight-binding model object
        
        - **NK**: int, or tuple of int, indicating number of k-points
        
    *kwargs*:

        - **NE**: int, number of energy bins for final output
        
        - **dE**: float, energy broadening of peaks, eV
        
        - **origin**: numpy array of 3 float, indicating the origin of the mesh to be used,
        relevant for example in kz-specific contributions to density of states
        
    *return*:

        - **DOS**: numpy array of float, density-of-states in states/eV
        
        - **Elin**: numpy array of float, energy domain in eV
    
    ***
    '''
    kpts = tetrahedra.gen_mesh(TB.avec, NK) + origin
    TB.Kobj.kpts = kpts
    TB.solve_H()
    # print('Calculating DOS...\n')
    if dE is None:
        dE = def_dE(TB.Eband)
        print('Broadening: {:0.04f} eV\n'.format(dE))
    Elin = np.arange(TB.Eband.min()-10*dE, TB.Eband.max()+10*dE,dE*0.5)
    with tqdm_numba(desc='Calculating DOS',total=len(kpts)) as pb:
        DOS = iterate_dos_broad(len(kpts), len(TB.basis), TB.Eband, Elin, dE, pb)
    if NE is not None:
        E_resample = np.linspace(Elin[0],Elin[-1],NE)
        DOS_resample = np.interp(E_resample,Elin,DOS)
        DOS = DOS_resample
        Elin = E_resample
    if ax is None:
        ax = plt.gca()
    ax.plot(Elin, DOS, **plot_kw)
    return DOS, Elin

@numba.njit(nogil=True)
def iterate_dos_broad(len_kpts, len_basis, Eband, Elin, dE, progress_hook):
    DOS = np.zeros(len(Elin))
    for ki in numba.prange(len_kpts):
        for bi in range(len_basis):
            DOS += gaussian(Eband[ki,bi],Elin,dE)
        progress_hook.update(1)
    return DOS / len_kpts

@numba.njit(nogil=True)
def def_dE(Eband):
    
    '''

    If energy broadening is not passed for density-of-states calculation,
    compute a reasonable value based on the energy between adjacent energies
    in the tight-binding calculation
    
    *args*:

        - **Eband**: numpy array of float, indicating band energies
        
    *return*:

        **dE**: float, energy broadening, as the smallest average energy spacing
        over all bands.
    
    ***
    '''
    dE = Eband.max() - Eband.min()
    for bi in numba.prange(np.shape(Eband)[1]):
        diff = np.abs(Eband[1:,bi] - Eband[:-1,bi]).mean()
        if dE > diff:
            dE = diff
    return dE

def ne_broad_numerical(TB,NK,NE=None,dE=None,origin=np.zeros(3)):
    '''
    Occupation function, as a numerical integral over the density of states function.
    
    
    *args*:

        - **TB**: tight-binding model object
        
        - **NK**: int, or tuple of int, indicating number of k-points
        
    *kwargs*:

        - **NE**: int, number of energy bins for final output
        
        - **dE**: float, energy spacing of bins, in eV
        
        - **origin**: numpy array of 3 float, indicating the origin of the mesh to be used,
        relevant for example in kz-specific contributions to density of states
        
    *return*:

        - **ne**: numpy array of float, integrated density-of-states at each energy
        
        - **Elin**: numpy array of float, energy domain in eV
    
    ***
    '''
    return _ne_broad_numerical_calc(*dos_broad(TB,NK,NE,dE,origin))

@numba.njit(nogil=True)
def _ne_broad_numerical_calc(DOS, Elin):
    delta = Elin[1] - Elin[0]
    ne = np.zeros_like(Elin)
    for i in numba.prange(len(Elin)):
        if i == 0:
            val = (2 * DOS[i] + DOS[i+1]) / 3
        elif i == (len(Elin) - 1):
            val = (2 * DOS[i] + DOS[i-1]) / 3
            ne[i] = ne[i-1]
        else:
            val = (DOS[i-1] + DOS[i] * 2 + DOS[i+1]) / 4
            ne[i] = ne[i-1]
        ne[i] += val * delta
    return ne, Elin

def find_EF_broad_dos(TB,NK,occ,NE=None,dE=None,origin=np.zeros(3)):
    '''
    Find the Fermi level of a model Hamiltonian, for a designated electronic
    occupation. Note this is evaluated at T=0, so EF is well-defined.

    *args*:

        - **TB**: tight-binding model object
        
        - **NK**: int, or tuple of int, indicating number of k-points
        
        - **occ**: float, desired electronic occupation
        
    *kwargs*:

        - **NE**: int, number of energy bins for final output
        
        - **dE**: float, energy spacing of bins, in eV
        
        - **origin**: numpy array of 3 float, indicating the origin of the mesh to be used,
        relevant for example in kz-specific contributions to density of states
        
    *return*:

        - **EF**: float, Fermi level in eV

    ***
    '''
    
    ne,Elin = ne_broad_numerical(TB,NK,NE,dE,origin)
    ind_EF = np.where(ne>occ)[0][0]
    EF = Elin[ind_EF]
    print('Fermi Energy is at: {:0.05f}+/-{:0.06f} eV'.format(EF,Elin[1]-Elin[0]))
    return EF

def ne_broad_analytical(TB,NK,NE=None,dE=None,origin=np.zeros(3),plot = True):
    '''

    Analytical evaluation of the occupation function. Uses scipy's errorfunction
    executable to evaluate the analytical form of a Gaussian-broadened state's contribution
    to the total occupation, at each energy
    
    *args*:

        - **TB**: tight-binding model object
        
        - **NK**: int, or tuple of int, indicating number of k-points
        
    *kwargs*:

        - **NE**: int, number of energy bins for final output
        
        - **dE**: float, energy spacing of bins, in eV
        
        - **origin**: numpy array of 3 float, indicating the origin of the mesh to be used,
        relevant for example in kz-specific contributions to density of states
        
        - **plot**: boolean, default to True, if false, suppress plot output
        
    *return*:

        - **nE**: numpy array of float, occupied states
        
        - **Elin**: numpy array of float, energy domain in eV

    ***
    '''
    kpts = tetrahedra.gen_mesh(TB.avec,NK)+origin #just a mesh over the Brillouin zone
    TB.Kobj.kpts = kpts
    TB.solve_H()
    print('Diagonalization complete')

    if dE is None:
        dE = def_dE(TB.Eband)
        print('Broadening: {:0.04f} eV\n'.format(dE))
    Elin = np.arange(TB.Eband.min()-10*dE,TB.Eband.max()+10*dE,dE*0.5)

    nE = np.zeros(len(Elin))
    for ki in range(len(kpts)):
        sys.stdout.write('\r'+progress_bar(ki+1,len(kpts)))
        for bi in range(len(TB.basis)): #iterate over all bands
            tmp_add =  error_function(TB.Eband[ki,bi],Elin,dE)
            nE+=tmp_add
    print('\n')
    nE /=len(kpts)
    if NE is not None:
        E_resample = np.linspace(Elin[0],Elin[-1],NE)
        nE_resample = np.interp(E_resample,Elin,nE)
        Elin = E_resample
        nE = nE_resample
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(Elin,nE)
    return nE,Elin

# @numba.njit()
def error_function(x0,x,sigma):
    
    '''
    Integral over the gaussian function, evaluated from -infinity to x, using
    the scipy implementation of the error function
    
    *args*:

        - **x0**: float, centre of Gaussian, in eV
        
        - **x**: numpy array of float, energy domain eV
        
        - **sigma**: float, width of Gaussian, in eV
        
    *return*:

        - analytical form of integral
    
    ***
    '''
    return 0.5*(erf((x-x0)/(np.sqrt(2)*sigma))+1)      
            
            
@numba.njit(nogil=True)
def gaussian(x0, x, sigma):
    '''
    Evaluate a normalized Gaussian function.
    
    *args*:

        - **x0**: float, centre of peak, in eV
        
        - **x**: numpy array of float, energy domain in eV
        
        - **sigma**: float, width of Gaussian, in eV

    *return*:

        - numpy array of float, gaussian evaluated.

    ***
    '''
    return np.exp(-(x - x0)**2 / (2 * sigma**2)) / np.sqrt(2*np.pi) / sigma
################# Density of States following the Blochl Prescription #######################
###############https://journals.aps.org/prb/pdf/10.1103/PhysRevB.49.16223####################
    
def dos_tetra(TB, NE, NK, ax=None, **plot_kw):
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
    kpts, tetra = tetrahedra.mesh_tetra(TB.avec, NK)
    TB.Kobj.kpts = kpts
    TB.solve_H()
    Elin = np.linspace(TB.Eband.min(), TB.Eband.max(), NE) 
    with tqdm_numba(desc='Calculating DOS', total=len(tetra)) as pb:
        DOS = _calc_dos_tetra(len(tetra), len(TB.basis), 
                                TB.Eband, Elin, tetra, pb)
    if ax is None:
        ax = plt.gca()
    ax.plot(Elin, DOS, **plot_kw)
    return Elin, DOS

@numba.njit(nogil=True)
def _calc_dos_tetra(len_tetra, len_basis, Eband, Elin, tetra, progress_hook):
    DOS = np.zeros_like(Elin, dtype=np.float64)
    for ki in numba.prange(len_tetra):
        E_tmp = Eband[tetra[ki]]
        for bi in range(len_basis):
            DOS += band_contribution(E_tmp[:,bi], Elin, len_tetra)
        progress_hook.update(1)
    return DOS

@numba.njit(nogil=True)
def band_contribution(eigenvals, w_domain, volume):
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
    args = np.array([*eig_sort, 1, volume], dtype=np.float64)
    DOS = dos_func(w_domain,args)
    return DOS

@numba.njit(nogil=True)
def band_occupation(eigenvals, w_domain, volume):
    '''
    Compute the contribution over a single tetrahedron, from a 
    single band, to the electronic occupation number
    
    *args*:

        - **eigenvals**: numpy array of float, energy values at corners
        
        - **w_domain**: numpy array of float, energy domain
        
        - **volume**: int, number of tetrahedra in the total mesh
    
    *return*:

        - **DOS**: numpy array of float, same length as w_domain
    
    ***
    '''
    eig_sort = sorted(eigenvals)
    args = np.array([*eig_sort, 1, volume], dtype=np.float64)
    n_elect = n_func(w_domain,args)
    return n_elect






########################Partial Density of States##############################

@numba.njit(nogil=True)
def proj_avg(eivecs, proj_matrix):
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
    # return np.real(
    #     0.25 * np.einsum('ijk,ijk->k', 
    #                      np.conj(eivecs),
    #                      np.einsum('ij,kjl->kil',
    #                                proj_matrix, eivecs),
    #                      optimize=True))
    return np.real(0.25 * _einsum_ijk_ijk_k(np.conj(eivecs), 
                                     _einsum_ij_kjl_kil(proj_matrix, eivecs)))

@numba.njit(nogil=True)
def _einsum_ij_kjl_kil(a, b):
    ii, jj = a.shape
    kk, _, ll = b.shape
    out = np.empty((kk, ii, ll), np.complex128)
    for k in numba.prange(kk):
        for i in range(ii):
            for l in range(ll):
                out[k,i,l] = 0.
                for j in range(jj):
                    out[k,i,l] += a[i,j] * b[k,j,l]
    return out
@numba.njit(nogil=True)
def _einsum_ijk_ijk_k(a, b):
    ii, jj, kk = a.shape
    out = np.zeros(kk, np.complex128)
    for i in range(ii):
        for j in range(jj):
            for k in range(kk):
                out[k] += a[i,j,k] * b[i,j,k]
    return out

@numba.njit(nogil=True)
def proj_mat(proj, lenbasis):
    '''
    Define projection matrix for fast evaluation of the partial density of states
    weighting. As the projector here is diagonal, and represents a Hermitian 
    matrix, it is by definition a real matrix operator.
    
    *args*:

        - **proj**: numpy array, either 1D (indices of projection), or 2D (indices of
        projection and weight of projection)
        
        - **lenbasis**: int, size of the orbital basis
    
    *return*:

        - **projector**: numpy array of float, lenbasis x lenbasis
    
    ***
    '''
    projector = np.eye(lenbasis, dtype=np.complex128)
    proj_vect = np.zeros(lenbasis, dtype=np.complex128)
    if proj.ndim == 1:
        # proj_vect[proj] = 1/np.sqrt(len(proj))
        for p in proj:
            # proj_vect[p] = 1/np.sqrt(len(proj))
            proj_vect[p] = 1
    elif proj.ndim == 2:
        for i in proj[:,0]:
            proj_vect[i] = proj[i,1]
        # proj_vect[proj[:,0]] = proj[:,1]
        proj_vect /= np.sqrt(np.inner(np.conj(proj_vect), proj_vect))
    projector *= np.real(proj_vect)
    return projector

def pdos_tetra(TB, NE, NK, proj, ax=None, **plot_kw):
    
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
    kpts, tetra = tetrahedra.mesh_tetra(TB.avec, NK)
    TB.Kobj.kpts = kpts
    TB.solve_H()
    Elin = np.linspace(TB.Eband.min(), TB.Eband.max(), NE)
    if isinstance(proj, list):
        with tqdm_numba(desc='Calculating pDOS', total=len(tetra)) as pb:
            DOS, pDOS = iterate_mult_pdos_tetra(
                len(tetra), len(TB.basis), len(proj), TB.Eband, Elin, TB.Evec,
                numba.typed.List([proj_mat(p, len(TB.basis)) for p in proj]),
                tetra, pb,
            )
    else:
        with tqdm_numba(desc='Calculating pDOS', total=len(tetra)) as pb:
            DOS, pDOS = iterate_pdos_tetra(
                len(tetra), len(TB.basis), TB.Eband, Elin, TB.Evec,
                proj_mat(proj, len(TB.basis)), tetra, pb,
            )
    if ax is None:
        ax = plt.gca()
    ax.plot(Elin, DOS, **plot_kw)
    if isinstance(proj, list):
        for i in range(len(proj)):
            ax.plot(Elin, pDOS[i,:], **plot_kw)
    else:
        ax.plot(Elin, pDOS, **plot_kw)
    return Elin, pDOS, DOS
    
@numba.njit(nogil=True)
def iterate_pdos_tetra(len_tetra, len_basis, Eband, Elin, Evec, projmat, tetra, progress_hook):
    DOS = np.zeros_like(Elin, dtype=np.float64)
    pDOS = np.zeros_like(Elin, dtype=np.float64)
    for ki in range(len_tetra):
        projection_avg = proj_avg(Evec[tetra[ki],:,:], projmat)
        for bi in range(len_basis):
            DOS_tetra = band_contribution(Eband[tetra[ki]][:,bi], Elin, len_tetra)
            pDOS += DOS_tetra * projection_avg[bi]
            DOS += DOS_tetra
        progress_hook.update(1)
    return DOS, pDOS
@numba.njit(nogil=True)
def iterate_mult_pdos_tetra(len_tetra, len_basis, len_proj, Eband, Elin, Evec, projmat_list, tetra, progress_hook):
    DOS = np.zeros_like(Elin, dtype=np.float64)
    pDOS = np.zeros((len_proj, len(Elin)), dtype=np.float64)
    for ki in range(len_tetra):
        projection_avg = [proj_avg(Evec[tetra[ki],:,:], pm) for pm in projmat_list]
        for bi in range(len_basis):
            DOS_tetra = band_contribution(Eband[tetra[ki]][:,bi], Elin, len_tetra)
            DOS += DOS_tetra
            for pi in numba.prange(len_proj):
                pDOS[pi,:] += DOS_tetra * projection_avg[pi][bi]
        # projection_avg = np.zeros(len_basis, dtype=np.float64)
        progress_hook.update(1)
    return DOS, pDOS

##############################-------D(E)---------#############################
@numba.njit(nogil=True)
def dos_func(energy,epars):
    '''

    Piecewise function for calculation of density of states
    
    *args*:

        - **energy**: numpy array of float (energy domain)
        
        - **epars**: tuple of parameters: e[0],e[1],e[2],e[3],V_T,V_G being the ranked band energies for the tetrahedron, 
        as well as the volume of both the tetrahedron and the Brillouin zone, all float
    
    *return*:

        - numpy array of float giving DOS contribution from this tetrahedron
    
    ***
    '''
    output = np.zeros_like(energy)
    for i in range(len(energy)):
        if (energy[i] < epars[0]) or (energy[i] >= epars[3]):
            output[i] = 0
        elif energy[i] < epars[1]:
            output[i] = e_12(energy[i], epars)
        elif energy[i] < epars[2]:
            output[i] = e_23(energy[i], epars)
        elif energy[i] < epars[3]:
            output[i] = e_34(energy[i], epars)
    return output

@numba.njit(nogil=True)
def e_12(energy, epars):
    return epars[4] / epars[5] * 3 * (energy - epars[0])**2 / (epars[1] - epars[0]) / (epars[2]-epars[0]) / (epars[3]-epars[0])

@numba.njit(nogil=True)
def e_23(energy, epars):
    e21, e31, e41, e42, e32 = epars[1] - epars[0], epars[2] - epars[0], epars[3] - epars[0], epars[3] - epars[1], epars[2] - epars[1]
    e2 = energy - epars[1]
    return epars[4] / epars[5] / e31 / e41 * (3 * e21 + 6 * e2 - 3 * (e31 + e42) / (e32 * e42) * e2**2)

@numba.njit(nogil=True)
def e_34(energy, epars):
    return epars[4] / epars[5] * 3 * (epars[3] - energy)**2 / (epars[3] - epars[0]) / (epars[3] - epars[1]) / (epars[3] - epars[2])
##############################-------D(E)---------#############################


##############################-------n(E)---------#############################
def find_EF_tetra_dos(TB,occ,dE,NK):

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
    e_domain, n_elec = n_tetra(TB,dE,NK)
    EF = e_domain[np.where(abs(n_elec-occ)==abs(n_elec-occ).min())[0][0]]
    return EF


def n_tetra(TB,dE,NK,plot=True,ax=None):
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
    kpts, tetra = tetrahedra.mesh_tetra(TB.avec, NK)
    TB.Kobj.kpts = kpts
    TB.solve_H()
    Elin = np.arange(TB.Eband.min(),TB.Eband.max(),dE)
    with tqdm_numba(desc='Calculating integrated DOS', total=len(tetra)) as pb:
        n_elect = _calc_n_tetra(len(tetra), len(TB.basis), TB.Eband, Elin, tetra, pb)
    if plot:
        if ax is None:
            ax = plt.gca()
        ax.plot(Elin,n_elect)               
    return Elin,n_elect

@numba.njit(nogil=True)
def _calc_n_tetra(len_tetra, len_basis, Eband, Elin, tetra, progress_hook):
    n_elect = np.zeros_like(Elin, dtype=np.float64)
    for ki in range(len_tetra):
        E_tmp = Eband[tetra[ki]]
        for bi in range(len_basis): #iterate over all bands
            n_elect += band_occupation(E_tmp[:,bi], Elin, len_tetra)
        progress_hook.update(1)
    return n_elect

@numba.njit(nogil=True)
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
    output = np.zeros_like(energy)
    for i in range(len(energy)):
        if energy[i] < epars[0]:
            output[i] = 0
        elif energy[i] < epars[1]:
            output[i] = n12(energy[i], epars)
        elif energy[i] < epars[2]:
            output[i] = n23(energy[i], epars)
        elif energy[i] < epars[3]:
            output[i] = n34(energy[i], epars)
        else:
            output[i] = epars[4] / epars[5]
    return output

@numba.njit(nogil=True)
def n12(energy,epars):
    return epars[4]/epars[5]*(energy-epars[0])**3/(epars[1]-epars[0])/(epars[2]-epars[0])/(epars[3]-epars[0])
@numba.njit(nogil=True)
def n23(energy,epars):
    e21,e31,e41,e42,e32 = epars[1]-epars[0],epars[2]-epars[0],epars[3]-epars[0],epars[3]-epars[1],epars[2]-epars[1]
    e2 = energy-epars[1]
    return epars[4]/epars[5]*(1/(e31*e41))*(e21**2+3*e21*(e2)+3*e2**2-(e31+e42)/(e32*e42)*(e2**3))
@numba.njit(nogil=True)
def n34(energy,epars):
    return epars[4]/epars[5]*(1-(epars[3]-energy)**3/(epars[3]-epars[0])/(epars[3]-epars[1])/(epars[3]-epars[2]))

##############################-------n(E)---------#############################
    
def progress_bar(N,Nmax):
    frac = N/Nmax
    st = ''.join(['|' for i in range(int(frac*30))])
    st = '{:30s}'.format(st)+'{:3d}%'.format(int(frac*100))
    return st

# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
