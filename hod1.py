#chm, Teruel, May 2020
#
#We try to assign central and satellite labels to mJPAS galaxies
#
from astropy.io import fits
from astropy.io import ascii
import scipy
from scipy.io.idl import readsav
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
#import wget
import sys
import os
import numpy as np
#
#
#Plotting specifications
plt.rc('font',**{'family':'Times New Roman','serif':['Times New Roman'],'size':13})
plt.rcParams['text.latex.preamble']=[r'\usepackage[varg]{txfonts}',]
plt.rc('text', usetex=True)
plt.rc('xtick', direction='in',labelsize=10)
#
# Reading data
#
i_deneb=1
#
if (i_deneb == 0):
        os.chdir('/data/chm/main/python/libs_chm/')
        import module_cosmo1 as mcosmo1
        os.chdir('/data2/miniJPAS/soft/')
else:
        os.chdir('/Users/chm/work/python/libs_chm/')
        import module_cosmo1 as mcosmo1
        os.chdir('/Users/chm/work/scratch/miniJPAS/soft')
cosmo1_dict={'Omegam':0.3089, 'OmegaL':1.-0.30890,  'h0':0.6774, 'nS':0.9667, 'w0':-1., 'wa':0., 'sg80':0.8147, 'Omegab':0.0486} # PDD
#cosmo1_dict={'Omegam':0.3111, 'OmegaL':1.-0.3111,  'h0':0.6766, 'nS':0.9665, 'w0':-1., 'wa':0., 'sg80':0.8102, 'Omegab':0.049} # Tiago's
#cosmo1_dict={'Omegam':0.315143, 'OmegaL':0.684857, 'h0':0.6726, 'nS':0.963, 'w0':-1., 'wa':0., 'sg80':0.83} # we create a "cosmo" dictionary with Planck DR2 values (fiducial model)
#
fgal='../AMICO/22424.tsv.galaxies'
dgal=np.loadtxt(fgal) ; ngal=np.size(dgal,axis=0)
igal=np.arange(ngal,dtype=int) ; pField=np.zeros(ngal)
#In [3]: dgal.shape                                                                                                      
#Out[3]: (12566, 48)
#
os.system('ls -a ../AMICO/amico_members/mini-JPAS_1???_members.cat > tmp.txt')
#os.system('ls -a ../AMICO/amico_members/*.cat > tmp.txt')
with open('tmp.txt', 'r') as f:
        data = f.read()
os.system('rm -rf tmp.txt')
#
data=data.split()
ngroups=len(data)
for ig in range(ngroups):
        d2=open(data[ig],'r')
        for lines in d2:
                lines=lines.split()
                cgal=(int(lines[0])==dgal[:,1])   & (np.abs(float(lines[1])-dgal[:,2])<8.3e-4)
                print('lines=',data[ig],np.sum(cgal))
                nassg=int(lines[7])
                for iag in range(nassg):
                        index=igal[cgal][0]
                        pField[index]+=float(lines[7+nassg+iag+1])    
                        print('iag=',iag,lines[7+nassg+iag+1])
        d2.close()       
#
plt.clf()
plt.hist(pField,bins=30,color='black',alpha=0.5,label='$P_{\\rm group}$ AMICO')
plt.yscale('log')
#
#
#Now we search for groups :
cgal=(dgal[:,9]<0.5) # at least probable galaxies 
ngal2=np.sum(cgal) ; grID=np.zeros( (ngal2,2), dtype=int)
r_vir=1. # Mpc, not comoving
Deltaz = 0.5e-2
#
v3d=np.zeros((ngal2,3))
for ig in range(ngal2):
        v3d[ig,:]=np.array([np.cos(dgal[:,2][cgal][ig]*np.pi/180.e0) * np.sin(dgal[:,3][cgal][ig]*np.pi/180.e0), \
                np.sin(dgal[:,2][cgal][ig]*np.pi/180.e0) * np.sin(dgal[:,3][cgal][ig]*np.pi/180.e0),  np.cos( dgal[:,3][cgal][ig]*np.pi/180.e0 ) ] )
        v3d[ig,:] = v3d[ig,:] / np.sqrt(np.sum(v3d**2,axis=1) )[ig] # we normalize
igr=int(0)
zgal=dgal[:,22][cgal] ; rgal=dgal[:,18][cgal] ; dotp=np.zeros(ngal2) ; indx=np.arange(ngal2,dtype=int)
for ig in range(ngal2):
        cgr=np.zeros(ngal2,dtype=bool) # boolean array of "False" elements
        dotp[:]=v3d[ig,0]*v3d[:,0] + v3d[ig,1]*v3d[:,1] + v3d[ig,2]*v3d[:,2]
        cgr=(dotp > np.cos( r_vir * cosmo1_dict['h0'] * (1.+zgal[ig]) / mcosmo1.w_R_Z(np.array([zgal[ig]]),**cosmo1_dict ) ) ) & ( np.abs(zgal[ig]-zgal)<Deltaz )
        if (np.sum(cgr)>0):
                if (grID[ig,0]==0): # New group
                        igr+=1
                        grID[:,0][cgr]=igr  
                        grID[:,1][cgr]=1 # satellite
                        cCen=(rgal[cgr]==np.min(rgal[cgr]))
                        indxC=indx[cgr][cCen][0]
                        grID[indxC,1]=2 # central (the most luminous one)
                if (grID[ig,0]!=0): # Old group [already identified]
                        grID[:,0][cgr]=grID[ig,0]
                        cgr= grID[:,0]==grID[ig,0]
                        cCen=(rgal[cgr ] == np.min( rgal[cgr] ) )
                        grID[:,1][cgr] = 1 # satellite
                        indxC=indx[cgr][cCen][0]
                        grID[indxC,1]=2 # central
#
#
pGroup2=np.ones(ngal2) ; pCentral2=np.ones(ngal2)
pGroup2[ (grID[:,0]!=0) ]  = 1.
pCentral2[ (grID[:,1]==1) ]=0.
#
fout1='hist_Pgroup_rvir_%.1f_Dz_%.3f.png'%(r_vir,Deltaz)
plt.hist(pGroup2,bins=30,color='red',alpha=0.5,label='$P_{\\rm group}$ rough finder')
plt.suptitle('R$_{\\rm vir}=%.1f$ Mpc, $\Delta z=%.3f$'%(r_vir,Deltaz))
plt.legend(loc='best')
plt.savefig(fout1,dpi=300)  # We finish 1st historgram on Group character
#
#
plt.clf()
plt.hist(pCentral2,bins=30,color='red',alpha=0.5,label='$P_{\\rm Central}$ rough finder')
plt.suptitle('R$_{\\rm vir}=%.1f$ Mpc, $\Delta z=%.3f$'%(r_vir,Deltaz))
plt.legend(loc='best')
fout2='hist_Pcentral_rvir_%.1f_Dz_%.3f.png'%(r_vir,Deltaz)
plt.savefig(fout2,dpi=300)  # We finish 1st historgram on Group character


