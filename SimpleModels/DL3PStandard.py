# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:57:07 2022

@author: abd__
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:53:20 2022
@author: Mike

This curve_fit regression routine of Python scipy with the SNe Ia data, as D_L vs redshift 
(z) from Brout et al. 2022, 'The Pantheon+ Analysis: Cosmological Constraints' ApJ 
vol. 938, 110 and the 15 TRGB data from Anand et al. 2022, ApJ 932, 15. This variation of the 
LCDM model has three free parameters: the Hubble constant, Hu, normalised matter density, O_m; 
the cosmological constant, O_L, with 1=O_m+O_L+O_k. O_k is the remainder for spacetime.

"""
print()
print("This is the D_L3PStandard model, a version of the standard model of cosmology.")
print("The correlation is D_L vs. redshift, z, with sinn(x) = x presuming Euclidean geometry.")
print()

# import the data file and the Python 3 libraries
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate as intg
import math
from sklearn.metrics import r2_score
from astropy.stats.info_theory import bayesian_info_criterion

# open data file
with open("TRGB_D_L_DATA.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the top row    
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,0]
ydata = exampledata[:,4]
error = exampledata[:,7]

# initial guesses for the normalized matter density, O_m and dark energy, O_L
O_m = 0.30 #initial guess for matter density
O_L = 0.69 #initial guess for Omega_L

# where t is the "dummy" variable for integration

def integr(x,O_m,O_L):
    return intg.quad(lambda t: 1/(((np.sqrt(((1+t)**3)*O_m + ((1+t)**2)*(1-O_m-O_L) + O_L)))), 0, x)[0]
   
def func2(x, O_m, O_L):
    return np.asarray([integr(xx,O_m,O_L) for xx in x])

# specify the speed of light
litesped = 299793

def func3(x,Hu,O_m,O_L):
    return (litesped*(1+x)/(np.sqrt(np.abs(1-O_m-O_L))*Hu))*(np.sqrt(np.abs(1-O_m-O_L))*func2(x,O_m,O_L))

# guess for the Hubble constant, Hu. No need to input numbers for the initial guesses but use the normalized matter density, O_m, and dark energy, O_L.
init_guess = np.array([70,O_m,O_L])

# allowed range for the three parameters
bnds=([60,0.001,0.001],[80,1.0,1.0])

# fitting the model to the data, note that when choosing absolute_sigma = False the standard deviations (error) are normalized
params, pcov = curve_fit(func3, xdata, ydata,p0 = init_guess, bounds = bnds, sigma = error, absolute_sigma = False)

# extracting and rounding the parameter, Hu, O_m, O_L, O_k values
ans_Hu, ans_O_m, ans_O_L = params
rans_Hu = round(ans_Hu,2)
rans_O_m = round(ans_O_m,4)
rans_O_L = round(ans_O_L,4)
rans_O_k = round(1 - rans_O_m - rans_O_L,4)

# extracting and rounding the estimated standard deviations.
perr = np.sqrt(np.diag(pcov))
SD_Hu, SD_O_m, SD_O_L = perr
rSD_Hu = round(SD_Hu,2)
rSD_O_m = round(SD_O_m,3)
rSD_O_L = round(SD_O_L,3)

# calculating the value of O_k, S.D. and rounding off 
O_k_SD = np.sqrt((SD_O_m)**2 + (SD_O_L)**2)
rSD_O_k = round(O_k_SD,3)

# normalised chisquar is calculated for N=1718 data pairs with P the parameter count (3) as
P=3
N=1718
e=2.71828183
#Calculate the sum of the individual errors squared
SSE = sum((ydata - func3(xdata,ans_Hu,ans_O_m,ans_O_L))**2)

#Calculate chi^2 as per astronomers
newxsqrd = sum(((ydata - func3(xdata,ans_Hu,ans_O_m,ans_O_L))**2)/(error**2))
newxsqrded = round(newxsqrd/(N-P),2)
"""
#Calculate the chi^2 as commonly done
chisqrd = sum(((ydata - func3(xdata,ans_Hu,ans_O_m,ans_O_L))**2)/(func3(xdata,ans_Hu,ans_O_m,ans_O_L)))
normchisqrd = round(chisqrd/(N-P),5)
"""
#The usual method for BIC calculation is
log_SSE = math.log(e,SSE)
small_bic = bayesian_info_criterion(log_SSE, P, N)
rBIC = round(small_bic,2)

# calculation of residuals
residuals = ydata - func3(xdata,ans_Hu,ans_O_m,ans_O_L)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#easy routine for calculating r squared
ycalc = func3(xdata,ans_Hu,ans_O_m,ans_O_L)
R_sqrd = r2_score(ydata, ycalc)
R_square = round(R_sqrd,4)

#Calculation of the weighted F-statistic
SSEw = sum((1/error)*(residuals)**2)
SSM = sum((1/error)*(ydata - np.mean(ydata))**2)
MSR = (SSM - SSEw)/(P)
MSE = SSEw/(N-P)
Fstat = MSR/MSE
rFstat = round(Fstat,1)

#plot of imported data and model
plt.figure(1,dpi=240)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.0,1.8)
plt.ylim(32.0,46.0)
#plt.xscale("linear")
#plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=14)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 4)
plt.plot(xdata, func3(xdata,ans_Hu,ans_O_m,ans_O_L), color = "green", label = "$D_L$3PStandard model")
plt.xlabel("Redshift z", fontsize = 18)
plt.ylabel("$D_L$ (Mpc)", fontsize = 16)
plt.legend(loc='best', fancybox=True, shadow=False)

#print results
print()
print("The calculated Hubble constant and S.D. are: ", rans_Hu, ",",rSD_Hu)
print("The calculated Omega_m and S.D. are: ",rans_O_m, ",",rSD_O_m )
print("The calculated Omega_L and S.D. are: ", rans_O_L,",",rSD_O_L)
print("The calculated Omega_k and S.D. are: ", rans_O_k,",",rSD_O_k)
print()
print('The r\u00b2 is:', R_square)
print('The weighted F-statistic is:', rFstat)
print("The reduced goodness of fit, as per astronomers, \u03C7\u00b2, is: ", newxsqrded)
print("The BIC estimate is: ",rBIC)
print()



#commands to save plots in two different formats
fig.savefig("DL3PStandard.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("DL3PStandard.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)

