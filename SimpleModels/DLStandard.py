#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:53:20 2022
@author: Mike

This curve_fit regression routine of Python scipy, uses 1701 D_L vs redshift (z) data from 
Brout et al. 2022, 'The Pantheon+ Analysis: Cosmological Constraints' ApJ vol. 938, 110 AND 
the 15 TRGB data, from GS Anand et al. 2022, 'Comparing Tip of the Red Giant Branch Distance 
Scales: 'ApJ vol. 932, 15. This variation of the standard (\Omega_{\Lambda} LCDM) 
model has two parameters: Hubble constant, Hu, normalised matter density, O_m; the cosmological 
constant, O_L, is the remainder of information in a universe with presumed Euclidean space geometry. 
This is sometimes termed the standard model of cosmology, Riess et al. 2004, ApJ, vol. 607, p. 665-687 
and Riess et al. 2022, ApJ Lett, vol. 934, p. L7.
"""
print()
print("This is the current standard model presuming Euclidean space geometry, sinn(x) = x.")

print()
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate as intg
#from scipy.stats
#from sklearn.metrics import r2_score
#from astropy.stats.info_theory import bayesian_info_criterion

# open data file
with open("TRGB_D_L_DATA.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the top row    
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,0]
ydata = exampledata[:,4]
error = exampledata[:,7]

# where t is the "dummy" variable during numerical integration
def integr(x,O_m):
    return intg.quad(lambda t: (1/(np.sqrt(((1+t)**2)*(1+O_m*t) - t*(2+t)*(1-O_m)))), 0, x)[0]
    
def func2(x, O_m):
    return np.asarray([integr(xx,O_m) for xx in x]) 

# specify the speed of light
litesped = 299792

#enablining the correlation of distance mag with redshift, z, 
#data by applying a log transformation
def func3(x,Hu,O_m):
    return (litesped*(1+x)/Hu)*(func2(x,O_m))

# guesses for the Hubble constant, Hu, and the normalized matter density, O_m
init_guess = np.array([70,0.30])

# allowed ranges for the two parameters
bnds=([60,0.0001],[80,1.0])

# fitting the model to the data, note that when choosing absolute_sigma = False the standard deviations (error) are normalized
params, pcov = curve_fit(func3, xdata, ydata,p0 = init_guess, bounds = bnds, sigma = error, absolute_sigma = False)

# extracting and rounding the parameter, Hu and O_m, values
ans_Hu, ans_O_m = params
rans_Hu = round(ans_Hu,2)
rans_O_m = round(ans_O_m,3)

# extracting and rounding the estimated standard deviations.
perr = np.sqrt(np.diag(pcov))
SD_Hu, SD_O_m = perr
rSD_Hu = round(SD_Hu,2)
rSD_O_m = round(SD_O_m,3)

#chisquared is calculated for 1718 data pairs with P the parameter count (2) as
P=2
N=1718
e=2.71828183

#Calculate the reduced chi^2 according to astronomers
newxsqrd = sum(((ydata - func3(xdata,ans_Hu,ans_O_m))**2)/(error**2))
newxsqrded = np.round(newxsqrd/(N-P),2)

#A version of the reduced chi^2 allowing for the origin
SSEnumer = sum(((ydata - func3(xdata,ans_Hu,ans_O_m))**2)/(sum(func3(xdata,ans_Hu,ans_O_m))))
red_chi2 = np.round(SSEnumer/(N-P),5)

#calculation of residuals
residuals = ydata - func3(xdata,ans_Hu,ans_O_m)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#routine for calculating weighted r squared
SSEw = sum((1/error)*(residuals)**2)
SSM = sum((1/error)*(ydata - np.mean(ydata))**2) 
R_Sqr = 1-(SSEw/SSM)
R_square = round(R_Sqr,4)


#Calculation of the weighted F-statistic
SSEw = sum((1/error)*(residuals)**2)
SSM = sum((1/error)*(ydata - np.mean(ydata))**2)
MSR = (SSM - SSEw)/(P)
MSE = SSEw/(N-P)
Fstat = MSR/MSE
rFstat = round(Fstat,1)

#plot of imported data and model
plt.figure(1,dpi=240)
plt.rcParams["font.family"] = "MathJax_Main"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.0,1.8)
plt.ylim(32.0,46.0)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=14)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 4)
plt.plot(xdata, func3(xdata,ans_Hu,ans_O_m), color = "green", label = "$D_L$Standard model")
plt.xlabel("Redshift z", fontsize = 18)
plt.ylabel("$D_L$ (Mpc)", fontsize = 16)
plt.legend(loc='best', fancybox=True, shadow=False)

#print results
print()
print("The calculated Hubble constant and S.D. are:", rans_Hu, ",",rSD_Hu)
print("The calculated normalised matter density and S.D. are:",rans_O_m, ",",rSD_O_m )
print()
print('The weighted r\u00b2 is:', R_square)
print('The weighted F-statistic is:', rFstat)
print("The usual version of reduced \u03C7\u00b2 is:", red_chi2)
print("The reduced goodness of fit, as per astronomers, \u03C7\u00b2 is:", newxsqrded)
print()

#commands to save plots in two different formats
fig.savefig("DLStandard.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("DLStandard.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)
