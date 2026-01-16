#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:41:24 2022

@author: Mike

This curve_fit regression routine of Python scipy, using the distance mag data, converted to 
luminosity distances, D_L, vs expansion factor, from Brout et al. 2022, 'The Pantheon+ Analysis: 
Cosmological Constraints' Astrophys. J. vol. 938, 110 AND the 15 TRGB data, from GS Anand et al. 
2022, 'Comparing Tip of the Red Giant Branch Distance Scales: 'ApJ. vol. 932, 15. The model used 
is the Melia 2012 solution,with only one parameter, the Hubble constant. 
No estimation is possible for the matter density nor dark energy nor space contributions.
"""
print()
print("This is the R_h modeled using luminosity distance vs. the redshift, z.") 
print("Estimates for the matter density and dark energy ")
print("(cosmological constant) cannot be made with this model")
# import data and library files
import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score
from astropy.stats.info_theory import bayesian_info_criterion

# open data file
with open("TRGB_D_L_DATA.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the first row
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,0]
ydata = exampledata[:,4]
error = exampledata[:,7]

# define the constant
litesped = 299792
e = 2.718281

b =[70] # The initial guess for the Hubble constant.

# define the function, where b is the Hubble constant
def func(x,b):
    return ((litesped*(1+x))/b)*np.log(1+x)

# evaluate and plot function
funcdata = func(xdata,b) 

# the lower and upper bounds allowed for the Hubble constant
bnds = (60.0, 100.0)

# curve_fit the model to the data, note that when absolute_sigma = False the errors are "normalized"
params, pcov = curve_fit(func,xdata,ydata,bounds = bnds, sigma = error, absolute_sigma = False)
perr = np.sqrt(np.diag(pcov))
      
# unpacking the Hubble parameter and the estimated standard error
Hubble, = params
Error, = perr

# rounding the above two values to 2 decimal places
normHubble = round(Hubble,2)
normError = round(Error,2)

# calculate the statistical fitness, using N=1702 as the number of data pairs and P=1 as the degree of freedom (paramater count)

#insert some constant values
P=1
N=1718


#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata[1:-1] - func(xdata,Hubble)[1:-1])**2)/(error[1:-1]**2))
#normalised newxsqrded is calculated as
newxsqrded = round(newxsqrd/(N-P),3)
"""
#Calculate the chi^2 according in the common manner
# since the error at the origin is 0 we have to ignore this only to estimate the common goodness of fit, but not the fit itself
chisq = sum((ydata[1:-1] - func(xdata,Hubble)[1:-1])**2/func(xdata,Hubble)[1:-1])
#normalised chisquar is calculated as
normchisquar = round((chisq/(N-P)),2) #rounded to 3 digits
"""
#The usual method for BIC calculation is
SSE = sum((ydata - func(xdata,Hubble))**2)
log_SSE = math.log(e,SSE)
small_bic = bayesian_info_criterion(log_SSE, P, N)
rBIC = round(small_bic,2)

#calculation of residuals
residuals = ydata - func(xdata,Hubble)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#easy routine for calculating r squared
ycalc = func(xdata,Hubble)
R_sqrd = r2_score(ydata, ycalc)
R_square = round(R_sqrd,4)

#Calculation of the weighted F-statistic
SSEw = sum((1/error)*(residuals)**2)
SSM = sum((1/error)*(ydata - np.mean(ydata))**2) 
MSR = (SSM - SSEw)/(P)
MSE = SSEw/(N-P)
Fstat = MSR/MSE
rFstat = round(Fstat,1)

#plot of data and results
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.3,1.0)
plt.ylim(0.0,15000)
#plt.xscale("linear")
#plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=8)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.xlabel("Redshift z ", fontsize=18)
plt.ylabel("Luminosity distance (Mpc)", fontsize=18)
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 5)
plt.plot(xdata, funcdata, color = "orange", label = "$D_LR_h$ model")
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

#print results
print()
print("The estimated Hubble constant is:", normHubble)
print("The S.D. of the Hubble constant is", normError)
print()
print('The r\u00b2 is:', R_square)
print('The weighed Fstat is:', rFstat)
print("And reduced goodness of fit, according to astronomers, \u03C7\u00b2 estimate is:", newxsqrded)
#print("And the common reduced goodness of fit \u03C7\u00b2 estimate is:", normchisquar)
print("The BIC estimate is:",rBIC)
print()

#Routines to save figues in eps and pdf formats
fig.savefig("DLR_h.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("DLR_h.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)
