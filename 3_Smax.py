import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.special
from scipy.optimize import newton 

# THE FOLLOWING CODE SOLVES THE EQUATION FOR THE MAXIMUM CLIQUE DIVERSITY ACROSS MU AND SIGMA VALUES
# G. AGUADÉ-GORGORIÓ, 24/12/2023

####################### PARAMETERS AND SIMULATION VALUES ##############################

def equation_to_solve(x, mu, sigma):
    # Avoid division by zero or negative values in sqrt
    if x <= 1 or x*(x-1) <= 0:
        return 0
    else:
        return mu + (sigma/np.sqrt(x*(x-1))) - ((x**1.1 / 14.80) * sigma* (1- np.sqrt(1 - ((2/(x*(x-1)-1))* (scipy.special.gamma(x*(x-1)/2) / scipy.special.gamma((x*(x-1)-1)/2))*(scipy.special.gamma(x*(x-1)/2) / scipy.special.gamma((x*(x-1)-1)/2))   )))) + 1

#
##################################################################################

####################### PARAMETERS AND SIMULATION VALUES ##############################

meanAstart=-1.8
meanAmax=0.5

varAstart=0.0
varAmax=1.6

frac=40

reps=40
S = 50

temps1 = 1000
temps2 = 50

EPS=10.**-20 #Threshold of immigration

#PRECISION = 5 #NUMBER OF DECIMAL POINTS CONSIDERED FOR SPECIES ABUNDANCE - AVOID OVER-PRECISE STATES

SI_threshold = 0.01

# DEFINING PARAMETERS:

# Intraspecies parameters: d

meand = 1.0
vard = 0.01
d=np.random.normal(meand,vard,S)


##################################################################################

# Interaction range:

meanAlist = []
varAlist = []

Frac_cycles = np.zeros((frac, frac)) # number of runs that end in non-stable behavior

z=0
while z < frac:

    print("row: ", z+1," of ",frac)
    startclock = timer()
    
    varA = varAstart + (z*(varAmax-varAstart)/float(frac)) + 0.000001
    #varAlist.append(varA)
    
    i=0
    while i<frac:
        
        meanA= meanAstart + (i*(meanAmax-meanAstart)/float(frac))

        A=np.random.normal(meanA, varA, size=(S,S)) #COOP matrix, all interactions uniform between 0 and coop    
        np.fill_diagonal(A,-d)
    
        num_unst = 0
        exp_warning=0
        for j in range(0,reps): #Perform experiment many times
            #print(j," of ",reps)
            # One simulation: 
            def run(S, tmax=temps1, EPS=EPS,**kwargs):
                def eqs(t,x):
                    dx = x*(1+np.dot(A,x)) 
                    dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                    return dx
                #Generate random initial conditions, but summing Xinit:
                x0 = [v for v in np.random.random(S)]           
                sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
                time=sol.t
                trajectories=sol.y
                return time, trajectories
    
            # Simulation: write spp trajectories
            time,trajectories = run(S)
            finalstate = [m for m in trajectories[:,-1]]
                            
            #################### CYCLE CHECK: RUN MORE TIME AND OBSERVE IF STATE IS DIFFERENT #########################
            
            def run(S,tmax=temps2,EPS=EPS,**kwargs):
                def eqs(t,x):
                    dx = x*(1+np.dot(A,x)) 
                    dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                    return dx
                #Solve the system of equations:
                x0 = finalstate           
                sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
                time=sol.t
                trajectories=sol.y
                return time, trajectories
    
            # Simulation: write spp trajectories
            timeplus,trajectoriesplus = run(S)
            finalstateplus = [m for m in trajectoriesplus[:,-1]] 
            
            ###########################################################################################################
            
            # DO WE SEE A NEW STATE, AN ALREADY-SEEN STATE, OR CYCLING/CHAOTIC BEHAVIOUR?
            
            # 1: SEE IF STATE IS STABLE IN TIME
            
            diff = 0.0
            for spp in range(S):
                if abs ( finalstate[spp] - finalstateplus[spp] ) > SI_threshold:
                    diff += 1
                    break
                            
            if diff > 0: # if two species have been deemed different, this state is not stable!
                num_unst+=1
        
          
        
        Frac_cycles[z,i] = num_unst / float(reps)
        
        

        i+=1
    z+=1    






# Interaction range:

meanAlist = []
varAlist = []

S = np.zeros((frac, frac)) # number of runs that end in non-stable behavior

z=0
while z < frac:

    print("row: ", z+1," of ",frac)
    startclock = timer()
    
    varA = varAstart + (z*(varAmax-varAstart)/float(frac))
    
    varAlist.append(varA)
    
    i=0
    while i<frac:
        
        meanA= meanAstart + (i*(meanAmax-meanAstart)/float(frac))
        if z==0:
            meanAlist.append(meanA)
                
        mu_value = meanA
        sigma_value = varA  
        
        if Frac_cycles[z,i] >0.95:
            S[z,i] = 0
        elif mu_value > -1.0 and sigma_value < (0.2*(mu_value+1)):
            S[z,i] = 0
        elif mu_value < -1.0 and sigma_value < (-0.75*(mu_value+1)):
            S[z,i] = 0        
        else:
            try:
                result = newton(equation_to_solve, x0=1.5, args=(mu_value, sigma_value), maxiter=100)
                S[z, i] = result
            except RuntimeError:
                # If Newton's method fails to converge, set the value to 0 or another default
                S[z, i] = 0  # You can change this to any default value       
        i+=1
    z+=1    
        
        
    endclock = timer()
    print("Line runtime", endclock - startclock)

################## PLOTS ###############################

from matplotlib.colors import ListedColormap  # Import ListedColormap


fig,(ax2)= plt.subplots(1,1,figsize=(15,10))

im2 = ax2.imshow(S, cmap="viridis")
stra = ["{:.3f}".format(i) for i in meanAlist]
strb = ["{:.3f}".format(i) for i in varAlist]
# Show all ticks and label them with the respective list entries
ax2.set_xticks(np.arange(len(meanAlist)))
ax2.set_yticks(np.arange(len(varAlist)))
ax2.set_xticklabels(stra)
ax2.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax2.invert_yaxis()
#ax2.invert_xaxis()

# Colorbar with adjusted colormap
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(im2, cax=cax, orientation='vertical')
cbar.set_label('S')

ax2.set_ylabel('sigma')
ax2.set_xlabel('mu')
plt.gca().invert_yaxis()
ax2.set_title("Clique diversity")



fig.tight_layout()
nom = "Smax_viridis.png"
plt.savefig(nom, format='png')
plt.close()

