import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
        

# THE FOLLOWING CODE SIMULATES THE DYNAMICS OF THE GLV MODEL AND COMPUTES THE FRACTION OF STABLE RUNS AS A FUNCTION OF MU,SIGMA
# G. AGUADÉ-GORGORIÓ, 24/12/2023

####################### PARAMETERS AND SIMULATION VALUES ##############################

meanAstart=-1.8
meanAmax=0.5

varAstart=0.0
varAmax=1.6

frac=50

reps=50
S = 50

temps1 = 1000
temps2 = 50

EPS=10.**-20 #Threshold of immigration

SI_threshold = 0.001

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
    varAlist.append(varA)
    
    i=0
    while i<frac:
        
        meanA= meanAstart + (i*(meanAmax-meanAstart)/float(frac))

        A=np.random.normal(meanA, varA, size=(S,S)) #COOP matrix, all interactions uniform between 0 and coop    
        if z==0:
            meanAlist.append(np.mean(A))
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
        
        
    endclock = timer()
    print("Line runtime", endclock - startclock)

################## PLOTS ###############################

#Figure 1: number of states

fig,(ax2)= plt.subplots(1,1,figsize=(15,10))



im2 = ax2.imshow(Frac_cycles, cmap="YlOrBr_r")
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
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')
ax2.set_ylabel('avg competition')
ax2.set_xlabel('avg cooperation')
plt.gca().invert_yaxis()
ax2.set_title("Fraction of cycling simulations/outgrowth")



fig.tight_layout()
nom = f"girat_GLV_heatmap_{S}.png"
plt.savefig(nom, format='png')
plt.close()

