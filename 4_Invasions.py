import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics as stats


# THE FOLLOWING CODE TESTS THE ROLE OF SPECIES INVASIONS AND CLIQUE STABILITY FOR THE NUMERICAL AND MATHEMATICAL SCHEMES
# G. AGUADÉ-GORGORIÓ, 24/12/2023

####################### PARAMETERS AND SIMULATION VALUES ##############################

meanAstart=-1.3
meanAmax=-0.5

stdAstart=0.3
stdAmax=0.6

# Intraspecies parameters: d

meand = 1.0
stdd = 0.01

Sstart = 10
Smax = 100
frac=Smax-Sstart

reps=100

systems = reps

ext_type = 4
gamma_kill = 1.5

##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################

print("MATHS TEST")


S_list = []
fraction_stable_list = []

temps1 = 2000
temps2 = 10

EPS=10.**-10 #Threshold of immigration

#PRECISION = 5 #NUMBER OF DECIMAL POINTS CONSIDERED FOR SPECIES ABUNDANCE - AVOID OVER-PRECISE STATES

SI_threshold = 0.001

# DEFINING PARAMETERS:

# Intraspecies parameters: d

meand = 1.0
vard = 0.01


z=0
while z < frac:

    print("row: ", z+1," of ",frac)
    startclock = timer()
    
    S = Sstart + z
    S_list.append(S)
    
    d=np.random.normal(meand,vard,S)
    
    i=0
    while i<1:
        
        num_unst = 0
        exp_warning=0
        
        for j in range(0,reps): #Perform experiment many times
            #print(j," of ",reps)
            # One simulation: 
            
            meanA = np.random.uniform(meanAstart,meanAmax)
            stdA = np.random.uniform(stdAstart,stdAmax)
            d=np.random.normal(meand,stdd,S)
            A=np.random.normal(meanA, stdA, size=(S,S))  
            np.fill_diagonal(A,-d)
            
            def run(S, tmax=temps1, EPS=EPS,**kwargs):
                def eqs(t,x):
                    dx = x*(1+np.dot(A,x)) 
                    dx[x<=EPS]=0#np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
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
                    dx[x<=EPS]=0#np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
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
                            
            if diff == 0: # the state is stable, let's try to invade it
            
                # for each element that is in the A_IR matrix, test if it can invade.
                
                # select only the xprime state: the survivors
                
                xprime = []
                extinct = []
                remain = []
                for spp in range(S):
                    if finalstate[spp]>0.01:
                        xprime.append(finalstate[spp])
                        remain.append(spp)
                    else:
                        extinct.append(spp)
                               
                A_IR = np.delete(np.delete(A,remain,axis=0),extinct,axis=1)

                #vector of growth rates:
                growth = 1 + np.dot(A_IR,xprime)
                #print(xprime)
                #print(growth)
                if np.any(growth > 0):
                    num_unst += 1 #at least one species will invade: the clique is unstable    
          
             
        
        fraction_stable_list.append( 1 - (num_unst / reps ) )
        
        

        i+=1
    z+=1    
        
        
    endclock = timer()
    print("Line runtime", endclock - startclock)





##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################

    
# CODE TO EXPLORE THE STATISTICS OF STATES UNDER COMP + COOP FOR THE GENERALIZED LOTKA-VOLTERRA MODEL (Barbier et al., 2018)
# G AGUADÉ-GORGORIÓ

####################### PARAMETERS AND SIMULATION VALUES ##############################


print("REAL DYNAMICS")

# Interaction range:

Slist = []

Frac_stable = [] # number of runs that end in non-stable behavior

z=0
while z < frac:

    print("row: ", z+1," of ",frac)
    startclock = timer()
    
    S = Sstart + z
    Slist.append(S)
    
    d=np.random.normal(meand,vard,S)
    
    i=0
    while i<1:
        
        num_unst = 0
        exp_warning=0
        
        for j in range(0,reps): #Perform experiment many times
            #print(j," of ",reps)
            # One simulation: 
            
            meanA = np.random.uniform(meanAstart,meanAmax)
            stdA = np.random.uniform(stdAstart,stdAmax)
            d=np.random.normal(meand,stdd,S)
            A=np.random.normal(meanA, stdA, size=(S,S))  
            np.fill_diagonal(A,-d)
            
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
        
             
        
        Frac_stable.append( 1 - (num_unst / reps ) )
        
        

        i+=1
    z+=1    
        
        
    endclock = timer()
    print("Line runtime", endclock - startclock)
    

print("REAL DYNAMICS m=0")

# Interaction range:

Slist_mzero = []

Frac_stable_mzero = [] # number of runs that end in non-stable behavior

z=0
while z < frac:

    print("row: ", z+1," of ",frac)
    startclock = timer()
    
    S = Sstart + z
    Slist_mzero.append(S)
    
    d=np.random.normal(meand,vard,S)
    
    i=0
    while i<1:
        
        num_unst = 0
        exp_warning=0
        
        for j in range(0,reps): #Perform experiment many times
            #print(j," of ",reps)
            # One simulation: 
            
            meanA = np.random.uniform(meanAstart,meanAmax)
            stdA = np.random.uniform(stdAstart,stdAmax)
            d=np.random.normal(meand,stdd,S)
            A=np.random.normal(meanA, stdA, size=(S,S))  
            np.fill_diagonal(A,-d)
            
            def run(S, tmax=temps1, EPS=EPS,**kwargs):
                def eqs(t,x):
                    dx = x*(1+np.dot(A,x)) 
                    dx[x<=EPS]=0#np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
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
                    dx[x<=EPS]=0#np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
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
        
             
        
        Frac_stable_mzero.append( 1 - (num_unst / reps ) )
        
        

        i+=1
    z+=1    
        
        
    endclock = timer()
    print("Line runtime", endclock - startclock)    

###########################################################################################################

print("RANDOM TESTS")

S_list_test = []
fraction_stable_list_test = []

S = Sstart
while S < Smax:
    
    print(S)
    
    sys = 0
    invasions = 0 #invasions/systems will give the fraction of times a system was invadable
    total_Sprime = 0
    stable_Sprime = 0
    
    while sys < systems:
        
        #1 find a clique
        meanA = np.random.uniform(meanAstart,meanAmax)
        stdA = np.random.uniform(stdAstart,stdAmax)
        d=np.random.normal(meand,stdd,S)
        A=np.random.normal(meanA, stdA, size=(S,S))  
        np.fill_diagonal(A,-d)
        
        repe = 0
        clique = 0
        while repe<reps and clique==0: # all these repetitions are only to try to find a clique!
           
            #########################################
            # INVASIONS SCHEME: WHO GETS OUT MOST OFTEN?
            if ext_type == 1:
            # Randomly extinct species:
           
                dead = np.random.randint(1,S)
                Sprime = S-dead

                # set which species die
                extinct = []
                deadcount = 0
                while deadcount < dead:
                    extinct.append(S-1-deadcount) #always kill the last species, its just random...
                    deadcount+=1#
            elif ext_type == 2:
                perceived = np.dot(A,np.ones(S))
                displaced = [(i + abs(min(perceived))) for i in perceived]
                prob_death = [ min( gamma_kill*(1 - (i/max(displaced))) , 1.0 ) for i in displaced] # species with highest impacts on other will have lowest likelihood of becoming extinct

                dead = []
                for spp in range(S):
                    if np.random.rand()< prob_death[spp]:
                        dead.append(1)
                    else:
                        dead.append(0)
            
                Sprime = S-sum(dead)
            
                # set which species die
                extinct = np.where(np.array(dead) == 1)[0]
            elif ext_type ==3:
                exerted = np.dot(np.ones(S),A)
                displaced = [i + abs(min(exerted)) for i in exerted]
                prob_death = [ min( gamma_kill*(1 - (i/max(displaced))) , 1.0 ) for i in displaced] # species with highest impacts on other will have lowest likelihood of becoming extinct

                dead = []
                for spp in range(S):
                    if np.random.rand()< prob_death[spp]:
                        dead.append(1)
                    else:
                        dead.append(0)
            
                Sprime = S-sum(dead)
            
                # set which species die
                extinct = np.where(np.array(dead) == 1)[0]
            elif ext_type ==4:
                both = np.dot(A,np.ones(S)) + np.dot(np.ones(S),A)
                displaced = [i + abs(min(both)) for i in both]
                prob_death = [ min( gamma_kill*(1 - (i/max(displaced))) , 1.0 ) for i in displaced] # species with highest impacts on other will have lowest likelihood of becoming extinct

                dead = []
                for spp in range(S):
                    if np.random.rand()< prob_death[spp]:
                        dead.append(1)
                    else:
                        dead.append(0)
            
                Sprime = S-sum(dead)
            
                # set which species die
                extinct = np.where(np.array(dead) == 1)[0]
                remain = np.where(np.array(dead) == 0)[0]
            #########################################
           
            Aprime = np.delete(np.delete(A,extinct,axis=0),extinct,axis=1)

            # solve the system of equations: find x' with A' acting as if extinct species where not present
            xprime = np.dot(np.linalg.inv(-Aprime),np.ones(Sprime))


            feasible = 0#
            for spp in range(len(xprime)):
                if xprime[spp]<0:
                    feasible +=1
                    break
            
            if feasible ==0:
                Jprime = np.zeros((Sprime,Sprime))
                effectsprime = np.dot(Aprime,xprime)
                for jrow in range(Sprime):
                    for jcol in range(Sprime):#
                        if jcol==jrow:
                            Jprime[jrow,jcol]=1- 2*d[jrow]*xprime[jrow] + effectsprime[jrow]
                        else:
                            Jprime[jrow,jcol] = xprime[jrow]*Aprime[jrow,jcol]#
        
                eigvalsprime, eigvecsprime = np.linalg.eig(Jprime)
                if (np.all(np.real(eigvalsprime) < 0)):
                    clique = 1
            repe+=1#
            
        #2 compute if there is an invader
        
        # ONLY TEST FOR INVASIONS IF WE HAVE FIND A CLIQUE! ELSE KEEP TESTING OTHER SYSTEMS?
        if clique == 1:
            #A_IR = A[Sprime:S, 0:Sprime] # this only works for the random scheme, where the A' is the first section of A
            # Creating A_IR
            #A_IR_rows = A[extinct]  # Rows with indices defined by extinct
            #non_extinct = list(set(range(S)) - set(extinct))  # Indices NOT in extinct
            #A_IR_columns = A[:, non_extinct]  # Columns without indices in extinct

            # Constructing A_IR using rows from A with indices defined by extinct
            # and columns without indices in extinct
            A_IR = A_IR = np.delete(np.delete(A,remain,axis=0),extinct,axis=1) #A_IR_rows[:, non_extinct]

            #vector of growth rates:
            growth = 1 + np.dot(A_IR,xprime)
            if np.any(growth > 0):
                invasions += 1 #at least one species will invade: the clique is unstable
            
            sys+=1
 
           

    S_list_test.append(S)
    fraction_stable_list_test.append(1 - (invasions/systems))
    S+=1
# Scatter plot

plt.scatter(Slist, Frac_stable, alpha = 0.3, label = "real clique stability")
plt.scatter(Slist_mzero, Frac_stable_mzero, alpha = 0.3, label = "real clique stability m=0")
plt.plot(S_list, fraction_stable_list, alpha=0.8, color="black", linestyle="-", label = "test on real cliques")
plt.plot(S_list_test, fraction_stable_list_test, alpha=0.8, color="gray", linestyle="--", label = "test on random cliques")
plt.xlabel('S')
plt.ylabel('Fraction of Stable')
plt.legend()

nom = "all_extinctions.png"
plt.savefig(nom, format='png')
plt.close()     
    
