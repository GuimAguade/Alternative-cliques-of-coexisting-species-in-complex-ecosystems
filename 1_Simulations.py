import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics as stats
import scipy

# THE FOLLOWING CODE SIMULATES THE DYNAMICS OF THE GLV MODEL IN THE CLIQUES PHASE
# AND EXTRACTS AGGREGATE AND SPECIES-LEVEL STATISTICAL PROPERTIES OF CLIQUES
# G. AGUADÉ-GORGORIÓ, 24/12/2023
        
####################### PARAMETERS AND SIMULATION VALUES ##############################

meanAstart=-1.3
meanAmax=-0.5

stdAstart=0.3
stdAmax=0.6

frac=50 # number of studied systems: frac*frac

reps=50 # number of times each system is explored (i.c.'s)

S = 50 # number of species

temps1 = 1000 # running time
temps2 = 100 # added time for integration

EPS=10.**-20 #Threshold of immigration

SI_threshold = 0.001 # comparison threshold between species

# Intraspecies parameters: d

meand = 1.0 # self regulation, diagonal of the interaction matrix
stdd = 0.01
d=np.random.normal(meand,stdd,S)


##################################################################################

# Interaction range:

meanAlist = []
stdAlist = []

Frac_cycles = np.zeros((frac, frac)) # number of runs that end in non-stable behavior
Sprime_list  = np.zeros((frac, frac)) # number of survivors in final state

Sprime_pla = [] # number of species S' 

muprime = [] # mean interaction mu'
sigmaprime = [] # std deviation of interactions sigma'
 
mudiff = [] # mu - mu'
sigmadiff = [] # sigma - sigma'

Aprime_all = [] # A'_ij as list
A_all = [] #A_ij as list

A_full = [] #A_ij in matrix form
Aprime_full = [] # A'_ij in matrix form

E_MB = [] # expected interaction patterns

z=0
while z < frac:

    print("row: ", z+1," of ",frac)
    startclock = timer()
    
    stdA = stdAstart + (z*(stdAmax-stdAstart)/float(frac)) 
    stdAlist.append(stdA)
    
    i=0
    while i<frac:
        
        meanA= meanAstart + (i*(meanAmax-meanAstart)/float(frac))

        A=np.random.normal(meanA, stdA, size=(S,S))  
        for a in range(S):
            for b in range(S):
                A_all.append(A[a,b])
        A_full.append(A)  
        if z==0:
            meanAlist.append(np.mean(A))
        meanAreal = np.mean(A)
        stdAreal = np.std(A)            
        
        np.fill_diagonal(A,-d)
        
        num_unst = 0
        Sprimetotal = 0
        for j in range(0,reps): #Perform experiment many times
            #print(j," of ",reps)
            # One simulation: 
            def run(S, tmax=temps1, EPS=EPS,**kwargs):
                def eqs(t,x):
                    dx = x*(1+np.dot(A,x)) 
                    dx[x<=EPS]=0 #np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
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
                    dx[x<=EPS]=0 #np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
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
        
            else:
                Sprime = 0 
                xzeros = []
                xprime_star = []
                for spp in range(S):
                    if finalstate[spp]>0.01:#SI_threshold: # if abundance is high enough, consider as survivor
                        Sprime+=1
                        Sprimetotal+=1
                        xzeros.append(finalstate[spp])
                        xprime_star.append(finalstate[spp])
                    else:
                        xzeros.append(0)    
                
                extinct = np.where(np.array(xzeros) == 0)[0]
                Aprime = np.delete(np.delete(A,extinct,axis=0),extinct,axis=1)
                
                # Compute the mean and standard deviation of non-zero  elements
                offvals = []
                               
                xbar = (1+ (meanAreal*(sum(xprime_star))) ) /(1+meanAreal)
                
                for rowprime in range(len(Aprime)):
                    for colprime in range(len(Aprime)):
                        if colprime != rowprime:
                            Aprime_all.append(Aprime[rowprime,colprime])
                            offvals.append(Aprime[rowprime,colprime])
                            
                            suma = 0
                            for gamma in range(len(xprime_star)):
                                if gamma != rowprime:
                                    suma += xprime_star[gamma]*xprime_star[gamma]
                                    
                            E_MB.append(meanAreal + ( (1+meanAreal)*( (xprime_star[rowprime] - xbar)* xprime_star[colprime] / suma) ) )
                
                muprima = np.mean(offvals) #mean and std of off-diagonal stuff only
                sigmaprima = np.std(offvals)
                
                muprime.append(muprima)
                sigmaprime.append(sigmaprima)
                
                mudiff.append(meanAreal-muprima)
                sigmadiff.append(stdAreal - sigmaprima)
                
                #np.fill_diagonal(Aprime, np.zeros(Sprime)) #erase diagonal information, so that it doesn't appear as correlation
                Aprime_full.append(Aprime)
                
                Sprime_pla.append(Sprime)
                
        Frac_cycles[z,i] = num_unst / float(reps)
        Sprime_list[z,i] = Sprimetotal / float(reps)
        

        i+=1
    z+=1    
        
        
    endclock = timer()
    print("Line runtime", endclock - startclock)

################## PLOTS ###############################




# Create the histogram without plotting
counts, bins, _ = plt.hist(Sprime_pla, bins=30, alpha=0.5, label='Sprime', density=True)

# Calculate mean and variance
bin_centers = 0.5 * (bins[1:] + bins[:-1])  # Compute bin centers
mean = np.sum(bin_centers * counts) / np.sum(counts)  # Calculate mean
variance = np.sum(counts * (bin_centers - mean)**2) / np.sum(counts)  # Calculate variance

# Output the mean and variance
print(f"Mean: {mean}")
print(f"Variance: {variance}")
plt.title(f'Histogram of S prime\nMean: {mean:.3f}, Variance: {variance:.3f}')
# Saving the histogram plot
plt.xlabel('S prime')
plt.ylabel('Frequency')
plt.legend()  
nom = "2_Sprime_histogram.png" 
plt.savefig(nom, format='png')
plt.close()

################################################
################################################
################################################
################################################
################################################
################################################
#Figure 3: distributions of A and Aprime

# Create the histogram without plotting
counts, bins, _ = plt.hist(Aprime_all, bins=30, alpha=0.5, label='Aprime', density=True)

# Calculate mean and variance
bin_centers = 0.5 * (bins[1:] + bins[:-1])  # Compute bin centers
mean = np.sum(bin_centers * counts) / np.sum(counts)  # Calculate mean
variance = np.sum(counts * (bin_centers - mean)**2) / np.sum(counts)  # Calculate variance

# Output the mean and variance
print(f"Mean: {mean}")
print(f"Variance: {variance}")


plt.hist(A_all, bins=30, alpha=0.5, label='A', density=True)

plt.xlabel('interaction strength')
plt.ylabel('Frequency')
plt.legend()  
plt.title(f'Histogram of A prime\nMean: {mean:.3f}, Variance: {variance:.3f}')
nom = "3_Aprime_histogram.png" 
plt.savefig(nom, format='png')
plt.close()


#########################################################
#########################################################
#########################################################
#Figure 5: diffs
from scipy.optimize import curve_fit
from scipy import stats

#plt.figure(figsize=(10, 4))  # Adjust the width and height as needed

# Scatter plot
plt.scatter(mudiff, sigmadiff, c=Sprime_pla, cmap='viridis', alpha=0.1)
plt.xlabel('mu-muprime')
plt.ylabel('sigma-sigmaprime')

# Set x and y axis ranges
#plt.xlim( 0.1, -0.4)
#plt.ylim(0.0, 0.4)
#plt.gca().set_aspect('equal', adjustable='box')


# Plot horizontal dashed line at y = -0.5
plt.axhline(y=-0.0, color='gray', linestyle='--', linewidth=0.5,label='y = -0.5')
plt.axvline(x=-0.0, color='gray', linestyle='--', linewidth=0.5,label='y = -0.5')


# Set x and y axis ranges
plt.xlim( -1.5,1.5)
plt.ylim(-0.8, 0.8)

plt.colorbar(label='Sprime_pla')  # You can set the label for the colorbar here


# Check for NaN 
mu_indices = np.isnan(mudiff) 
sigma_indices = np.isnan(sigmadiff) 

# Filter valid indices
validmu_indices = np.where(~mu_indices)[0]  # Get indices where invalid_indices is False
validsigma_indices = np.where(~sigma_indices)[0]  # Get indices where invalid_indices is False

# Convert valid indices to lists
validmu_indices = validmu_indices.tolist()
validsigma_indices = validsigma_indices.tolist()

# Filter arrays using valid indices (as lists)
mudiff_clean = [mudiff[i] for i in validmu_indices]
sigmadiff_clean = [sigmadiff[i] for i in validsigma_indices]

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(mudiff_clean, sigmadiff_clean)

# Plot the regression line
plt.plot(mudiff_clean, slope * np.array(mudiff_clean) + intercept, color='firebrick', linestyle="--", alpha = 0.5, label='Regression line')

# Display numerical values as text on the plot
plt.text(-1.5,-0.5, f'Slope: {slope:.4f}\nIntercept: {intercept:.4f}\nR: {r_value:.4f}', fontsize=10)

print("figure Delta: regression m,b,R")

# Print the values in the terminal
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")

# Calculate covariance and correlation
covariance = np.cov(mudiff_clean, sigmadiff_clean)[0, 1]
correlation = np.corrcoef(mudiff_clean, sigmadiff_clean)[0, 1]

# Display covariance and correlation as text on the plot
plt.text(-0.5, -0.5, f'Covariance: {covariance:.4f}\nCorrelation: {correlation:.4f}', fontsize=10)

print("figure Delta: cov and corr")

# Print the values in the terminal
print(f"Covariance: {covariance:.4f}")
print(f"Correlation: {correlation:.4f}")

#plt.gca().set_aspect('equal', adjustable='box')
nom = "5_differences.png" 
plt.savefig(nom, format='png')
plt.close()

##########################################################################


Figure 12: 3D

# Calculate function values
def custom_function(Sprime_pla, muprime):
    return (np.sqrt(2 * Sprime_pla) * (1 + muprime)) / (1 + Sprime_pla)

# Creating a meshgrid for surface plotting
Sprime_pla_mesh, muprime_mesh = np.meshgrid(Sprime_pla, muprime)
function_values = custom_function(Sprime_pla_mesh, muprime_mesh)

# Plotting
fig = plt.figure()#
ax = fig.add_subplot(111, projection='3d')

# Plot the function surface
ax.plot_surface(Sprime_pla_mesh, muprime_mesh, function_values, alpha=0.1)

# Scatter plot for sigmaprime
ax.scatter(Sprime_pla, muprime, sigmaprime, color='red')

# Set labels and title
ax.set_xlabel('Sprime_pla')
ax.set_ylabel('muprime')
ax.set_zlabel('sigmaprime')
ax.set_title('Function Surface and sigmaprime Scatter Plot')
plt.show()

nom = "12_S_mu_sigma.png" 
plt.savefig(nom, format='png')
plt.close()


################################################
################################################
################################################

#Figure 1: stable?

fig,(ax2)= plt.subplots(1,1,figsize=(15,10))



im2 = ax2.imshow(Frac_cycles, cmap="YlOrBr_r")
stra = ["{:.3f}".format(i) for i in meanAlist]
strb = ["{:.3f}".format(i) for i in stdAlist]
# Show all ticks and label them with the respective list entries
ax2.set_xticks(np.arange(len(meanAlist)))
ax2.set_yticks(np.arange(len(stdAlist)))
ax2.set_xticklabels(stra)
ax2.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")#
ax2.invert_yaxis()
ax2.invert_xaxis()
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')
ax2.set_ylabel('sigma')
ax2.set_xlabel('mu')
plt.gca().invert_yaxis()
ax2.set_title("Fraction of cycling simulations/outgrowth")

fig.tight_layout()
nom = "1_fluctuations_S50_hiprecision.png" 
plt.savefig(nom, format='png')
plt.close()

################################################
################################################
################################################
############################################
################################################
################################################


from scipy.stats import linregress

# Scatter plot
plt.scatter(E_MB, Aprime_all, color="firebrick", alpha=0.01, label='E_MB vs Aprime_all')

# Linear regression
m, b, r_value, _, _ = linregress(E_MB, Aprime_all)
plt.plot(np.array(E_MB), m * np.array(E_MB) + b, color='firebrick', linestyle="--", alpha = 0.5, label='Regression line')  # Plot regression line

# Plot x=y line
plt.plot(E_MB, E_MB, color='black', linestyle='-', alpha=0.8, label='x=y')

# Plot settings
plt.xlabel('Theory')
plt.ylabel('Observation')

print("figure 1: MB correlation m,b,R")

# Annotate text with slope, intercept, and R-squared
# Print slope, intercept, and R-squared values in the terminal
print(f"Slope (m): {m:.3f}")
print(f"Intercept (b): {b:.3f}")
print(f"R-squared: {r_value**2:.3f}")

# Plot horizontal dashed line at y = -0.5
plt.axhline(y=-0.0, color='gray', linestyle='--', linewidth=0.5,label='y = -0.5')
plt.axvline(x=-0.0, color='gray', linestyle='--', linewidth=0.5,label='y = -0.5')



# Show plot
nom = "0_MB_patterns2.png" 
plt.savefig(nom, format='png')
plt.close()


################################################
################################################
################################################

#MB FIGURE 1B - RUNNING

from scipy.stats import norm

# Calculate running average
window_percentage = 0.1
window_size = int(len(E_MB) * window_percentage)

y_running_avg = []
for i in range(len(E_MB)):
    lower_bound = max(0, i - window_size // 2)
    upper_bound = min(len(E_MB), i + window_size // 2)
    mean_y = np.mean(Aprime_all[lower_bound:upper_bound])
    y_running_avg.append(mean_y)

# Sort data based on y_running_avg
sorted_indices = sorted(range(len(y_running_avg)), key=lambda k: y_running_avg[k])
E_MB_sorted = [E_MB[i] for i in sorted_indices]
Aprime_all_sorted = [Aprime_all[i] for i in sorted_indices]
y_running_avg_sorted = sorted(y_running_avg)

# Scatter plot of measured values against theoretical values
plt.scatter(E_MB_sorted, Aprime_all_sorted, alpha=0.1, label='Measured vs Theoretical')

# Plot running average
plt.scatter(E_MB_sorted, y_running_avg_sorted, color='firebrick', alpha = 0.1, label='Running Average (10%)')


# Linear regression
m, b = np.polyfit(E_MB_sorted, y_running_avg_sorted,  1)  # Perform linear regression (fit a 1st-degree polynomial)
plt.plot(np.array(E_MB_sorted), m * np.array(E_MB_sorted) + b, color='firebrick', linestyle="--", alpha = 0.5, label='Regression line')  # Plot regression line

# Plot x=y line
plt.plot(E_MB_sorted, E_MB_sorted, color='black', linestyle='-', alpha=0.8, label='x=y')
# Plot horizontal dashed line at y = -0.5
plt.axhline(y=-0.5, color='black', linestyle='--', label='y = -0.5')
# Plot settings
plt.xlabel('Theoretical (E_MB)')
plt.ylabel('Measured (Aprime_all)')
plt.legend()
plt.title('Comparison of Measured vs Theoretical Values')

# Show plot
nom = "0b_MB_patterns_running.png"
plt.savefig(nom, format='png')
plt.close()


#Figure 2: number of species

fig,(ax2)= plt.subplots(1,1,figsize=(15,10))


im2 = ax2.imshow(Sprime_list, cmap="YlOrBr_r")
stra = ["{:.3f}".format(i) for i in meanAlist]
strb = ["{:.3f}".format(i) for i in stdAlist]
# Show all ticks and label them with the respective list entries
ax2.set_xticks(np.arange(len(meanAlist)))
ax2.set_yticks(np.arange(len(stdAlist)))
ax2.set_xticklabels(stra)
ax2.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax2.invert_yaxis()
ax2.invert_xaxis()
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')
ax2.set_ylabel('sigma')
ax2.set_xlabel('mu')
plt.gca().invert_yaxis()
ax2.set_title("Clique size (Sprime)") 
        
fig.tight_layout()#
nom = "2_clique_sizeS50.png" 
plt.savefig(nom, format='png')
plt.close()

################################################
################################################
################################################


#Figure 4: muprime, sigmaprime - new domain?

# Scatter plot
plt.scatter(muprime, sigmaprime, c=Sprime_pla, cmap='viridis', alpha=0.1)
plt.xlabel('mu')
plt.ylabel('sigma')

# Set x and y axis ranges
plt.xlim( 0.5, -1.5)#
plt.ylim(0.0, 1.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar(label='Sprime_pla')  # You can set the label for the colorbar here

from scipy.optimize import curve_fit
from scipy import stats

# Check for NaN 
mu_indices = np.isnan(muprime) #
sigma_indices = np.isnan(sigmaprime) 

# Filter valid indices
validmu_indices = np.where(~mu_indices)[0]  # Get indices where invalid_indices is False
validsigma_indices = np.where(~sigma_indices)[0]  # Get indices where invalid_indices is False

# Convert valid indices to lists
validmu_indices = validmu_indices.tolist()
validsigma_indices = validsigma_indices.tolist()

# Filter arrays using valid indices (as lists)
muprime_clean = [muprime[i] for i in validmu_indices]
sigmaprime_clean = [sigmaprime[i] for i in validsigma_indices]

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(muprime_clean, sigmaprime_clean)

# Plot the regression line
plt.plot(muprime_clean, slope * np.array(muprime_clean) + intercept, color='red', label='Regression line')

# Display numerical values as text on the plot
plt.text(0.25,1.0, f'Slope: {slope:.4f}\nIntercept: {intercept:.4f}\nR: {r_value:.4f}', fontsize=10)

# Calculate covariance and correlation
covariance = np.cov(muprime_clean, sigmaprime_clean)[0, 1]
correlation = np.corrcoef(muprime_clean, sigmaprime_clean)[0, 1]
# Display covariance and correlation as text on the plot
plt.text(-0.75, 1.0, f'Covariance: {covariance:.4f}\nCorrelation: {correlation:.4f}', fontsize=10)

nom = "4_muprime_sigmaprime_domain.png" 
plt.savefig(nom, format='png')
plt.close()

################################################
################################################
################################################

#Figure 6: S'(mu')

# Scatter plot
plt.scatter(muprime, Sprime_pla, alpha=0.1)
plt.xlabel('mu')
plt.ylabel('S')
# Set x and y axis ranges
plt.xlim( 0.1, -0.4)
plt.ylim(0.0, 0.4)
plt.gca().set_aspect('equal', adjustable='box')

nom = "6_muprime_Sprime.png" 
plt.savefig(nom, format='png')
plt.close()

################################################
################################################
################################################

#Figure 7: S'(sigma')

# Scatter plot
plt.scatter(sigmaprime, Sprime_pla, alpha=0.1)
plt.xlabel('sigma')
plt.ylabel('S')

# Set x and y axis ranges
plt.xlim( 0.1, -0.4)
plt.ylim(0.0, 0.4)
plt.gca().set_aspect('equal', adjustable='box')

nom = "7_sigmaprime_Sprime.png" 
plt.savefig(nom, format='png')
plt.close()

################################################
################################################
################################################

#Figure 8: S'(mu')

# Create a dictionary to store the minimum sigmaprime values for each unique Sprime_pla pair
min_sigmaprime = {}
max_sigmaprime = {}

# Iterate through the elements of Sprime_pla and sigmaprime simultaneously
for i in range(len(Sprime_pla)):
    current_sprime = Sprime_pla[i]
    current_sigmaprime = sigmaprime[i]

    # Check if the current_sprime is already in the dictionary
    if current_sprime in min_sigmaprime:
        # Update the minimum sigmaprime value if necessary
        if current_sigmaprime < min_sigmaprime[current_sprime]:
            min_sigmaprime[current_sprime] = current_sigmaprime
    else:
        # If the current_sprime is not in the dictionary, add it with its corresponding sigmaprime value
        min_sigmaprime[current_sprime] = current_sigmaprime

for i in range(len(Sprime_pla)):
    current_sprime = Sprime_pla[i]
    current_sigmaprime = sigmaprime[i]

    # Check if the current_sprime is already in the dictionary
    if current_sprime in max_sigmaprime:
        # Update the minimum sigmaprime value if necessary
        if current_sigmaprime > max_sigmaprime[current_sprime]:
            max_sigmaprime[current_sprime] = current_sigmaprime
    else:
        # If the current_sprime is not in the dictionary, add it with its corresponding sigmaprime value
        max_sigmaprime[current_sprime] = current_sigmaprime

muG = []
muGErr = []
S_list = []
mu_rand = []
mu_div = []
mu_div_G = []

Stest = 2
while Stest < max(Sprime_pla)+1 : # aquest metode necessita que hagin aparegut cliques de tots els tamanys...
    S_list.append(Stest)
    
    #if Stest in min_sigmaprime:
        #muG.append((min_sigmaprime[Stest] / (np.sqrt(2.0/float(Stest))) ) - 1)
     #   muGErr.append( ( min_sigmaprime[Stest]  / ( 14.8*np.power(Stest,-1.1) ) ) - 1 )
        #mu_div_G.append( ( 1 - (min_sigmaprime[Stest] / ( 14.8*np.power(Stest,-1.1)) ) ) / float(Stest) )
    #else:
        #muG.append( 0)
     #   muGErr.append(0 )
        #mu_div_G.append(0)
    if Stest ==2:
        SD = 0
    else:
        gammas = scipy.special.gamma(Stest*(Stest-1)/2) / scipy.special.gamma((Stest*(Stest-1)-1)/2)
        SD = stdAstart * (1- np.sqrt(1 - ( (2/(Stest*(Stest-1)-1))* (  gammas**2  ) ) ))        
        
    muGErr.append( ( SD  / ( 14.8*np.power(Stest,-1.1) ) ) - 1 )
    mu_rand.append(meanAmax + 2.5*(stdAstart/np.sqrt(Stest*(Stest-1))))
    #mu_div.append(meand / float(Stest))
    Stest +=1

#######################################################################################################################################

#         THE RANDOM TEST          #



reps = 1000

Sprimerand = []
muprimerand = []
sigmaprimerand = []

Sprime_Max = max(Sprime_pla)
Sprime = min(Sprime_pla)

while Sprime < (Sprime_Max+1):
    
    for repe in range(reps):
    
        Sprimerand.append(Sprime)
        # Generate random A within range
    
        meanA = np.random.uniform(meanAstart,meanAmax)
        varA = np.random.uniform(stdAstart,stdAmax)
    
        A=np.random.normal(meanA, varA, size=(S,S))  
        np.fill_diagonal(A,-d)
       
        dead = S - Sprime
    
        # set which species die
    
        extinct = []
        deadcount = 0
        while deadcount < dead:
            extinct.append(S-1-deadcount) #always kill the last species, its just random...
            deadcount+=1

        Aprime = np.delete(np.delete(A,extinct,axis=0),extinct,axis=1)
    
        offvals = []
            
        for rowprime in range(len(Aprime)):
            for colprime in range(len(Aprime)):
                if colprime != rowprime:
                    offvals.append(Aprime[rowprime,colprime])
            
        muprimerand.append(np.mean(offvals))
        sigmaprimerand.append(np.std(offvals))
       
                
    Sprime+=1


#######################################################################################################################################

# Scatter plot
plt.scatter(Sprime_pla, muprime, alpha=0.1, c=sigmaprime, cmap='viridis', label="vals")
plt.colorbar(label='sigma\'')  # You can set the label for the colorbar here


plt.scatter(Sprimerand, muprimerand, marker='x',color='lightgray', alpha = 0.010)
plt.plot(S_list, mu_rand,  linestyle='--',alpha=0.8, color = "gray", label="1sigma/sqrt(S')")
plt.plot(S_list, muGErr,  linestyle='--',alpha=0.8, color = "firebrick", label="mu_critical")
plt.xlabel('S')
plt.ylabel('mu')

# Plot horizontal dashed lines at y = -0.1 and y = -0.3
plt.axhline(y=meanAmax, color='black', linestyle='--')
plt.axhline(y=meanAstart, color='black', linestyle='--')
#plt.legend()

plt.ylim(-1.8, 0.75)

nom = "8_S_mu.png" 
plt.savefig(nom, format='png')
plt.close()

################################################
################################################
################################################

#Figure 9: S'(sigma')

# Create a dictionary to store the minimum sigmaprime values for each unique Sprime_pla pair
max_muprime = {}

for i in range(len(Sprime_pla)):
    current_sprime = Sprime_pla[i]
    current_muprime = muprime[i]

    # Check if the current_sprime is already in the dictionary
    if current_sprime in max_muprime:
        # Update the max murime value if necessary
        if current_muprime > max_muprime[current_sprime]:
            max_muprime[current_sprime] = current_muprime
    else:
        # If the current_sprime is not in the dictionary, add it with its corresponding sigmaprime value
        max_muprime[current_sprime] = current_muprime

sigmaG = []
sigmaGErr = []
sigma_rand = []


S_list_s = []

Stest = 2
while Stest < max(Sprime_pla) +1:
    S_list_s.append(Stest)
    if Stest in min_sigmaprime:
        sigmaG.append( np.sqrt(2.0/float(Stest)) * (1+max_muprime[Stest])     )
        sigmaGErr.append( 14.8*np.power(Stest,-1.1) * (1+max_muprime[Stest])   )
    else:
        sigmaG.append( 0     )
        sigmaGErr.append(0   )
        

    if Stest ==2:
        sigma_rand.append(0)
    else:
        gammas = scipy.special.gamma(Stest*(Stest-1)/2) / scipy.special.gamma((Stest*(Stest-1)-1)/2)
        SD = stdAstart * np.sqrt(1 - ( (2/(Stest*(Stest-1)-1))* (  gammas**2  ) ) )        
        sigma_rand.append( stdAstart - 2*SD)
    Stest +=1
print(S_list_s)
# Scatter plot
plt.scatter(Sprime_pla, sigmaprime,  c=muprime, cmap='viridis', alpha=0.1)
#plt.plot(S_list_s, sigmaG,  linestyle='--',alpha=0.3, label="sigmaG")
#plt.plot(S_list_s, sigmaGErr,  linestyle='--',alpha=0.3, label="sigmaGErr")
plt.plot(S_list_s, sigma_rand,  linestyle='--',alpha=0.8, color = "gray", label="1sigma/sqrt(S')")

plt.xlabel('S')
plt.ylabel('sigma')

plt.axhline(y=stdAstart, color='black', linestyle='--', label = "low het")
plt.axhline(y=stdAmax, color='black', linestyle='--', label = "high het")

plt.legend()

nom = "9_S_sigma.png" 
plt.savefig(nom, format='png')
plt.close()



################################################
################################################
################################################

#Figure 11: correlations between A' elements (symmetry?)

# Plotting correlations for all matrices in A_full and Aprime_full

for i in range(len(Aprime_full)):
    A = Aprime_full[i]
  
    plt.scatter(A.flatten(), np.transpose(A).flatten(), alpha=0.1)#
    plt.title('Correlations between Aprime and AprimeT elements')
    plt.xlabel('Aij')
    plt.ylabel('Aji')
    plt.grid()

# Assuming A_fullarr contains multiple matrices
correlation_coefficients = []  # List to store valid correlation coefficients

for i in range(len(Aprime_full)):
    A = Aprime_full[i]

    # Check if there is variance in the matrix A
    if np.var(A.flatten()) > 1e-9:  # Adjust the threshold as needed
        # Calculate correlation coefficient
        correlation_matrix = np.corrcoef(A.flatten(), np.transpose(A).flatten())
        correlation_coefficient = correlation_matrix[0, 1]

        correlation_coefficients.append(correlation_coefficient)  # Store the coefficient

# Check if there are valid correlation coefficients
if len(correlation_coefficients) > 0:
    # Calculate the average correlation coefficient
    average_coefficient = np.mean(correlation_coefficients)

    # Annotate average correlation coefficient on the plot
    plt.text(0.5, 0.1, f'Average Correlation: {average_coefficient:.8f}',
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
         
plt.gca().set_aspect('equal', adjustable='box')

plt.tight_layout()
nom = "11_correlations_Aprime.png" 
plt.savefig(nom, format='png')
plt.close()


