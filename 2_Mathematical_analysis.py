import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics as stats
import scipy


# THE FOLLOWING CODE SELECTS A RANDOM SUBSET OF SPECIES AND THEIR INTERACTIONS FROM A RANDOM MATRIX,
# TEST IF THEY FULFILL THE CLIQUES CRITERIA AND EXTRACTS THEIR AGGREGATE AND SPECIES-LEVEL STATISTICAL PROPERTIES 
# G. AGUADÉ-GORGORIÓ, 24/12/2023


#CHOOSE TYPE OF EXTINCTION: 1 RANDOM, 2 PERCEIVED, 3 EXERTED, 4 BOTH

ext_type = 1

gamma_kill = 1.1

S = 50

systems = 100000
reps = 100

####################### PARAMETERS AND SIMULATION VALUES ##############################

meanAstart=-1.3
meanAmax=-0.5

stdAstart=0.3
stdAmax=0.6

# Intraspecies parameters: d

meand = 1.0
stdd = 0.01
d=np.random.normal(meand,stdd,S)



Sprime_pla = []

muprime = []
sigmaprime = []

#mudiff = []
#sigmadiff = []

F_count = 0
FS_count = 0

sys = 0

while sys < systems:
    
    # define a system
    print(100*(sys+1)/systems)
    
    meanA = np.random.uniform(meanAstart,meanAmax)
    stdA = np.random.uniform(stdAstart,stdAmax)
    
    A=np.random.normal(meanA, stdA, size=(S,S))  
    
    meanAreal = np.mean(A)
    stdAreal = np.std(A)
    
    np.fill_diagonal(A,-d)
    
    # explore it a number of times
    repe = 0
    while repe < reps:
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
            #########################################
   
        Aprime = np.delete(np.delete(A,extinct,axis=0),extinct,axis=1)
       
        # solve the system of equations: find x' with A' acting as if extinct species where not present/independent
    
        xprime = np.dot(np.linalg.inv(-Aprime),np.ones(Sprime))
    
        
        # are all elements positive?
    
        feasible = 0
        for spp in range(len(xprime)):
            if xprime[spp]<0:
                feasible +=1
                break
            
        if feasible ==0:
            #print("feasible clique")
            F_count +=1
        
            #find J' of the small system of surivors-only
        
            Jprime = np.zeros((Sprime,Sprime))
        
            effectsprime = np.dot(Aprime,xprime)
        
            for jrow in range(Sprime):
                for jcol in range(Sprime):
                    if jcol==jrow:
                        Jprime[jrow,jcol]=1- d[jrow]*xprime[jrow] + effectsprime[jrow]
                    else:
                        Jprime[jrow,jcol] = xprime[jrow]*Aprime[jrow,jcol]
        
            eigvalsprime, eigvecsprime = np.linalg.eig(Jprime)
        
        
            # 1. is the system reduced-stable (J')
        
            if (np.all(np.real(eigvalsprime) < 0)):
    
                # Store muprime, sigmaprime, Sprime
                #print("J' is Stable, we don't care about migration or J-stability - ok")
                
                offvals = []
            
                for rowprime in range(len(Aprime)):
                    for colprime in range(len(Aprime)):
                        if colprime != rowprime:
                            offvals.append(Aprime[rowprime,colprime])
            
                FS_count +=1
            
                muprima = np.mean(offvals) #mean and std of off-diagonal stuff only, like for A
                sigmaprima = np.std(offvals)
            
                muprime.append(muprima)
                sigmaprime.append(sigmaprima)
                
                Sprime_pla.append(Sprime)
         
                
        repe+=1
    sys+=1

print("systems: ", systems)    
print("runs: ", reps)
print("total: ", reps*systems)
print("feasible sets: ", F_count)
print("feasible, stable sets: ", FS_count)


# Check a condition to decide whether to proceed with plotting or not
#proceed_with_plotting = False  # Set this to True if you want to plot

#if not proceed_with_plotting:
#    exit()  # This terminates the program immediately

#print("proceed to plot")
    
################################################
################################################
################################################



# Create the histogram without plotting
counts, bins, _ = plt.hist(Sprime_pla, bins=30, alpha=0.5, label='Sprime', density=True)

# Calculate mean and variance
bin_centers = 0.5 * (bins[1:] + bins[:-1])  # Compute bin centers
mean = np.sum(bin_centers * counts) / np.sum(counts)  # Calculate mean
variance = np.sum(counts * (bin_centers - mean)**2) / np.sum(counts)  # Calculate variance
plt.title(f'Histogram of S prime\nMean: {mean:.3f}, Variance: {variance:.3f}')
# Output the mean and variance
print(f"Mean: {mean}")
print(f"Variance: {variance}")

# Saving the histogram plot
plt.xlabel('S prime')
plt.ylabel('Frequency')
plt.legend()   
  
nom = "2TEST_Sprime_histogram"
nom = f"{nom}_{ext_type}"  # Append the ext_type to the filename if it's not None
nom += ".png" 
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
variance = np.sum(counts * (bin_centers - mean)**2) / np.sum(counts)  # Calculate varianc#e

# Output the mean and variance
print(f"Mean: {mean}")
print(f"Variance: {variance}")

plt.title(f'Histogram of A prime\nMean: {mean:.3f}, Variance: {variance:.3f}')
plt.hist(A_all, bins=30, alpha=0.5, label='A', density=True)
plt.xlabel('interaction strength')
plt.ylabel('Frequency')
plt.legend()  
  
nom = "3TEST_Aprime_histogram"
nom = f"{nom}_{ext_type}"  # Append the ext_type to the filename if it's not None
nom += ".png" 
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
plt.xlim( 0.5, -1.5)
plt.ylim(0.0, 1.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar(label='Sprime_pla')  # You can set the label for the colorbar here
from scipy.optimize import curve_fit
from scipy import stats

# Check for NaN 
mu_indices = np.isnan(muprime) 
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

nom = "4TEST_muprime_sigmaprime_domain.png" 
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

nom = "8TEST_S_mu"  # Base filename without extension
nom = f"{nom}_{ext_type}"  # Append the ext_type to the filename if it's not None
nom += ".png"
plt.savefig(nom, format='png')
plt.close()


################################################
################################################
################################################

#Figure 5: diffs
from scipy.optimize import curve_fit
from scipy import stats

#plt.figure(figsize=(10, 4))  # Adjust the width and height as needed

# Scatter plot
plt.scatter(mudiff, sigmadiff, c=Sprime_pla, cmap='viridis', alpha=0.1)
plt.xlabel('mu-muprime')
plt.ylabel('sigma-sigmaprime')

# Set x and y axis ranges
plt.xlim( 0.1, -0.4)
plt.ylim(0.0, 0.4)
plt.gca().set_aspect('equal', adjustable='box')


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

nom = "5TEST_differences"
nom = f"{nom}_{ext_type}"  # Append the ext_type to the filename if it's not None
nom += ".png" 
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

nom = "6TEST_muprime_Sprime.png" 
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

nom = "7TEST_sigmaprime_Sprime.png" 
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

nom = "8TEST_S_mu"  # Base filename without extension
nom = f"{nom}_{ext_type}"  # Append the ext_type to the filename if it's not None
nom += ".png"
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
sigma_rand = []#

S_list_s = []
Stest = 6
while Stest < max(Sprime_pla) +1:
    S_list_s.append(Stest)
    if Stest in min_sigmaprime:
        sigmaG.append( np.sqrt(2.0/float(Stest)) * (1+max_muprime[Stest])     )
        sigmaGErr.append( 14.8*np.power(Stest,-1.1) * (1+max_muprime[Stest])   )
    else:
        sigmaG.append( 0     )
        sigmaGErr.append(0   )
    sigma_rand.append(stdAstart - (stdAmax/np.sqrt(Stest)))
    Stest +=1

# Scatter plot
plt.scatter(Sprime_pla, sigmaprime,  alpha=0.1)
plt.plot(S_list_s, sigmaG,  linestyle='--',alpha=0.3, label="sigmaG")
plt.plot(S_list_s, sigmaGErr,  linestyle='--',alpha=0.3, label="sigmaGErr")
plt.plot(S_list_s, sigma_rand,  linestyle='--',alpha=0.3, label="1sigma/sqrt(S')")

plt.xlabel('S')
plt.ylabel('sigma')

plt.axhline(y=stdAstart, color='gray', linestyle='--', label = "low het")
plt.axhline(y=stdAmax, color='black', linestyle='--', label = "high het")

plt.legend()

nom = "9TEST_S_sigma.png" 
plt.savefig(nom, format='png')
plt.close()



################################################
################################################
################################################

#Figure 11: correlations between A' elements (symmetry?)

# Plotting correlations for all matrices in A_full and Aprime_full

for i in range(len(Aprime_full)):
    A = Aprime_full[i]
  
    plt.scatter(A.flatten(), np.transpose(A).flatten(), alpha=0.1)
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
nom = "11TEST_correlations_Aprime.png" 
plt.savefig(nom, format='png')
plt.close()

