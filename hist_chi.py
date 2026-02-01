import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import norm, maxwell, chisquare, chi2

#Calculate the max # of bins that have at least 5 points per bin
def max_bins_binary(data, min_per_bin=5):
    #Convert data to a numpy array
    data = np.asarray(data)
    n = len(data)
    
    #Check if the # of data is less than the minimum
    if n < min_per_bin:
        return 1
    
    low = 1
    high = n // min_per_bin
    best_bins = 1
    
    #Binary search
    while low <= high:
        mid = (low + high) // 2
        
        #Calculate # of points in each bin when bin=mid
        counts, _ = np.histogram(data, bins=mid)
        
        #Check if the # of points in each bin is > 5
        if np.all(counts >= min_per_bin):
            #If higher, update the number of bins  
            best_bins = mid
            low = mid + 1
        else:
            #If lower, use the previus number of bins
            high = mid - 1

    return best_bins

EVENT_COLORS = {
    'FTDE + No detectable': '#dc143c',     
    'Se + Sc': '#393b79', 
    'Double DP': '#20B2AA',
    'Double FTDE': '#ffbb78',            
    'Double single capture':'#8c564b',
    'FTDE + PTDE': '#98df8a', 
    'Single FTDE + No detectable': '#808080',  
}


#Creating dummy dataframe
#------------------------------------------------------------------------------------------#

def dummy_df(N, seed):

    #Initial binary phase
    np.random.seed(seed + 321)
    M1 = np.random.uniform(0,2*np.pi,size = N)
    M1_degrees = np.degrees(M1)

    #Initial impact parameter
    np.random.seed(seed + 654)
    rp2 = np.random.uniform(0, 1.2, size = N)

    #Random distribution of events
    np.random.seed(None)
    events = ['DP + FTDE', 'FTDE + DP', 'Single ejection + Single capture',
          'Double DP', 'Double FTDE', 'Double single capture', 
          'FTDE + PTDE', 'Single FTDE + No detectable']

    event_ID = []
    amount_used = 0
    for i, name in enumerate(events):
        #Last event frequency = what is left to reach N
        if i == len(events) - 1:
            count = N - amount_used
        else:
            #Choose a number between 0 and what if left to reach N
            remaining = N - amount_used
            count = np.random.randint(0, remaining + 1) if remaining > 0 else 0
        
        occurrence = [name] * count
        #Avoid the array to be [[],[],[],...]
        event_ID.extend(occurrence)
        amount_used += count

    datos = {'M1': M1_degrees,
             'rp2': rp2,
             'event_ID': event_ID,
             'Detectable': 'No'
    }

    #Creating the dataframe
    df = pd.DataFrame(datos)
    
    return df

#------------------------------------------------------------------------------------------#

df = dummy_df(10000, 4123)

#Excluding this events from the dataframe
df = df[df['event_ID'] != 'Single ejection + Single capture']

#Data to plot
x = 'rp2'  

#Create the figure and the axes
fig, ax = plt.subplots(figsize=(10, 5))

for i, (name, group) in enumerate(df.groupby('event_ID')):

    #Data to plot 
    data = group[x]
    n_total = len(data)
    
    #Calculate the media and standard deviation
    mu, std = norm.fit(data)
    #Calculate the number of bins
    bins = max_bins_binary(data)

    #Histogram of the data
    obs_freqs, bin_edges = np.histogram(data, bins=bins)
    #Calculate the cumulative density distribution of the data
    cdf_vals = norm.cdf(bin_edges, loc=mu, scale=std)
    #Expected values on each bin
    exp_freqs = n_total * np.diff(cdf_vals)

    #Masking to only take bins with 5 or more point for the expected value
    valid = exp_freqs >= 5
    obs_valid = obs_freqs[valid]
    exp_valid = exp_freqs[valid]
    exp_valid *= obs_valid.sum() / exp_valid.sum()

    #reduced chi-square test
    dof = len(obs_valid) - 3
    chi2_stat = np.sum((obs_valid - exp_valid) ** 2 / exp_valid)
    chi2_red = chi2_stat / dof

    color = EVENT_COLORS.get(name, 'gray')
    ax.hist(data, bins=bins, density=True, histtype='step', color=color, label=name)

    x_fit = np.linspace(data.min(), data.max(), 1000)
    pdf_fit = norm.pdf(x_fit, loc=mu, scale=std)
    ax.plot(x_fit, pdf_fit, color=color, linestyle='--')

    #If you want to put info of each reduced χ² test in the plot uncomment this 

    #y_pos = 0.98 - 0.07* i
    #texto = f'{name}: μ = {mu:.2f}, σ = {std:.2f}, reduced χ² = {chi2_red:.2f}'
    #ax.text(0.01, y_pos, texto, transform=ax.transAxes, color=color,
    #        fontsize=9, ha='left', va='top')

    print(f'{name}: bins = {bins}, χ²_red = {chi2_red:.2f}, μ = {mu:.2f}, σ = {std:.2f}, dof = {dof}')

ax.set_xlabel(r'Impact parameter $[au]$')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.show()