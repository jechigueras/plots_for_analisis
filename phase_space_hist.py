import random
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.ticker import StrMethodFormatter

#Calculate event radii
def Radius_values(R,M_BH):
    M_star = 1
    #Stars and BH radii in au
    R1 = R[0] * 0.004650467260962158
    R2 = R[1] * 0.004650467260962158
    R3 =  2*M_BH / 10065.320121972596**2
    #DP radius
    SMBH_Star_Coll_rad = R3
    Star_Coll_rad = (R1+R2)*(1+1e-3)

    Psi_star= (1.47 + np.exp((M_star - 0.669)/0.137) ) / (1. + 2.34 * np.exp((M_star - 0.669)/0.137) )
    Psi_SMBH = (0.80 + 0.26 * (M_BH / 10**6)**0.5)
    Rt_star = R1 * (M_BH/M_star)**(0.333333333333333333333)
    #FTDE radius
    Rt_Full_Star = Rt_star  * Psi_star* Psi_SMBH
    #PTDE radius
    Rt_Partial_Star = 2*Rt_star
    return SMBH_Star_Coll_rad,Star_Coll_rad, Rt_Full_Star,Rt_Partial_Star


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

df = dummy_df(10000, 34234)

#Exclude Double DP
df = df[df['event_ID'] != 'Double DP'].copy()

#Change name for shorter one
df['event_ID'] = df['event_ID'].replace('Single ejection + Single capture', 'Se + Sc')

#Masking to group two events and mark as detectable
mask_ftde = df['event_ID'].isin(['DP + FTDE', 'FTDE + DP'])
df.loc[mask_ftde, 'event_ID'] = 'FTDE + No detectable'
df.loc[mask_ftde, 'Detectable'] = 'Yes'

#Masking to mark events as detectables
mask_detectable = df['event_ID'].isin(['Double FTDE', 'FTDE + PTDE', 'Single FTDE + No detectable'])
df.loc[mask_detectable, 'Detectable'] = 'Yes'

#Divide dataframe in two: detectable, no detectable
df_detectable = df[df['Detectable'] == 'Yes'].copy()
df_no_detectable = df[df['Detectable'] == 'No'].copy()

#Constrain the 'M1' from 360° -> 180° since is simetrical 
df_detectable['M1'] = df_detectable['M1'] % 180
df_no_detectable['M1'] = df_no_detectable['M1'] % 180

#Create the figure and the grid for the histograms
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(
    3, 3,
    width_ratios=[4, 0.8, 0.2],
    height_ratios=[1, 4, 0.5],
    wspace=0, hspace=0
)

ax_scatter = fig.add_subplot(gs[1, 0])
xhist = fig.add_subplot(gs[0, 0])
yhist = fig.add_subplot(gs[1, 1])


order = 10

#Scatter plot
sns.scatterplot(
    data=df_detectable, x='rp2', y='M1', hue='event_ID', edgecolor=None, palette=EVENT_COLORS,
    s=10, ax=ax_scatter, zorder=order-2, legend=True, rasterized=True 
)

#Ploting event radii
event_radius = Radius_values([1,1], 1e6)

PTDE_radius = event_radius[3]
FTDE_radius = event_radius[2]
DP_radius = event_radius[0]

for y, lbl, stl in zip(
    [PTDE_radius, FTDE_radius, DP_radius],
    ['PTDE radius', 'FTDE radius', 'DP radius'],
    ['dashdot', 'dashed', 'solid']
):
    ax_scatter.axvline(x=y, color='black', linestyle=stl, linewidth=2, zorder=order-1, label=lbl)

#ax_scatter.set_xscale('log')
ax_scatter.set_xlabel(r'Initial impact parameter $[au]$')
ax_scatter.set_ylabel(r'Initial binary phase $[\degree]$')
ax_scatter.set_yticks([0, 45, 90, 135, 180])
ax_scatter.xaxis.set_major_formatter(StrMethodFormatter('{x:.5f}'))
ax_scatter.grid(zorder=order-3)
legend = ax_scatter.legend(loc='best', markerscale=3)
legend.set_zorder(order)

#Marginal histograms
for name, group in df_detectable.groupby('event_ID'):
    #Horizontal
    xhist.hist(
        group['rp2'], bins=max_bins_binary(group['rp2']),
        histtype='step', color=EVENT_COLORS.get(name, 'black'),
        density=True
    )
    #Vertical
    yhist.hist(
        group['M1'], bins="auto", orientation='horizontal',
        histtype='step', color=EVENT_COLORS.get(name, 'black'),
        density=True
    )

#xhist.set_xscale('log')
xhist.set_xticks([])
yhist.set_yticks([])
yhist.set_ylim(ax_scatter.get_ylim())
fig.suptitle(f'Detectable event distribution \nTotal: {len(df_detectable)}')

plt.tight_layout()
plt.show()