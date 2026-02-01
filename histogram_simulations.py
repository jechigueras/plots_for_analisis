import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import Counter


def multi_histo(*histos, labels=None):
    
    #If df don't have name, we name them Set 1, Set 2, ...
    if labels is None:
        labels = [f'Set {i+1}' for i in range(len(histos))]

    #Get the event names of all the dataframes
    all_events = set()
    for df in histos:
        all_events.update(df['event_ID'].unique())
    base_events = sorted(list(all_events))

    histos_processed = []
    for df in histos:
        total_sims = len(df)
        
        #Count the ocurrence of every event and put in a 'ocurrence' column
        counts = df['event_ID'].value_counts().to_frame('occurrence')
        
        #Calculate the event fraction and add it to the dataframe
        counts['fraction'] = counts['occurrence'] / total_sims
        
        #If an event have 0 occurences, then we add as 0
        counts = counts.reindex(base_events).fillna(0).reset_index()
        counts.rename(columns={'index': 'event_ID'}, inplace=True)
        
        histos_processed.append(counts)

    #Bar configuration
    x = np.arange(len(base_events))
    width = 0.8 / len(histos)
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111)

    for i, (df, label) in enumerate(zip(histos_processed, labels)):
        
        #Put the event bars of every set next to each other 
        offset = -width * (len(histos) - 1) / 2 + i * width
        #Plot the fraction of every event, no the total count
        bars = ax.bar(x + offset, df['fraction'], width=width, label=label, alpha=0.8)

        #Put the count of each event at half height of every bar
        for j, bar in enumerate(bars):
            occ = int(df['occurrence'].iloc[j])
            ax.annotate(f'{occ}',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()/2),
                            xytext=(0, 3), # 3 puntos de offset vertical
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=90, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(base_events, rotation=45, ha='right')
    ax.set_ylabel(r'Fraction $dN / N_{total}$', size=13)
    ax.set_title('Comparison of Event Fractions')
    ax.legend()
    plt.tight_layout()
    plt.show()
    

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

dataframe1 = dummy_df(10000, 431)
dataframe2 = dummy_df(10000, 2134)
dataframe3 = dummy_df(10000, 983)
dataframe4 = dummy_df(10000, 71623)
multi_histo(dataframe1,dataframe2,dataframe3,dataframe4, labels=None)
