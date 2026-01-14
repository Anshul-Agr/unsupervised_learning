import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def classify_vi_method(file_path):
    print(f" Processing {file_path} ")
    
    try:
        df = pd.read_csv(file_path, header=1) 
        df.columns = ['Time', 'Current', 'Voltage']
        df = df.iloc[1:].astype(float)
      
        df['Time_us'] = (df['Time'] - df['Time'].min()) * 1e6
        df = df.reset_index(drop=True)
    except:
        return

    # DETECTING PULSES (Voltage Trigger)

    V_TRIGGER, I_TRIGGER = 1.0, 50.0 
    df['Active'] = (df['Voltage'] > V_TRIGGER) | (df['Current'] > I_TRIGGER)
    
    # Gap Merging
    df['Edge_Up'] = (df['Active'] & ~df['Active'].shift(1).fillna(False))
    df['Edge_Down'] = (~df['Active'] & df['Active'].shift(1).fillna(False))
    starts = df.index[df['Edge_Up']].tolist()
    ends = df.index[df['Edge_Down']].tolist()
    
    # Merge Logic
    if len(starts) > len(ends): ends.append(df.index[-1])
    if len(ends) > len(starts): starts.insert(0, df.index[0])
    
    merged_starts, merged_ends = [], []
    if starts:
        curr_s, curr_e = starts[0], ends[0]
        for i in range(1, len(starts)):
            if (df.loc[starts[i], 'Time_us'] - df.loc[curr_e, 'Time_us']) < 10.0:
                curr_e = ends[i]
            else:
                merged_starts.append(curr_s)
                merged_ends.append(curr_e)
                curr_s, curr_e = starts[i], ends[i]
        merged_starts.append(curr_s)
        merged_ends.append(curr_e)

    # EXTRACTING V & I SEPARATELY

    pulse_data = []
    for s, e in zip(merged_starts, merged_ends):
        if s == 0 or e == df.index[-1]: continue
        
        slice_df = df.loc[s:e]
        duration = slice_df['Time_us'].max() - slice_df['Time_us'].min()
        if duration < 5.0: continue
        
        pulse_data.append({
            'Start_Idx': s, 'End_Idx': e,
            'Mid_Time': (df.loc[s, 'Time_us'] + df.loc[e, 'Time_us'])/2,
            'Avg_Voltage': slice_df['Voltage'].mean(), # Feature 1
            'Max_Current': slice_df['Current'].max(),  # Feature 2
            'Start_Time': df.loc[s, 'Time_us'],
            'End_Time': df.loc[e, 'Time_us']
        })
        
    stats_df = pd.DataFrame(pulse_data)

    if not stats_df.empty:
        #  2D CLUSTERING (V and I)

        # K-Means on the 2D space (Voltage, Current)
        # Note: Normalizing
        X = stats_df[['Avg_Voltage', 'Max_Current']].values
        
        # Simple Min-Max Scaling for clustering stability
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        
        # 4 clusters: Normal, Arc, Short, Open
        if len(X) >= 4:
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            stats_df['Cluster'] = kmeans.fit_predict(X_norm)
            centers = kmeans.cluster_centers_
            
            # Map Clusters to Names based on their Center location
            # Unscaling centers
            real_centers = centers * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)
            
            cluster_map = {}
            for i, center in enumerate(real_centers):
                v_c, i_c = center[0], center[1]
                
                if i_c < 200: 
                    label = 'Open/Weak' # Low Current
                elif v_c < 0.5:
                    label = 'Short'     # High Current, Low Voltage
                elif v_c < 1.5:
                    label = 'Arc'       # High Current, Med Voltage
                else:
                    label = 'Normal'    # High Current, High Voltage
                cluster_map[i] = label
                
            stats_df['Label'] = stats_df['Cluster'].map(cluster_map)
        else:
            print("Failure to label the groups")


        # Assign Colors
        color_map = {'Short':'red', 'Arc':'orange', 'Normal':'green', 'Open/Weak':'gray'}
        stats_df['Color'] = stats_df['Label'].map(color_map)

        # PLOTTING V-I SCATTER 

        plt.figure(figsize=(8, 6))
        for label, color in color_map.items():
            subset = stats_df[stats_df['Label'] == label]
            plt.scatter(subset['Max_Current'], subset['Avg_Voltage'], 
                        c=color, label=label, s=100, edgecolors='k')
        
        plt.xlabel('Peak Current (A)')
        plt.ylabel('Average Voltage (V)')
        plt.title('V-I Feature Space Classification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('vi_scatter.png')
        plt.show()
        
        # 5. TIME SERIES PLOT
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(df['Time_us'], df['Voltage'], color='forestgreen')
        ax2.plot(df['Time_us'], df['Current'], color='forestgreen')
        
        for _, row in stats_df.iterrows():
            ax1.axvspan(row['Start_Time'], row['End_Time'], color=row['Color'], alpha=0.2)
            ax1.text(row['Mid_Time'], df['Voltage'].max() + 0.2, row['Label'], 
                     color=row['Color'], ha='center', fontsize=9, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig('vi_classification.png')
        plt.show()

if __name__ == "__main__":
    classify_vi_method('Scope_Data.csv')
