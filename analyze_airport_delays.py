import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
DATA_PATH = "2018.csv"
PLOTS_DIR = "advanced_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def analyze_airport_delays():
    print("Loading data...")
    # Load only necessary columns to save memory
    cols = ['ORIGIN', 'CANCELLED', 'ARR_DELAY', 
            'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 
            'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
    
    df = pd.read_csv(DATA_PATH, usecols=cols)
    
    # Filter out cancelled flights
    df = df[df['CANCELLED'] == 0].copy()
    
    # Fill NaN delay causes with 0 (NaN implies 0 delay or not reported)
    delay_cols = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
    df[delay_cols] = df[delay_cols].fillna(0)
    
    # Identify Top 10 Airports by Volume
    print("Identifying Top 10 Airports...")
    top_10_airports = df['ORIGIN'].value_counts().nlargest(10).index.tolist()
    print(f"Top 10 Airports: {top_10_airports}")
    
    # Filter for Top 10
    df_top10 = df[df['ORIGIN'].isin(top_10_airports)]
    
    # Calculate Average Delay Minutes per Flight for each cause
    # We average over ALL flights (including on-time ones) to see the "expected delay contribution"
    print("Calculating delay statistics...")
    delay_stats = df_top10.groupby('ORIGIN')[delay_cols].mean()
    
    # Sort by total delay to make the plot readable
    delay_stats['Total'] = delay_stats.sum(axis=1)
    delay_stats = delay_stats.sort_values('Total', ascending=False).drop(columns='Total')
    
    # Plotting
    print("Generating plot...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Create Stacked Bar Chart
    delay_stats.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='viridis', width=0.8)
    
    plt.title('Average Delay Minutes per Flight by Cause (Top 10 Airports)', fontsize=16, pad=20)
    plt.xlabel('Airport', fontsize=12)
    plt.ylabel('Average Minutes of Delay per Flight', fontsize=12)
    plt.legend(title='Delay Cause', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(PLOTS_DIR, "top10_airports_delay_causes.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Print insights
    print("\nSummary of Delay Drivers (Avg Minutes):")
    print(delay_stats)

if __name__ == "__main__":
    analyze_airport_delays()
