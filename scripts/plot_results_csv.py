
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_csv(file_path):
    print(f"Processing {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    df = pd.read_csv(file_path)

    # 2. Data Processing
    # Ensure columns exist
    required_cols = ['LR', 'WD', 'Accuracy', 'BS_Type']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Column {col} missing in {file_path}")
            return

    df['LR'] = df['LR'].astype(float)
    df['WD'] = df['WD'].astype(float)
    # Calculate x-axis: LR * WD
    df['LR_x_WD'] = df['LR'] * df['WD']

    # 3. Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Scatter plot
    scatter = sns.scatterplot(
        data=df, 
        x='LR_x_WD', 
        y='Accuracy', 
        hue='BS_Type', 
        style='BS_Type',
        palette={'small': '#e74c3c', 'large': '#3498db'}, # Custom colors: red/blue
        s=150, # Point size
        alpha=0.8, # Transparency
        edgecolor='k' # Edge color
    )

    # 4. Key setting: Log scale for X-axis
    plt.xscale('log')

    # 5. Add labels and decorations
    base_name = os.path.basename(file_path)
    title_suffix = base_name.replace("results_", "").replace(".csv", "")
    plt.title(f'Relationship between (LR $\\times$ WD) and Accuracy ({title_suffix})', fontsize=16)
    plt.xlabel('LR $\\times$ WD (Log Scale)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(title='Batch Size Type', fontsize=12, title_fontsize=12)

    # Optional: Annotate points
    for line in range(0, df.shape[0]):
        plt.text(
            df.LR_x_WD[line], 
            df.Accuracy[line]+0.002, 
            f"{df.LR[line]:.0e}x{df.WD[line]:.0e}", 
            horizontalalignment='center', 
            size='small', 
            color='black', 
            weight='light',
            rotation=45 
        )

    plt.tight_layout()
    
    # Save to outputs/plots
    output_dir = "outputs/plots"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"accuracy_{title_suffix}.png")
    
    plt.savefig(output_filename)
    print(f"  -> Saved image: {output_filename}")
    plt.close() # Close figure to free memory

if __name__ == "__main__":
    # Define files to process
    files = [
        "outputs/results/results_0shot.csv",
        "outputs/results/results_8shot.csv"
    ]
    
    for f in files:
        # Check if file exists relative to script or simple path
        # Assuming script run from project root, path given is relative to project root
        plot_csv(f)
