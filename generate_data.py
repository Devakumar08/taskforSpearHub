import pandas as pd
import numpy as np

# Generate synthetic manufacturing dataset
np.random.seed(42)
num_samples = 100

data = {
    "Machine_ID": [f"Machine_{i}" for i in range(1, num_samples + 1)],
    "Temperature": np.random.uniform(50, 150, num_samples),  # Random temp between 50-150
    "Run_Time": np.random.randint(60, 500, num_samples),  # Random run time between 60-500
    "Downtime_Flag": np.random.choice([0, 1], num_samples)  # 0 (No downtime) or 1 (Downtime)
}

# Save as CSV
df = pd.DataFrame(data)
df.to_csv("manufacturing_data.csv", index=False)

print("Synthetic dataset saved as 'manufacturing_data.csv'")
