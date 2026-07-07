import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)

# Generate 1,000,000 random points in 3D space bounded between [0, 1]
# Dimensions: 1,000,000 rows, 3 columns (X, Y, Z)
points_count = 1000000
dimensions = 3
data = np.random.uniform(low=0.0, high=1.0, size=(points_count, dimensions))

# Convert to a DataFrame for structured handling and easy exporting
df = pd.DataFrame(data, columns=['X', 'Y', 'Z'])

# Save to a standard, compressed CSV format to keep file size small
output_filename = "UniformDist.csv"
df.to_csv(output_filename, index=False)
print(f"Dataset successfully created and saved as '{output_filename}'!")

# --- Quick verification check ---
print("\nDataset Verification:")
print(f"Total rows: {len(df)}")
print(f"Minimum value: {df.min().min():.4f}")
print(f"Maximum value: {df.max().min():.4f}")
print("\nFirst 5 rows of UniformDist:")
print(df.head())

