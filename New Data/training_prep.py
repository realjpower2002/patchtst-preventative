import numpy as np
import glob
import os

# Producing train/validation data from the second dataset
# Path to the second dataset
data_dir = "./data/1/2nd_test/2nd_test"
files = sorted(glob.glob(os.path.join(data_dir, "*")))
print(f"Found {len(files)} files in {data_dir}")

data_arrays = []
for f in files:
    try:
        # Load data from file (tab-separated values with 4 columns, no header)
        data = np.loadtxt(f, delimiter='\t')
        data_arrays.append(data)
    except Exception as e:
        print(f"Error loading file {f}: {e}")

# Concatenate all data
all_data = np.vstack(data_arrays)
print(f"Total data points: {len(all_data)}")

# Normalize along columns
all_data = (all_data - all_data.mean(axis=0)) / all_data.std(axis=0)
print("Normalized Shape:", all_data.shape)

# Create sequences
seq_len = 512
sequences = np.array([
    all_data[i:i+seq_len]
    for i in range(0, len(all_data)-seq_len, 64)  # Use a stride of 64 to create more sequences
])

print(f"Created {len(sequences)} sequences of length {seq_len}")
print("Sequences shape:", sequences.shape)  # Should be (samples, seq_len, features)

# Select first 20,000 samples for training
train_sequences = sequences[:20000]
print("Training data shape:", train_sequences.shape)

# Next 5,000 samples for validation
val_sequences = sequences[20000:25000]
print("Validation data shape:", val_sequences.shape)

print(len(sequences), len(train_sequences), len(val_sequences))

# Save the data
np.save("new_training_data", train_sequences)
np.save("new_validation_data", val_sequences)
print("Data saved to new_training_data.npy and new_validation_data.npy")
