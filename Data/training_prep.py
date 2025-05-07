import pandas as pd
import numpy as np
import glob

# Producing train/validation data
files = sorted(glob.glob("./data/*.csv"))[:50]

dfs = []

for f in files:
    df = pd.read_csv(f)
    df.drop(columns=["timestamp"], inplace=True)
    df.dropna(inplace=True)
    dfs.append(df)

# Concatenate and convert
tensor = pd.concat(dfs, axis=0).to_numpy(dtype=np.float32)

# Normalize along columns (this will have to be done for inputs
# during inference as well)
tensor = (tensor - tensor.mean(axis=0)) / tensor.std(axis=0)

print("Normalized Shape:",tensor.shape)

seq_len = 512
sequences = np.array([
    tensor[i:i+seq_len]
    for i in range(0, len(tensor)-seq_len, 1)
])

print(sequences)

print("Shape:", sequences.shape)  # Should be (samples, features)

np.save("training_data", sequences)

