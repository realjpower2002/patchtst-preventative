# This generates the 199 embeddings of length 128 corresponding to the
# 199 consecutive 512-length sequences in the original 100,000 "good"
# samples at the start of the training data

from transformers import PatchTSTForPretraining

model = PatchTSTForPretraining.from_pretrained("../checkpoints/checkpoint-35672").eval().to("cuda")

import numpy as np

good_sequences = np.load("../Data/good_data.npy")

import torch

good_dataset = torch.tensor(good_sequences, dtype=torch.float32).to("cuda")

with torch.no_grad():
    outputs = model(
        past_values=good_dataset,  # shape [B, S, C]
        output_hidden_states=True,
        return_dict=True
    )

last_hidden = outputs.hidden_states[-1]

# Mean pool across patches (dim=2)
mean_pooled = last_hidden.mean(dim=2)  # shape: [batch_size, num_channels, hidden_size]

# Optionally, average across channels too if you want one embedding per sample
embeddings = mean_pooled.mean(dim=1)  # shape: [batch_size, hidden_size]

print("Produced embeddings with shape: ",embeddings.shape)