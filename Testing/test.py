# We get the "distance" in the form of quantization error from the knn model
# produced via the embeddings from the transformer.

# A heuristic can then be used to determine the ideal "safe" machine operating
# bounds

from transformers import PatchTSTForPretraining

import numpy as np

import torch

from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import normalize

import os

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

from torch.utils.data import DataLoader, TensorDataset


model = PatchTSTForPretraining.from_pretrained("../checkpoints/checkpoint-35672").eval().to("cuda")

if not os.path.exists("./embeddings.npy"):

    bad_sequences = np.load("../Data/bad_data.npy")

    BATCH_SIZE = 64  # Try small value like 16–64 depending on GPU

    dataset = TensorDataset(torch.tensor(bad_sequences, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    all_embeddings = []

    with torch.no_grad():
        for (batch,) in dataloader:
            batch = batch.to("cuda")
            out = model(
                past_values=batch,
                output_hidden_states=True,
                return_dict=True
            )
            last_hidden = out.hidden_states[-1]
            pooled = last_hidden.mean(dim=2).mean(dim=1)  # [B, D]
            all_embeddings.append(pooled.cpu())  # move off GPU

    
    bad_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    np.save("embeddings.npy", bad_embeddings)

    # # Mean pool across patches (dim=2)
    # mean_pooled = last_hidden.mean(dim=2)  # shape: [batch_size, num_channels, hidden_size]

    # # Optionally, average across channels too if you want one embedding per sample
    # bad_embeddings = mean_pooled.mean(dim=1)  # shape: [batch_size, hidden_size]

    print("Produced embeddings with shape: ",bad_embeddings.shape)

else:
    bad_embeddings = np.load("./embeddings.npy")


# ~~~ KNN PORTION ~~~

import faiss

# For each embedding, we determine which of the knn "neurons" it is closest
# to, and find the quantization error

good_embeddings = np.load("../Embeddings/embeddings.npy")

good_embeddings = good_embeddings
bad_embeddings = bad_embeddings

good_embeddings = normalize(good_embeddings, norm='l2')
bad_embeddings = normalize(bad_embeddings, norm="l2")

print(good_embeddings.shape)
print(bad_embeddings.shape)

# Get the distance to the nearest 3 and then find 
neighbors = NearestNeighbors(n_neighbors=3, metric='cosine').fit(good_embeddings)

distances, indices = neighbors.kneighbors(bad_embeddings)

print("Distances : ", distances.mean(axis=1))

print(distances.mean(axis=1).size)

import pickle as pkl

with open("./distances.pkl", "wb") as file:
    pkl.dump(distances.mean(axis=1), file)

# For each bad point, compute the feature-wise squared difference to its neighbors
contributions = np.zeros((bad_embeddings.shape[0], bad_embeddings.shape[1]))  # shape: (N_bad, D)

for i, (bad_vec, neighbor_idxs) in enumerate(zip(bad_embeddings, indices)):
    neighbors = good_embeddings[neighbor_idxs]  # shape: (3, D)
    diffs = (neighbors - bad_vec)**2  # shape: (3, D)
    contributions[i] = diffs.mean(axis=0)  # mean squared diff per feature