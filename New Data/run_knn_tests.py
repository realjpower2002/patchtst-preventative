import numpy as np
import torch
import os
import glob
import pickle
import matplotlib.pyplot as plt
from transformers import PatchTSTConfig, PatchTSTForPretraining
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import torch.nn.functional as F
import time

def load_and_preprocess_test_directory(test_dir_path):
    """Load and preprocess test data from a directory in batches to avoid memory issues"""
    files = sorted(glob.glob(os.path.join(test_dir_path, "*")))
    print(f"Found {len(files)} files in {test_dir_path}")
    
    # Process in batches of files
    seq_len = 512
    stride = 128  # Increased stride to reduce number of sequences
    batch_size = 5  # Reduced batch size to lower memory usage
    all_sequences = []
    total_data_points = 0
    
    # First pass to compute global mean and std - one file at a time
    print("Computing global statistics...")
    data_sum = None
    data_sum_sq = None
    data_count = 0
    
    for i, f in enumerate(files):
        try:
            data = np.loadtxt(f, delimiter='\t')
            
            # Update statistics
            if data_sum is None:
                data_sum = np.zeros(data.shape[1])
                data_sum_sq = np.zeros(data.shape[1])
            
            data_sum += np.sum(data, axis=0)
            data_sum_sq += np.sum(data**2, axis=0)
            data_count += data.shape[0]
            
            # Free memory
            del data
            
            if (i+1) % 20 == 0:
                print(f"Processed statistics for {i+1}/{len(files)} files...")
                
        except Exception as e:
            print(f"Error loading file {f}: {e}")
    
    # Compute global mean and std
    if data_count == 0:
        return None
        
    global_mean = data_sum / data_count
    global_var = (data_sum_sq / data_count) - (global_mean**2)
    global_std = np.sqrt(global_var)
    
    # Process files in smaller batches and create sequences incrementally
    print("Processing data in batches...")
    sequence_count = 0
    max_sequences_per_batch = 1000  # Limit memory usage by storing fewer sequences
    
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        
        for f in batch_files:
            try:
                # Process one file at a time to minimize memory usage
                data = np.loadtxt(f, delimiter='\t')
                
                # Normalize using global statistics
                data = (data - global_mean) / global_std
                
                # Create sequences for this file only
                if len(data) > seq_len:
                    file_sequences = np.array([
                        data[j:j+seq_len] 
                        for j in range(0, len(data)-seq_len, stride)
                    ])
                    
                    total_data_points += len(data)
                    sequence_count += len(file_sequences)
                    all_sequences.append(file_sequences)
                    
                    # If accumulated too many sequences, yield a batch and reset
                    if sum(len(s) for s in all_sequences) >= max_sequences_per_batch:
                        yield np.vstack(all_sequences)
                        all_sequences = []
                
                # Free memory
                del data
                
            except Exception as e:
                print(f"Error loading file {f}: {e}")
        
        # Print progress
        print(f"Processed {min(i+batch_size, len(files))}/{len(files)} files...")
    
    # Return any remaining sequences
    if all_sequences:
        yield np.vstack(all_sequences)
        
    print(f"Total data points: {total_data_points}")
    print(f"Created {sequence_count} sequences of length {seq_len}")

def get_embeddings(model, sequences, device, batch_size=16):
    """Generate embeddings for sequences using the model in batches to save memory"""
    model.eval()
    
    # Use smaller batch size to reduce memory usage
    all_embeddings = []
    
    # Handle either a numpy array or a batch generator
    if isinstance(sequences, np.ndarray):
        # Process in smaller batches for large arrays
        max_sequences_per_batch = 500  # Reduced from 1000
        
        for i in range(0, len(sequences), max_sequences_per_batch):
            print(f"Processing embeddings batch {i//max_sequences_per_batch + 1}/{(len(sequences)-1)//max_sequences_per_batch + 1}")
            batch_sequences = sequences[i:i+max_sequences_per_batch]
            
            # Process this mini-batch
            dataset = torch.utils.data.TensorDataset(torch.tensor(batch_sequences, dtype=torch.float32))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
            
            batch_embeddings = []
            
            with torch.no_grad():
                for (batch,) in dataloader:
                    batch = batch.to(device)
                    out = model(
                        past_values=batch,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    last_hidden = out.hidden_states[-1]
                    pooled = last_hidden.mean(dim=2).mean(dim=1)  # [B, D]
                    batch_embeddings.append(pooled.cpu())  # move off GPU
                    
                    # Clear CUDA cache more aggressively
                    if device == torch.device('cuda'):
                        torch.cuda.empty_cache()
            
            if batch_embeddings:
                batch_result = torch.cat(batch_embeddings, dim=0).numpy()
                all_embeddings.append(batch_result)
                
                # Free up memory
                del batch_sequences, dataset, dataloader, batch_embeddings
                if device == torch.device('cuda'):
                    torch.cuda.empty_cache()
    else:
        # If it's a generator, process each yielded batch
        for batch_idx, batch_sequences in enumerate(sequences):
            print(f"Processing generator batch {batch_idx+1}")
            
            # Process this batch from the generator
            dataset = torch.utils.data.TensorDataset(torch.tensor(batch_sequences, dtype=torch.float32))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
            
            batch_embeddings = []
            
            with torch.no_grad():
                for (batch,) in dataloader:
                    batch = batch.to(device)
                    out = model(
                        past_values=batch,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    last_hidden = out.hidden_states[-1]
                    pooled = last_hidden.mean(dim=2).mean(dim=1)  # [B, D]
                    batch_embeddings.append(pooled.cpu())  # move off GPU
                    
                    # Clear CUDA cache more aggressively
                    if device == torch.device('cuda'):
                        torch.cuda.empty_cache()
            
            if batch_embeddings:
                batch_result = torch.cat(batch_embeddings, dim=0).numpy()
                all_embeddings.append(batch_result)
                
                # Free up memory immediately
                del batch_sequences, dataset, dataloader, batch_embeddings
                if device == torch.device('cuda'):
                    torch.cuda.empty_cache()
            
    if not all_embeddings:
        return np.array([])
        
    # Return embeddings without concatenating if in generator mode
    if len(all_embeddings) == 1:
        return all_embeddings[0]
    else:
        return np.vstack(all_embeddings)

def create_knn_model(embeddings, n_neighbors=199):
    """Create a KNN model with normalized embeddings"""
    normalized_embeddings = normalize(embeddings, norm='l2')
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(normalized_embeddings)
    return knn, normalized_embeddings

def calculate_distances(knn, normalized_embeddings, test_embeddings):
    """Calculate distances from test embeddings to KNN neighbors"""
    # Normalize test embeddings
    normalized_test = normalize(test_embeddings, norm='l2')
    
    # Get distances to nearest neighbors
    distances, indices = knn.kneighbors(normalized_test)
    
    # Calculate mean distance for each test point
    mean_distances = distances.mean(axis=1)
    
    return mean_distances, distances, indices

def plot_distances(distances, test_name, output_dir):
    """Plot distances and save to file"""
    plt.figure(figsize=(12, 6))
    plt.plot(distances)
    plt.xlabel("Sequence Index")
    plt.ylabel("Mean Distance to Nearest Neighbors")
    plt.title(f"KNN Distances for {test_name}")
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"knn_distances_{test_name}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Saved plot to {output_file}")

def get_latest_checkpoint(checkpoint_dir):
    """Get the path to the latest checkpoint"""
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Extract checkpoint numbers and find the highest
    checkpoint_numbers = [int(os.path.basename(cp).split('-')[1]) for cp in checkpoints]
    max_index = checkpoint_numbers.index(max(checkpoint_numbers))
    return checkpoints[max_index]

def main():
    # Paths
    checkpoint_dir = "./new_checkpoints"
    output_dir = "./knn_results"
    
    # Limit memory usage
    import gc
    import os
    
    test_dirs = [
        "./data/1/3rd_test/4th_test/txt" if os.path.exists("./data/1/3rd_test/4th_test/txt") else "./data/1/3rd_test",
        "./data/1/2nd_test/2nd_test",  # Now using 2nd_test as the primary dataset
        "./data/1/3rd_test/4th_test/txt" if os.path.exists("./data/1/3rd_test/4th_test/txt") else "./data/1/3rd_test",
        "./data/1/1st_test/1st_test",
    ]
    
    # Try to free up memory at the start
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Get latest checkpoint
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    print(f"Using latest checkpoint: {latest_checkpoint}")
    
    # Configure model
    config = PatchTSTConfig(
        num_input_channels=4,  # Update to 4 features for second dataset
        context_length=512,
        patch_length=16,
        stride=8,
        mask_type="random",
        random_mask_ratio=0.15,
        use_cls_token=False,
    )
    
    # Load model
    model_path = os.path.join(latest_checkpoint, "model.safetensors")
    model = PatchTSTForPretraining(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use safetensors to load the model
    from safetensors.torch import load_file
    model.load_state_dict(load_file(model_path))
    model = model.to(device)
    print(f"Model loaded successfully. Using device: {device}")
    
    # Get "good" sequences to build KNN model
    print("Loading reference data for KNN model...")
    good_embeddings = None
    
    if os.path.exists("./new_training_data.npy"):
        # Load in smaller chunks to reduce peak memory usage
        chunk_size = 5000  # Adjust based on your memory constraints
        good_data = np.load("./new_training_data.npy", mmap_mode='r')
        total_sequences = len(good_data)
        print(f"Found {total_sequences} good sequences in new_training_data.npy")
        
        # Process in chunks
        all_good_embeddings = []
        for chunk_start in range(0, total_sequences, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_sequences)
            print(f"Processing reference chunk {chunk_start+1}-{chunk_end} of {total_sequences}")
            
            # Only load the chunk we need into memory
            chunk_sequences = good_data[chunk_start:chunk_end].copy()
            
            # Get embeddings for this chunk
            chunk_embeddings = get_embeddings(model, chunk_sequences, device)
            all_good_embeddings.append(chunk_embeddings)
            
            # Clean up
            del chunk_sequences
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        # Combine all chunks
        good_embeddings = np.vstack(all_good_embeddings)
        print(f"Generated reference embeddings with shape {good_embeddings.shape}")
    else:
        # Try using the first test directory as reference data
        print("No pre-saved training data found, using first test directory as reference")
        good_generator = load_and_preprocess_test_directory(test_dirs[0])
        
        # Process generator in batches
        all_good_embeddings = []
        for batch_idx, batch_sequences in enumerate(good_generator):
            print(f"Processing reference batch {batch_idx+1}")
            batch_embeddings = get_embeddings(model, batch_sequences, device)
            all_good_embeddings.append(batch_embeddings)
            
            # Clean up
            del batch_sequences
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Combine all batches if we have multiple
        if len(all_good_embeddings) > 1:
            good_embeddings = np.vstack(all_good_embeddings)
        elif len(all_good_embeddings) == 1:
            good_embeddings = all_good_embeddings[0]
        else:
            raise ValueError("Could not load reference data for KNN model")
            
        print(f"Generated reference embeddings with shape {good_embeddings.shape}")
    
    # Create KNN model with 199 neighbors
    print("Creating KNN model with 199 neighbors...")
    knn, normalized_good_embeddings = create_knn_model(good_embeddings, n_neighbors=min(199, len(good_embeddings)-1))
    
    # Free memory after creating KNN model
    del good_embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Process each test directory
    for test_dir in test_dirs:
        test_name = os.path.basename(os.path.dirname(test_dir))
        print(f"\nProcessing test directory: {test_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get generator for test sequences
        test_generator = load_and_preprocess_test_directory(test_dir)
        
        # Process each batch from the generator
        all_mean_distances = []
        batch_idx = 0
        
        for batch_sequences in test_generator:
            print(f"Processing test batch {batch_idx+1} with {len(batch_sequences)} sequences")
            
            # Generate embeddings for this batch
            print("Generating embeddings for test batch...")
            batch_test_embeddings = get_embeddings(model, batch_sequences, device)
            print(f"Generated embeddings with shape {batch_test_embeddings.shape}")
            
            # Calculate distances for this batch
            print("Calculating distances for batch...")
            batch_mean_distances, batch_all_distances, batch_indices = calculate_distances(
                knn, normalized_good_embeddings, batch_test_embeddings)
            
            all_mean_distances.append(batch_mean_distances)
            
            # Save batch distances to file
            batch_distances_file = os.path.join(output_dir, f"distances_{test_name}_batch_{batch_idx+1}.pkl")
            with open(batch_distances_file, "wb") as f:
                pickle.dump({
                    "mean_distances": batch_mean_distances, 
                    "all_distances": batch_all_distances, 
                    "indices": batch_indices,
                    "batch_range": (batch_idx, batch_idx + len(batch_sequences))
                }, f)
            print(f"Saved batch distances to {batch_distances_file}")
            
            # Free memory
            del batch_sequences, batch_test_embeddings, batch_all_distances, batch_indices
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            batch_idx += 1
        
        # Combine all batches
        if all_mean_distances:
            mean_distances = np.concatenate(all_mean_distances)
            
            # Save combined distances to file
            distances_file = os.path.join(output_dir, f"distances_{test_name}_combined.pkl")
            with open(distances_file, "wb") as f:
                pickle.dump({"mean_distances": mean_distances}, f)
            print(f"Saved combined distances to {distances_file}")
            
            # Plot distances
            plot_distances(mean_distances, test_name, output_dir)
            
            # Free memory
            del all_mean_distances, mean_distances
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")