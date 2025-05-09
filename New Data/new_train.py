import numpy as np
import torch
import os
import shutil
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import (
    PatchTSTConfig,
    PatchTSTForPretraining,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

# Wrap dataset to produce past values for timeseries model
class TSPretrainDataset(Dataset):
    def __init__(self, array):
        # array: (N, seq_len, channels)
        self.data = torch.from_numpy(array).float()
    def __len__(self): 
        return len(self.data)
    def __getitem__(self, i):
        return {"past_values": self.data[i]}

# Create directories
os.makedirs("./new_checkpoints", exist_ok=True)
os.makedirs("./new_logs", exist_ok=True)
os.makedirs("./new_backups", exist_ok=True)
os.makedirs("./new_diagrams", exist_ok=True)

# Load data
train_raw = np.load("./new_training_data.npy")   # (50000, 512, 8)
eval_raw = np.load("./new_validation_data.npy")  # (10000, 512, 8)

train_ds = TSPretrainDataset(train_raw)
eval_ds = TSPretrainDataset(eval_raw)

print(f"Loaded training data: {train_raw.shape}")
print(f"Loaded validation data: {eval_raw.shape}")

# Configure PatchTST for pretraining
config = PatchTSTConfig(
    num_input_channels=4,        # second dataset has 4 features instead of 8
    context_length=512,          # your sequence length
    patch_length=16,
    stride=8,
    mask_type="random",          # random masking
    random_mask_ratio=0.15,      # mask 15% of patches
    use_cls_token=False,         # no [CLS] token needed
)

model = PatchTSTForPretraining(config)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Created model. Using device: {device}")

# Store training and validation losses for plotting
train_losses = []
val_losses = []
train_steps = []
val_steps = []
epochs = []

# Custom callback for backing up checkpoints and generating diagrams
class BackupAndDiagramCallback(TrainerCallback):
    def __init__(self):
        self.step = 0
        self.epoch = 0
        self.current_epoch_train_losses = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        # Collect training loss
        if "loss" in logs:
            self.step += 1
            self.current_epoch_train_losses.append(logs["loss"])
            train_losses.append(logs["loss"])
            train_steps.append(self.step)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch += 1
        epochs.append(self.epoch)
        
        # Backup the latest checkpoint
        latest_checkpoint = f"./new_checkpoints/checkpoint-{state.global_step}"
        backup_dir = f"./new_backups/epoch-{self.epoch}"
        
        if os.path.exists(latest_checkpoint):
            os.makedirs(backup_dir, exist_ok=True)
            # Backup model.safetensors file
            shutil.copy(
                os.path.join(latest_checkpoint, "model.safetensors"), 
                os.path.join(backup_dir, "model.safetensors")
            )
            print(f"Backed up checkpoint for epoch {self.epoch}")
        
        # Generate and save training progress diagram
        if len(train_losses) > 0 and len(val_losses) > 0:
            
            # Create figure with shared y-axis to ensure consistent scale
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot training loss with step numbers
            ax.plot(train_steps, train_losses, label=f"Training Loss (Epoch 1-{self.epoch})", alpha=0.7)
            
            # Plot validation loss with matching step numbers
            ax.plot(val_steps, val_losses, label=f"Validation Loss (Epoch 1-{self.epoch})", marker='o')
            
            # Set y-axis limits explicitly to ensure consistency between training and validation
            all_losses = train_losses + val_losses
            if all_losses:
                min_loss = min(all_losses) * 0.95  # Add 5% padding
                max_loss = max(all_losses) * 1.05
                ax.set_ylim(min_loss, max_loss)
            
            # Add epoch markers
            for i, epoch_num in enumerate(epochs):
                if i > 0:  # Skip first epoch which might not have a clear boundary
                    # Find the step that corresponds to the epoch boundary
                    step_idx = len(train_losses) * i // self.epoch
                    if step_idx < len(train_steps):
                        step = train_steps[step_idx]
                        ax.axvline(x=step, color='gray', linestyle='--', alpha=0.5)
                        ymin = ax.get_ylim()[0]
                        yrange = ax.get_ylim()[1] - ymin
                        ax.text(step, ymin + yrange*0.02, 
                                f"Epoch {epoch_num}", rotation=90, verticalalignment='bottom')
            
            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Loss")
            ax.set_title(f"Training and Validation Loss vs Steps (Epoch 1-{self.epoch})")
            ax.legend()
            ax.grid(True)
            
            # Ensure there's no earlier plt.figure causing conflicts
            plt.tight_layout()
            fig.savefig(f"./new_diagrams/loss_epoch_{self.epoch}.png")
            plt.close('all')
            
            print(f"Generated diagram for epoch {self.epoch}")
        
        # Reset for next epoch
        self.current_epoch_train_losses = []

# Trainer just sees `past_values` and the model computes `.loss` internally
training_args = TrainingArguments(
    output_dir="./new_checkpoints",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    logging_dir="./new_logs",
    report_to="none"
)

class TrainerWithValidation(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # eval_dataset is passed to trainer object as a member variable training, but
        # can also be overridden
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        # Your custom evaluation logic here
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        # Use the same loss calculation method as the training loop to ensure consistency
        dataloader = DataLoader(eval_dataset, batch_size=self.args.per_device_eval_batch_size)
        losses = []

        for batch in dataloader:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                output = self.model(**batch)
                loss = output.loss
                losses.append(loss.item())
            
        # Use the same loss processing as training
        avg_loss = sum(losses) / len(losses)
        val_losses.append(avg_loss)  # Store validation loss for plotting
        val_steps.append(backup_callback.step)  # Store the current step for validation loss
        print(f"\nEval loss : {avg_loss:.6f} at step {backup_callback.step}\n")
            
        return metrics

# Create callback
backup_callback = BackupAndDiagramCallback()

trainer = TrainerWithValidation(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    callbacks=[backup_callback]
)

# Start training from scratch
trainer.train()