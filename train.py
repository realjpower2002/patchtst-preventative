import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    PatchTSTConfig,
    PatchTSTForPretraining,
    Trainer,
    TrainingArguments,
)

# Wrap dataset to produce past values for timeseries model
class TSPretrainDataset(Dataset):
    def __init__(self, array):
        # array: (N, seq_len, channels)
        self.data = torch.from_numpy(array).float()
    def __len__(self): 
        return len(self.data)
    def __getitem__(self, i):
        return {"past_values": self.data[i]}

# Load data
raw = np.load("./Data/training_data.npy")   # e.g. (102272, 512, 8)
np.random.shuffle(raw)
n = len(raw)
train_ds = TSPretrainDataset(raw[: int(0.8 * n)])
eval_ds  = TSPretrainDataset(raw[int(0.8 * n) :])

print("Loaded training and validation data.")


# Configure PatchTST for pretraining
config = PatchTSTConfig(
    num_input_channels=8,        # your per‐step feature count
    context_length=512,          # your sequence length
    patch_length=16,
    stride=8,
    mask_type="random",          # random masking
    random_mask_ratio=0.15,      # mask 15% of patches
    use_cls_token=False,         # no [CLS] token needed
)

model = PatchTSTForPretraining(config)  # :contentReference[oaicite:0]{index=0}

print("Created model.")

# Trainer just sees `past_values` and the model computes `.loss` internally
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    logging_dir="./logs",
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

        dataloader = DataLoader(eval_dataset, batch_size=self.args.per_device_eval_batch_size)
        total_loss = 0
        count = 0

        for batch in dataloader:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                output = self.model(**batch)
                loss = output.loss
                total_loss += loss.item()
                count += 1
            
        avg_loss = total_loss/count
        print(f"\nEval loss : {avg_loss:.6f}\n")
            
        return metrics
        

trainer = TrainerWithValidation(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-25480")
