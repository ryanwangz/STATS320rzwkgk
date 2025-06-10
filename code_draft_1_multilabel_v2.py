
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from spikeinterface.preprocessing import bandpass_filter
import spikeforest as sf
from tqdm.auto import tqdm


class SpikeCountDataset(Dataset):
    """Dataset that returns multi-channel signal windows and multi‑hot neuron labels."""
    def __init__(self, rec_ext, sort_ext, win_samples, mean=None, std=None):
        self.rec   = rec_ext
        self.sort  = sort_ext
        self.win   = win_samples
        self.starts = np.arange(
            0,
            rec_ext.get_num_frames() - win_samples + 1,
            win_samples,
            dtype=np.int64
        )
        # Sorted list of neuron IDs for consistent ordering
        self.unit_ids = sorted(sort_ext.get_unit_ids())
        self.num_units = len(self.unit_ids)
        # Pre‑fetch spike trains
        self.trains = {
            u: sort_ext.get_unit_spike_train(u, segment_index=0)
            for u in self.unit_ids
        }
        self.n_chan = rec_ext.get_num_channels()

        # Handle mean / std as torch tensors if provided
        if mean is not None and std is not None:
            self.mean = torch.as_tensor(mean, dtype=torch.float32)
            self.std  = torch.as_tensor(std, dtype=torch.float32)
        else:
            self.mean = self.std = None

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        e = s + self.win

        # Load traces and convert to (channels, time) tensor
        traces = self.rec.get_traces(start_frame=s, end_frame=e).astype(np.float32).T
        X = torch.from_numpy(traces)  # shape (C, T)

        # Multi‑hot labels (1 if neuron fired at least once in window)
        y = torch.zeros(self.num_units, dtype=torch.float32)
        for i, u in enumerate(self.unit_ids):
            st = self.trains[u]
            if np.any((st >= s) & (st < e)):
                y[i] = 1.0

        # Normalize if mean/std provided
        if (self.mean is not None) and (self.std is not None):
            mean = self.mean.unsqueeze(1)
            std  = self.std.unsqueeze(1)
            X = (X - mean) / std

        return X, y


class BasicBlock1D(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.down  = (
            nn.Sequential(
                nn.Conv1d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
            if (stride != 1 or in_planes != planes)
            else nn.Identity()
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.down(x)
        return self.relu(out)

class ResNet1D(nn.Module):
    def __init__(self, layers, in_ch, out_dim=1):
        super().__init__()
        self.in_planes = 64
        self.conv1     = nn.Conv1d(in_ch, 64,
                                   kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1       = nn.BatchNorm1d(64)
        self.relu      = nn.ReLU(inplace=True)
        self.maxpool   = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1    = self._make_layer(layers[0],  64)
        self.layer2    = self._make_layer(layers[1], 128, stride=2)
        self.layer3    = self._make_layer(layers[2], 256, stride=2)
        self.layer4    = self._make_layer(layers[3], 512, stride=2)
        self.avgpool   = nn.AdaptiveAvgPool1d(1)
        self.fc        = nn.Linear(512, out_dim)

    def _make_layer(self, blocks, planes, stride=1):
        layers = [BasicBlock1D(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).squeeze(-1)
        return self.fc(x)

def main(args):
    hybrid_janelia_uri = (
        "sha1://43298d72b2d0860ae45fc9b0864137a976cb76e8?"
        "hybrid-janelia-spikeforest-recordings.json"
    )
    all_recordings = sf.load_spikeforest_recordings(hybrid_janelia_uri)

    train_name = "rec_64c_1200s_11"
    train_rec  = None
    for R in all_recordings:
        if R.recording_name == train_name:
            train_rec = R
            break
    if train_rec is None:
        raise Exception(f"Could not find recording '{train_name}' in the catalog.")

    print("───────────────────────────────────────────────")
    print(f"Training → {train_rec.study_set_name}/{train_rec.study_name}/{train_rec.recording_name}")
    print(f"   Channels: {train_rec.num_channels}, Duration: {train_rec.duration_sec}s, Sampling: {train_rec.sampling_frequency} Hz")
    print("──────────────────────────────────────────────\n")

    val_name = "rec_64c_600s_12"
    val_rec   = None
    for R in all_recordings:
        if R.recording_name == val_name:
            val_rec = R
            break
    if val_rec is None:
        raise Exception(f"Could not find recording '{val_name}' in the catalog.")

    print("───────────────────────────────────────────────")
    print(f"Validation → {val_rec.study_set_name}/{val_rec.study_name}/{val_rec.recording_name}")
    print(f"   Channels: {val_rec.num_channels}, Duration: {val_rec.duration_sec}s, Sampling: {val_rec.sampling_frequency} Hz")
    print("──────────────────────────────────────────────\n")

    recording_train   = train_rec.get_recording_extractor()
    sorting_train_gt  = train_rec.get_sorting_true_extractor()
    recording_val     = val_rec.get_recording_extractor()
    sorting_val_gt    = val_rec.get_sorting_true_extractor()

    recording_train_f = bandpass_filter(recording_train, freq_min=300, freq_max=6000)
    recording_val_f   = bandpass_filter(recording_val,   freq_min=300, freq_max=6000)

    fs          = recording_train_f.get_sampling_frequency()  # should be 30000.0
    WIN_SAMPLES = int(0.01 * fs)                             # 10 ms → 300 samples
    
    train_dataset_nonnorm = SpikeCountDataset(
        recording_train_f, sorting_train_gt, WIN_SAMPLES
    )
    train_loader_nonnorm = DataLoader(
        train_dataset_nonnorm,
        batch_size=64,
        shuffle=True,
        drop_last=True,
        num_workers=4   # <— worker processes; needs __main__ guard
    )

    val_dataset_nonnorm = SpikeCountDataset(
        recording_val_f, sorting_val_gt, WIN_SAMPLES
    )
    val_loader_nonnorm = DataLoader(
        val_dataset_nonnorm,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )
    print("Computing per‐channel mean/std on training windows…")
    sum_   = torch.zeros(train_dataset_nonnorm.n_chan)
    sum_sq = torch.zeros(train_dataset_nonnorm.n_chan)
    count_ = 0

    with torch.no_grad():
        for X_batch, _ in tqdm(train_loader_nonnorm, desc="→ Accumulating stats", leave=False):
            B, C, W = X_batch.shape
            sum_   += X_batch.sum(dim=(0, 2))
            sum_sq += (X_batch ** 2).sum(dim=(0, 2))
            count_ += B * W

    mean   = (sum_ / count_).numpy()
    var    = (sum_sq / count_) - (sum_ / count_) ** 2
    std    = torch.sqrt(var).numpy()
    print("Done.  (mean & std computed)\n")

    train_dataset = SpikeCountDataset(
        recording_train_f,
        sorting_train_gt,
        WIN_SAMPLES,
        mean=mean,
        std=std
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    val_dataset = SpikeCountDataset(
        recording_val_f,
        sorting_val_gt,
        WIN_SAMPLES,
        mean=mean,
        std=std
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )

    device    = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model     = ResNet1D([2, 2, 2, 2], in_ch=train_dataset.n_chan, out_dim=train_dataset.num_units).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start_epoch = 1
    if args.checkpoint_dir and (args.resume_epoch is not None):
        ckpt_path = os.path.join(
            args.checkpoint_dir,
            f"./multilabel_classification/checkpoint_epoch_{args.resume_epoch}.pth"
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"Loading checkpoint from {ckpt_path} …")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']}.  Next epoch = {start_epoch}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        # --- Training Pass ---
        model.train()
        running_train = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for X, y in loop:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            running_train += loss.item() * X.size(0)
            loop.set_postfix(batch_loss=loss.item())

        avg_train_mse = running_train / len(train_dataset)
        print(f"Epoch {epoch:02d}  TRAIN MSE: {avg_train_mse:.4f}")

        # --- Validation Pass ---
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for X_val, y_val in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                pred_val = model(X_val)
                loss_val = criterion(pred_val, y_val)
                running_val += loss_val.item() * X_val.size(0)

        avg_val_mse = running_val / len(val_dataset)
        print(f"Epoch {epoch:02d}  VAL   MSE: {avg_val_mse:.4f}\n")

        # --- Save Checkpoint ---
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ckpt_out = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        out_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(ckpt_out, out_path)
        print(f"Checkpoint saved → {out_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train/Resume ResNet1D on Hybrid Janelia spike counts"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Folder containing checkpoint_epoch_{n}.pth files"
    )
    parser.add_argument(
        "--resume_epoch",
        type=int,
        default=None,
        help="Epoch number to resume from (loads checkpoint_epoch_{resume_epoch}.pth)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Total number of epochs to run (if resuming at N, runs from N+1 up to this)"
    )
    args = parser.parse_args()

    main(args)
