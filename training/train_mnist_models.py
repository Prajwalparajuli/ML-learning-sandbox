import argparse
import base64
import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_JSON = REPO_ROOT / "app/src/data/deep-learning/eval/mnist_eval_1000.json"
TRAINING_DIR = REPO_ROOT / "training"
EMNIST_TRAIN_IMAGES = TRAINING_DIR / "emnist-digits-train-images-idx3-ubyte.gz"
EMNIST_TRAIN_LABELS = TRAINING_DIR / "emnist-digits-train-labels-idx1-ubyte.gz"
EMNIST_TEST_IMAGES = TRAINING_DIR / "emnist-digits-test-images-idx3-ubyte.gz"
EMNIST_TEST_LABELS = TRAINING_DIR / "emnist-digits-test-labels-idx1-ubyte.gz"
MLP_MODEL_JSON = REPO_ROOT / "app/src/data/deep-learning/models/mnist/mlp/model.json"
MLP_WEIGHTS_JSON = REPO_ROOT / "app/src/data/deep-learning/models/mnist/mlp/weights-shard1.json"
CNN_MODEL_JSON = REPO_ROOT / "app/src/data/deep-learning/models/mnist/cnn/model.json"
CNN_WEIGHTS_JSON = REPO_ROOT / "app/src/data/deep-learning/models/mnist/cnn/weights-shard1.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_local_mnist_eval() -> Tuple[np.ndarray, np.ndarray]:
    payload = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    count = int(payload["count"])
    rows, cols, _ = payload["image_shape"]
    raw = base64.b64decode(payload["images_u8_b64"])
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(count, rows, cols).astype(np.float32) / 255.0
    labels = np.asarray(payload["labels"], dtype=np.int64)
    return arr, labels


def read_idx_images_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buffer = f.read()
    if len(buffer) < 16:
        raise ValueError(f"Invalid IDX image file: {path}")
    magic = int.from_bytes(buffer[0:4], byteorder="big")
    if magic != 2051:
        raise ValueError(f"IDX image magic mismatch for {path}: {magic}")
    count = int.from_bytes(buffer[4:8], byteorder="big")
    rows = int.from_bytes(buffer[8:12], byteorder="big")
    cols = int.from_bytes(buffer[12:16], byteorder="big")
    data = np.frombuffer(buffer, dtype=np.uint8, offset=16)
    data = data.reshape(count, rows, cols).astype(np.float32) / 255.0
    return data


def read_idx_labels_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buffer = f.read()
    if len(buffer) < 8:
        raise ValueError(f"Invalid IDX label file: {path}")
    magic = int.from_bytes(buffer[0:4], byteorder="big")
    if magic != 2049:
        raise ValueError(f"IDX label magic mismatch for {path}: {magic}")
    count = int.from_bytes(buffer[4:8], byteorder="big")
    labels = np.frombuffer(buffer, dtype=np.uint8, offset=8)
    if labels.shape[0] != count:
        raise ValueError(f"IDX label count mismatch for {path}")
    return labels.astype(np.int64)


def fix_emnist_orientation(images: np.ndarray) -> np.ndarray:
    # EMNIST raw IDX orientation differs from MNIST. This aligns it with runtime canvas orientation.
    return np.flip(np.transpose(images, (0, 2, 1)), axis=2).copy()


def load_emnist_digits() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = [
        EMNIST_TRAIN_IMAGES,
        EMNIST_TRAIN_LABELS,
        EMNIST_TEST_IMAGES,
        EMNIST_TEST_LABELS,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing EMNIST files: {missing}")

    train_images = read_idx_images_gz(EMNIST_TRAIN_IMAGES)
    train_labels = read_idx_labels_gz(EMNIST_TRAIN_LABELS)
    test_images = read_idx_images_gz(EMNIST_TEST_IMAGES)
    test_labels = read_idx_labels_gz(EMNIST_TEST_LABELS)

    train_images = fix_emnist_orientation(train_images)
    test_images = fix_emnist_orientation(test_images)
    return train_images, train_labels, test_images, test_labels


def stratified_split_train_val(labels: np.ndarray, train_ratio: float, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for c in range(10):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.asarray(train_idx), np.asarray(val_idx)


class MnistLocalDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _random_affine(img: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(-22, 22)
        tx = random.uniform(-0.2, 0.2) * 28
        ty = random.uniform(-0.2, 0.2) * 28
        scale = random.uniform(0.78, 1.18)
        shear_x = random.uniform(-12, 12)
        theta = torch.tensor(
            [
                [scale * np.cos(np.deg2rad(angle + shear_x)), -scale * np.sin(np.deg2rad(angle)), tx / 14.0],
                [scale * np.sin(np.deg2rad(angle)), scale * np.cos(np.deg2rad(angle + shear_x)), ty / 14.0],
            ],
            dtype=torch.float32,
        )
        grid = F.affine_grid(theta.unsqueeze(0), (1, 1, 28, 28), align_corners=False)
        out = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return out.squeeze(0)

    @staticmethod
    def _stroke_width(img: torch.Tensor) -> torch.Tensor:
        roll = random.random()
        if roll < 0.35:
            # Dilate (thicker strokes)
            return F.max_pool2d(img.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        if roll < 0.55:
            # Erode (thinner strokes)
            return (-F.max_pool2d((-img).unsqueeze(0), kernel_size=3, stride=1, padding=1)).squeeze(0)
        return img

    @staticmethod
    def _blur_noise(img: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.45:
            kernel = torch.tensor(
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                dtype=img.dtype,
                device=img.device,
            ) / 16.0
            img = F.conv2d(
                img.unsqueeze(0),
                kernel.view(1, 1, 3, 3),
                padding=1,
            ).squeeze(0)
        if random.random() < 0.5:
            noise = torch.randn_like(img) * random.uniform(0.01, 0.06)
            img = img + noise
        return img.clamp_(0.0, 1.0)

    @staticmethod
    def _normalize_center_mass(img: torch.Tensor) -> torch.Tensor:
        # Shift center of mass to image center, mimicking runtime centering.
        flat = img.view(-1)
        mass = flat.sum()
        if mass < 1e-6:
            return img
        coords = torch.arange(28, device=img.device, dtype=img.dtype)
        row_mass = img[0].sum(dim=1)
        col_mass = img[0].sum(dim=0)
        center_r = (coords * row_mass).sum() / mass
        center_c = (coords * col_mass).sum() / mass
        target = torch.tensor(13.5, device=img.device, dtype=img.dtype)
        shift_r = int(torch.round(target - center_r).item())
        shift_c = int(torch.round(target - center_c).item())
        shifted = torch.zeros_like(img)
        src_r0 = max(0, -shift_r)
        src_r1 = min(28, 28 - shift_r) if shift_r >= 0 else 28
        dst_r0 = max(0, shift_r)
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        src_c0 = max(0, -shift_c)
        src_c1 = min(28, 28 - shift_c) if shift_c >= 0 else 28
        dst_c0 = max(0, shift_c)
        dst_c1 = dst_c0 + (src_c1 - src_c0)
        shifted[:, dst_r0:dst_r1, dst_c0:dst_c1] = img[:, src_r0:src_r1, src_c0:src_c1]
        return shifted

    def __getitem__(self, idx):
        x = torch.from_numpy(self.images[idx]).unsqueeze(0)
        y = int(self.labels[idx])

        if self.augment:
            x = self._random_affine(x)
            x = self._stroke_width(x)
            x = self._blur_noise(x)
            if random.random() < 0.35:
                # Randomly drop weak pixels to simulate dry brush/light stroke.
                x = torch.where(x > random.uniform(0.03, 0.09), x, torch.zeros_like(x))
            x = self._normalize_center_mass(x)

        return x.float(), y


class MlpCompat(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.dropout = nn.Dropout(0.22)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CnnCompat(nn.Module):
    def __init__(self, conv_kernels: int):
        super().__init__()
        self.conv = nn.Conv2d(1, conv_kernels, kernel_size=3, stride=1, padding=0, bias=True)
        self.dropout2d = nn.Dropout2d(0.12)
        self.fc = nn.Linear(conv_kernels * 13 * 13, 10)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.dropout2d(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.flatten(1)
        x = self.fc(x)
        return x


@dataclass
class TrainResult:
    best_val_acc: float
    test_acc: float
    model: nn.Module


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> TrainResult:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    best_val = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        total_correct = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * y.size(0)
            total_seen += y.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()

        scheduler.step()
        train_acc = total_correct / max(total_seen, 1)
        val_acc = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch:02d} "
            f"loss={total_loss / max(total_seen, 1):.4f} "
            f"train_acc={train_acc * 100:.2f}% "
            f"val_acc={val_acc * 100:.2f}%"
        )
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc = evaluate(model, test_loader, device)
    return TrainResult(best_val_acc=best_val, test_acc=test_acc, model=model)


def export_mlp(model: MlpCompat, hidden_size: int) -> None:
    state = model.state_dict()
    w1 = state["fc1.weight"].cpu().numpy().astype(np.float32).tolist()
    b1 = state["fc1.bias"].cpu().numpy().astype(np.float32).tolist()
    w2 = state["fc2.weight"].cpu().numpy().astype(np.float32).tolist()
    b2 = state["fc2.bias"].cpu().numpy().astype(np.float32).tolist()

    weights = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    MLP_WEIGHTS_JSON.write_text(json.dumps(weights, separators=(",", ":")), encoding="utf-8")

    meta = json.loads(MLP_MODEL_JSON.read_text(encoding="utf-8"))
    meta["hidden_size"] = int(hidden_size)
    meta["version"] = int(meta.get("version", 1)) + 1
    MLP_MODEL_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def export_cnn(model: CnnCompat, conv_kernels: int) -> None:
    state = model.state_dict()
    kernels = state["conv.weight"].squeeze(1).cpu().numpy().astype(np.float32).tolist()
    conv_bias = state["conv.bias"].cpu().numpy().astype(np.float32)
    dense_w = state["fc.weight"].cpu().numpy().astype(np.float32)
    dense_b = state["fc.bias"].cpu().numpy().astype(np.float32)

    # Fold conv bias approximately into dense bias via mean pooled receptive contribution.
    dense_b_adjusted = dense_b.copy()
    bias_gain = 0.12
    for cls in range(10):
        for ch in range(conv_kernels):
            start = ch * 13 * 13
            end = start + 13 * 13
            dense_b_adjusted[cls] += bias_gain * conv_bias[ch] * float(dense_w[cls, start:end].mean())

    weights = {
        "kernels": kernels,
        "dense_w": dense_w.tolist(),
        "dense_b": dense_b_adjusted.tolist(),
    }
    CNN_WEIGHTS_JSON.write_text(json.dumps(weights, separators=(",", ":")), encoding="utf-8")

    meta = json.loads(CNN_MODEL_JSON.read_text(encoding="utf-8"))
    meta["conv_kernels"] = int(conv_kernels)
    meta["feature_size"] = int(conv_kernels * 13 * 13)
    meta["version"] = int(meta.get("version", 1)) + 1
    CNN_MODEL_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_loaders(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = MnistLocalDataset(train_images[train_idx], train_labels[train_idx], augment=True)
    val_ds = MnistLocalDataset(train_images[val_idx], train_labels[val_idx], augment=False)
    test_ds = MnistLocalDataset(test_images, test_labels, augment=False)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.72)
    parser.add_argument("--val-ratio", type=float, default=0.14)
    parser.add_argument("--dataset", choices=["emnist_digits", "local_eval"], default="emnist_digits")
    parser.add_argument("--mlp-hidden", type=int, default=384)
    parser.add_argument("--cnn-kernels", type=int, default=48)
    parser.add_argument("--mlp-epochs", type=int, default=18)
    parser.add_argument("--cnn-epochs", type=int, default=24)
    parser.add_argument("--mlp-lr", type=float, default=1.2e-3)
    parser.add_argument("--cnn-lr", type=float, default=9e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
        device = torch.device(f"cuda:{args.gpu_index}")
    else:
        device = torch.device("cpu")
    print(f"device={device}")

    if args.dataset == "emnist_digits":
        train_images, train_labels, test_images, test_labels = load_emnist_digits()
    else:
        images, labels = load_local_mnist_eval()
        train_images, train_labels = images, labels
        test_images, test_labels = images, labels

    train_idx, val_idx = stratified_split_train_val(train_labels, args.train_ratio, args.val_ratio, args.seed)
    train_loader, val_loader, test_loader = build_loaders(
        train_images, train_labels, test_images, test_labels, train_idx, val_idx, args.batch_size, device
    )
    print(
        f"dataset={args.dataset} "
        f"samples train={len(train_idx)} val={len(val_idx)} test={len(test_labels)}"
    )

    if not args.skip_mlp:
        print("\n=== Training MLP ===")
        mlp = MlpCompat(hidden_size=args.mlp_hidden)
        mlp_result = train_model(
            mlp,
            train_loader,
            val_loader,
            test_loader,
            device,
            epochs=args.mlp_epochs,
            lr=args.mlp_lr,
            weight_decay=args.weight_decay,
        )
        print(
            f"MLP best_val={mlp_result.best_val_acc * 100:.2f}% "
            f"test={mlp_result.test_acc * 100:.2f}%"
        )
        export_mlp(mlp_result.model, hidden_size=args.mlp_hidden)
        print("Exported MLP weights + model metadata.")

    if not args.skip_cnn:
        print("\n=== Training CNN ===")
        cnn = CnnCompat(conv_kernels=args.cnn_kernels)
        cnn_result = train_model(
            cnn,
            train_loader,
            val_loader,
            test_loader,
            device,
            epochs=args.cnn_epochs,
            lr=args.cnn_lr,
            weight_decay=args.weight_decay,
        )
        print(
            f"CNN best_val={cnn_result.best_val_acc * 100:.2f}% "
            f"test={cnn_result.test_acc * 100:.2f}%"
        )
        export_cnn(cnn_result.model, conv_kernels=args.cnn_kernels)
        print("Exported CNN weights + model metadata.")

    print("\nDone.")


if __name__ == "__main__":
    main()
