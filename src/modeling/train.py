from tqdm import tqdm

import torch
from sklearn.metrics import f1_score

from .. import CONFIG


def train_one_epoch(model, loader, optimizer, criterion, scheduler, scaler, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    for images, labels in pbar:
        images = images.to(CONFIG.device)
        labels = labels.to(CONFIG.device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=CONFIG.device):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"}
        )

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(CONFIG.device)
            labels = labels.to(CONFIG.device)

            with torch.amp.autocast(device_type=CONFIG.device):
                outputs = model(images)
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

            preds = torch.argmax(outputs, 1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(loader) if criterion is not None else 0

    return {
        "metrics": {
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "accuracy": accuracy,
            "avg_loss": avg_loss,
        },
        "predictions": {
            "all_preds": all_preds,
            "all_labels": all_labels,
        },
    }
