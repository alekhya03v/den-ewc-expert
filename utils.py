# utils.py

import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_model(model, dataloader, task_id, device):
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        x = images.view(images.size(0), -1)

        logits = model(x, task_id)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / max(total, 1)


def print_accuracy_matrix(acc_matrix):
    print("\n=== Accuracy Matrix (%) ===")
    for train_task, row in enumerate(acc_matrix):
        row_str = " | ".join([f"{x:6.2f}" for x in row])
        print(f"After Task {train_task}: {row_str}")


def compute_average_accuracy(acc_matrix, current_task_id):
    vals = acc_matrix[current_task_id][:current_task_id + 1]
    vals = [v for v in vals if v >= 0]
    return sum(vals) / max(len(vals), 1)


def compute_average_forgetting(acc_matrix, current_task_id):
    """
    Forgetting after learning up to current_task_id:
    For each old task j:
        max_acc_j_before - current_acc_j
    """
    if current_task_id == 0:
        return 0.0

    forgetting_vals = []
    for j in range(current_task_id):
        best_prev = max(acc_matrix[t][j] for t in range(current_task_id))
        current = acc_matrix[current_task_id][j]
        forgetting_vals.append(best_prev - current)

    return sum(forgetting_vals) / max(len(forgetting_vals), 1)
