# ewc.py

import copy
import torch
import torch.nn.functional as F


def clone_params(model):
    return {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


@torch.no_grad()
def merge_fishers(fisher_old, fisher_new, alpha=0.5):
    """
    Optional helper if you want running fusion of fishers.
    """
    if fisher_old is None:
        return fisher_new

    merged = {}
    keys = fisher_new.keys()
    for k in keys:
        if k in fisher_old:
            merged[k] = alpha * fisher_old[k] + (1.0 - alpha) * fisher_new[k]
        else:
            merged[k] = fisher_new[k]
    return merged


def compute_fisher(model, dataloader, task_id, device, max_batches=100):
    model.eval()
    fisher = {
        n: torch.zeros_like(p, device=device)
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    count = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)
        x = images.view(images.size(0), -1)

        model.zero_grad()
        logits = model(x, task_id)
        log_probs = F.log_softmax(logits, dim=1)

        # Use predicted label likelihood approximation
        preds = log_probs.argmax(dim=1)
        loss = F.nll_loss(log_probs, preds)
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2

        count += 1

    if count > 0:
        for n in fisher:
            fisher[n] /= count

    return fisher


def ewc_penalty(model, prev_params_list, prev_fishers_list, ewc_lambda=1000.0):
    """
    Multi-task EWC penalty:
    sum over all past task snapshots
    """
    if len(prev_params_list) == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for old_params, fisher in zip(prev_params_list, prev_fishers_list):
        for n, p in model.named_parameters():
            if n in old_params and n in fisher:
                loss += (fisher[n] * (p - old_params[n]) ** 2).sum()

    return ewc_lambda * loss
