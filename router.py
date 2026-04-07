# router.py

from gate import compute_avg_reconstruction_error


def choose_expert_by_gate(gates, gate_to_expert, dataloader, device):
    """
    gates: dict[task_id] = trained gate model
    gate_to_expert: dict[task_id] = expert_id
    """
    best_task_id = None
    best_error = float("inf")

    for gate_task_id, gate_model in gates.items():
        err = compute_avg_reconstruction_error(gate_model, dataloader, device)
        print(f"[Router] Gate task {gate_task_id} reconstruction error: {err:.6f}")

        if err < best_error:
            best_error = err
            best_task_id = gate_task_id

    if best_task_id is None:
        return None, None

    selected_expert_id = gate_to_expert[best_task_id]
    return selected_expert_id, best_error
