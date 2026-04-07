# trainer.py

import copy
import torch
import torch.nn as nn
import torch.optim as optim

from gate import GateAutoEncoder, train_gate_autoencoder
from expert import ExpandableExpert
from ewc import compute_fisher, clone_params, ewc_penalty
from router import choose_expert_by_gate
from utils import evaluate_model, compute_average_accuracy, compute_average_forgetting


class HybridContinualTrainer:
    def __init__(
        self,
        device,
        input_dim=784,
        gate_latent_dim=128,
        expert_h1=400,
        expert_h2=400,
        num_classes=10,
        gate_epochs=5,
        expert_epochs=5,
        gate_lr=1e-3,
        expert_lr=1e-3,
        ewc_lambda=1000.0,
        expansion_size=64,
        new_expert_threshold=0.030,
        expand_acc_threshold=85.0,
        forgetting_threshold=5.0,
    ):
        self.device = device

        self.input_dim = input_dim
        self.gate_latent_dim = gate_latent_dim
        self.expert_h1 = expert_h1
        self.expert_h2 = expert_h2
        self.num_classes = num_classes

        self.gate_epochs = gate_epochs
        self.expert_epochs = expert_epochs
        self.gate_lr = gate_lr
        self.expert_lr = expert_lr
        self.ewc_lambda = ewc_lambda
        self.expansion_size = expansion_size

        self.new_expert_threshold = new_expert_threshold
        self.expand_acc_threshold = expand_acc_threshold
        self.forgetting_threshold = forgetting_threshold

        # storage
        self.gates = {}               # task_id -> gate model
        self.experts = {}             # expert_id -> expert model
        self.gate_to_expert = {}      # gate task_id -> expert_id
        self.task_to_expert = {}      # task_id -> expert_id

        self.expert_memories = {}     # expert_id -> {"params": [...], "fishers": [...]}

        self.next_expert_id = 0

    def create_new_expert(self):
        expert = ExpandableExpert(
            input_dim=self.input_dim,
            h1=self.expert_h1,
            h2=self.expert_h2,
            num_classes=self.num_classes
        ).to(self.device)

        expert_id = self.next_expert_id
        self.next_expert_id += 1
        self.experts[expert_id] = expert
        self.expert_memories[expert_id] = {"params": [], "fishers": []}

        print(f"[System] Created new expert {expert_id}")
        return expert_id

    def train_expert_single_task(self, expert, train_loader, val_loader, task_id, expert_id, use_ewc=True):
        expert.train()
        optimizer = optim.Adam(expert.parameters(), lr=self.expert_lr)
        criterion = nn.CrossEntropyLoss()

        memory = self.expert_memories[expert_id]
        prev_params_list = memory["params"] if use_ewc else []
        prev_fishers_list = memory["fishers"] if use_ewc else []

        for epoch in range(self.expert_epochs):
            expert.train()
            running_task_loss = 0.0
            running_ewc_loss = 0.0
            count = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                x = images.view(images.size(0), -1)

                optimizer.zero_grad()

                logits = expert(x, task_id)
                task_loss = criterion(logits, labels)

                reg_loss = ewc_penalty(
                    expert,
                    prev_params_list=prev_params_list,
                    prev_fishers_list=prev_fishers_list,
                    ewc_lambda=self.ewc_lambda
                ) if use_ewc else torch.tensor(0.0, device=self.device)

                loss = task_loss + reg_loss
                loss.backward()
                optimizer.step()

                running_task_loss += task_loss.item()
                running_ewc_loss += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else float(reg_loss)
                count += 1

            val_acc = evaluate_model(expert, val_loader, task_id, self.device)
            print(
                f"[Expert {expert_id} | Task {task_id}] "
                f"Epoch {epoch+1}/{self.expert_epochs} "
                f"TaskLoss: {running_task_loss / max(count,1):.4f} "
                f"EWC: {running_ewc_loss / max(count,1):.4f} "
                f"ValAcc: {val_acc:.2f}"
            )

    def store_expert_memory(self, expert_id, fisher_loader, task_id):
        expert = self.experts[expert_id]
        fisher = compute_fisher(expert, fisher_loader, task_id, self.device, max_batches=100)
        params = clone_params(expert)

        self.expert_memories[expert_id]["params"].append(params)
        self.expert_memories[expert_id]["fishers"].append(fisher)

        print(f"[Memory] Stored Fisher and params for expert {expert_id} after task {task_id}")

    def maybe_expand_expert(self, expert_id, all_tasks, current_task_id):
        expert = self.experts[expert_id]
        expert.expand_h2(self.expansion_size)
        expert.to(self.device)
        print(f"[DEN] Expanded expert {expert_id}. New model size: {expert.get_model_size()} params")

        # Evaluate after expansion later through retraining

    def train_task(self, task_id, task_info, all_tasks, acc_matrix):
        train_loader = task_info["train_loader"]
        val_loader = task_info["val_loader"]
        task_name = task_info["task_name"]

        print(f"\n==============================")
        print(f"Training Task {task_id}: {task_name}")
        print(f"==============================")

        # 1. Train gate for current task
        gate_model = GateAutoEncoder(input_dim=self.input_dim, latent_dim=self.gate_latent_dim)
        gate_model = train_gate_autoencoder(
            gate_model,
            train_loader,
            self.device,
            epochs=self.gate_epochs,
            lr=self.gate_lr
        )
        self.gates[task_id] = gate_model

        # 2. If first task, create first expert
        if len(self.experts) == 0:
            expert_id = self.create_new_expert()
            expert = self.experts[expert_id]
            expert.add_head(task_id, self.num_classes)
            expert.to(self.device)

            self.train_expert_single_task(
                expert, train_loader, val_loader, task_id, expert_id, use_ewc=False
            )
            self.store_expert_memory(expert_id, train_loader, task_id)

            self.gate_to_expert[task_id] = expert_id
            self.task_to_expert[task_id] = expert_id
            return

        # 3. Route to existing expert
        selected_expert_id, gate_err = choose_expert_by_gate(
            self.gates, self.gate_to_expert, train_loader, self.device
        )

        # Since current task gate is also in self.gates, it may pick itself.
        # We only want to compare against previous tasks for routing.
        previous_gates = {k: v for k, v in self.gates.items() if k != task_id}

        if len(previous_gates) == 0:
            selected_expert_id = None
            best_prev_err = float("inf")
        else:
            best_prev_err = float("inf")
            best_prev_task = None
            from gate import compute_avg_reconstruction_error

            for prev_task_id, g in previous_gates.items():
                err = compute_avg_reconstruction_error(g, train_loader, self.device)
                print(f"[Router-prev] Gate task {prev_task_id} error on new task {task_id}: {err:.6f}")
                if err < best_prev_err:
                    best_prev_err = err
                    best_prev_task = prev_task_id

            selected_expert_id = self.gate_to_expert[best_prev_task] if best_prev_task is not None else None

        # 4. Decide whether new expert is needed
        if selected_expert_id is None or best_prev_err > self.new_expert_threshold:
            print(
                f"[Decision] Task {task_id} seems too different "
                f"(best_prev_err={best_prev_err:.6f}). Creating new expert."
            )
            expert_id = self.create_new_expert()
            expert = self.experts[expert_id]
            expert.add_head(task_id, self.num_classes)
            expert.to(self.device)

            self.train_expert_single_task(
                expert, train_loader, val_loader, task_id, expert_id, use_ewc=False
            )
            self.store_expert_memory(expert_id, train_loader, task_id)

            self.gate_to_expert[task_id] = expert_id
            self.task_to_expert[task_id] = expert_id
            return

        # 5. Reuse selected expert
        expert_id = selected_expert_id
        expert = self.experts[expert_id]
        expert.add_head(task_id, self.num_classes)
        expert.to(self.device)

        print(f"[Decision] Reusing expert {expert_id} for task {task_id}")

        self.train_expert_single_task(
            expert, train_loader, val_loader, task_id, expert_id, use_ewc=True
        )

        # 6. Check if expansion is needed
        new_task_val_acc = evaluate_model(expert, val_loader, task_id, self.device)

        # Evaluate forgetting on old tasks that belong to this expert
        for old_task_id in range(task_id):
            if old_task_id in self.task_to_expert:
                old_expert_id = self.task_to_expert[old_task_id]
                if old_expert_id == expert_id:
                    old_test_loader = all_tasks[old_task_id]["test_loader"]
                    acc_matrix[task_id][old_task_id] = evaluate_model(expert, old_test_loader, old_task_id, self.device)

        avg_forgetting = compute_average_forgetting(acc_matrix, task_id)

        print(f"[Check] New task val acc: {new_task_val_acc:.2f}")
        print(f"[Check] Avg forgetting so far: {avg_forgetting:.2f}")

        need_expand = (new_task_val_acc < self.expand_acc_threshold) or (avg_forgetting > self.forgetting_threshold)

        if need_expand:
            print("[Decision] Expansion triggered.")
            self.maybe_expand_expert(expert_id, all_tasks, task_id)

            # retrain after expansion
            self.train_expert_single_task(
                expert, train_loader, val_loader, task_id, expert_id, use_ewc=True
            )
        else:
            print("[Decision] Expansion not needed.")

        self.store_expert_memory(expert_id, train_loader, task_id)
        self.gate_to_expert[task_id] = expert_id
        self.task_to_expert[task_id] = expert_id

    def evaluate_all_tasks(self, all_tasks, current_task_id, acc_matrix):
        print(f"\n[Evaluation] After learning task {current_task_id}")
        for eval_task_id in range(current_task_id + 1):
            expert_id = self.task_to_expert[eval_task_id]
            expert = self.experts[expert_id]
            test_loader = all_tasks[eval_task_id]["test_loader"]

            acc = evaluate_model(expert, test_loader, eval_task_id, self.device)
            acc_matrix[current_task_id][eval_task_id] = acc

            print(f"  Task {eval_task_id} using Expert {expert_id}: Test Acc = {acc:.2f}")

        avg_acc = compute_average_accuracy(acc_matrix, current_task_id)
        avg_forgetting = compute_average_forgetting(acc_matrix, current_task_id)
        print(f"[Summary] Avg Accuracy: {avg_acc:.2f}")
        print(f"[Summary] Avg Forgetting: {avg_forgetting:.2f}")
