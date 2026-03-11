# sweep.py
import argparse
import optuna
import wandb
import torch
from torch.utils.data import DataLoader as StdDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

import datasets
import models
from trainer import train_loop, evaluate_loop
from utils import set_seed

def objective(trial, args):
    """The function Optuna runs repeatedly to hunt for the best model."""
    
    # 1. Let Optuna suggest hyperparameters dynamically
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
    
    # --- UPGRADED: Allow Optuna to build incredibly deep networks (up to 15 layers!) ---
    num_layers = trial.suggest_int("num_layers", 3, 15)

    # 2. Initialize Weights & Biases for this specific mutant model
    run = wandb.init(
        project="AlgoBound-Sweeps",
        config={
            "lr": lr, "hidden_dim": hidden_dim, "num_layers": num_layers, 
            "task": args.task, "model": args.model, "use_vn": args.use_vn
        },
        reinit=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    set_seed(42)

    # 3. Load Data dynamically based on the task
    DataLoader = PyGDataLoader if args.task in ['bfs', 'dijkstra'] else StdDataLoader
    
    # --- UPGRADED: Pass the Virtual Node flag to the datasets! ---
    if args.task == 'xor': train_data = datasets.CumulativeXORDataset(2000, args.train_size)
    elif args.task == 'kadane': train_data = datasets.MaxSubarrayDataset(2000, args.train_size)
    elif args.task == 'bfs': train_data = datasets.BFSDataset(2000, args.train_size, use_vn=args.use_vn)
    elif args.task == 'dijkstra': train_data = datasets.DijkstraDataset(2000, args.train_size, use_vn=args.use_vn)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # 4. Build the correct model with Optuna's parameters
    if args.task == 'xor' and args.model == 'transformer': 
        model = models.SmallTransformer(d_model=hidden_dim, num_layers=num_layers)
    elif args.task == 'xor' and args.model == 'lstm': 
        model = models.SmallLSTM(hidden_dim=hidden_dim, num_layers=num_layers)
    elif args.task == 'xor' and args.model == 'rope':
        model = models.RoPETransformer(d_model=hidden_dim, num_layers=num_layers)
    elif args.task == 'kadane' and args.model == 'lstm': 
        model = models.KadaneLSTM(hidden_dim=hidden_dim) 
    elif args.task == 'bfs' and args.model == 'gnn': 
        model = models.SimpleMPNN(hidden_dim=hidden_dim) 
    elif args.task == 'dijkstra' and args.model == 'gnn': 
        model = models.DijkstraGNN(hidden_dim=hidden_dim, num_layers=num_layers)
    else:
        raise ValueError("Invalid task/model combination for sweep.")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 5. Train the Model using our Early Stopping trainer
    model = train_loop(model, train_loader, optimizer, args.task, device, max_epochs=100, patience=10)

    # 6. Evaluate on an Out-of-Distribution scale (5x the training size)
    ood_size = args.train_size * 5 
    
    # --- UPGRADED: Test Data also gets the Virtual Node flag ---
    if args.task == 'xor': test_data = datasets.CumulativeXORDataset(500, ood_size)
    elif args.task == 'kadane': test_data = datasets.MaxSubarrayDataset(500, ood_size)
    elif args.task == 'bfs': test_data = datasets.BFSDataset(300, ood_size, use_vn=args.use_vn)
    elif args.task == 'dijkstra': test_data = datasets.DijkstraDataset(300, ood_size, use_vn=args.use_vn)

    test_loader = DataLoader(test_data, batch_size=4)
    final_score = evaluate_loop(model, test_loader, args.task, device)

    # Log the final OOD accuracy/MAE to the cloud and close the run
    metric_name = "ood_accuracy" if args.task == 'xor' else "ood_mae"
    wandb.log({metric_name: final_score})
    run.finish()

    return final_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["xor", "bfs", "dijkstra", "kadane"])
    parser.add_argument("--model", type=str, required=True, choices=["transformer", "lstm", "gnn", "rope"])
    parser.add_argument("--train_size", type=int, default=10)
    parser.add_argument("--trials", type=int, default=10, help="How many models to test")
    # NEW: The Virtual Node Flag
    parser.add_argument("--use_vn", action="store_true", help="Enable the Virtual Node for graph datasets.")
    args = parser.parse_args()

    print(f"\n[INFO] Starting Optuna Sweep ({args.trials} trials) for {args.task.upper()} on {args.model.upper()}...")
    if args.use_vn:
        print("[INFO] VIRTUAL NODE INJECTOR: ENABLED")
    
    # Optuna needs to MAXIMIZE Accuracy, but MINIMIZE MAE
    direction = "maximize" if args.task == "xor" else "minimize"
    study = optuna.create_study(direction=direction)
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)

    print("\n[SUCCESS] Sweep Complete!")
    print("Best Hyperparameters Found:")
    print(study.best_params)