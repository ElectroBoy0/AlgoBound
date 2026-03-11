# main.py
import argparse
import torch
from torch.utils.data import DataLoader as StdDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

import datasets
import models
from trainer import train_loop, evaluate_loop
from utils import set_seed, save_and_plot_results

def get_parser():
    # formatter_class allows us to cleanly align the help text
    parser = argparse.ArgumentParser(
        description="AlgoBound: OOD Generalization Testbed\n"
                    "A framework for testing algorithmic reasoning limits in neural architectures.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- 1. CORE ROUTING ---
    core = parser.add_argument_group("🎯 Core Settings")
    core.add_argument("--task", type=str, required=True, choices=["xor", "bfs", "dijkstra", "kadane"], 
                      help="The algorithmic reasoning task to evaluate.")
    core.add_argument("--model", type=str, required=True, choices=["transformer", "lstm", "gnn", "rope", "all"], 
                      help="The neural architecture to test ('all' runs a comparative benchmark).")
    
    # --- 2. DATA & SCALING ---
    data = parser.add_argument_group("📊 Data & Scaling")
    data.add_argument("--train_size", type=int, default=10, 
                      help="Training dimension (sequence length or graph nodes). Default: 10")
    data.add_argument("--eval_sizes", type=int, nargs='+', default=[20, 50, 100, 200], 
                      help="List of Out-Of-Distribution sizes to test on. Default: 20 50 100 200")
    data.add_argument("--samples", type=int, default=3000, 
                      help="Number of training samples generated. Default: 3000")
    
   # --- 3. TRAINING DYNAMICS ---
    hyper = parser.add_argument_group("⚙️ Training Dynamics")
    hyper.add_argument("--epochs", type=int, default=500, 
                       help="Maximum epochs before forced cutoff. Default: 500")
    hyper.add_argument("--patience", type=int, default=15, 
                       help="Early stopping patience (epochs without improvement). Default: 15")
    # UPDATED: Default LR is now the champion's LR
    hyper.add_argument("--lr", type=float, default=0.00112, 
                       help="Learning rate for the Adam optimizer. Default: 0.00112")
    hyper.add_argument("--seed", type=int, default=42, 
                       help="Random seed for exact reproducibility. Default: 42")
    
    # --- 4. DYNAMIC ARENA CONTROLS ---
    hyper.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size.")
    hyper.add_argument("--num_layers", type=int, default=3, help="Number of neural layers.")
    hyper.add_argument("--use_vn", action="store_true", help="Inject Virtual Node for custom graphs.")
    hyper.add_argument("--compare_champion", action="store_true", help="Race custom model against dry-cherry-25.")
    
    return parser

def main():
    args = get_parser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[INFO] Starting AlgoBound Framework on {device}")
    
    # DataLoader Type
    DataLoader = PyGDataLoader if args.task in ['bfs', 'dijkstra'] else StdDataLoader

    # 1. Multi-Model Factory (Updated for Virtual Node Ablation & Arena Mode)
    models_to_run = {}
    if args.task == 'xor':
        if args.model in ['transformer', 'all']: models_to_run['Transformer (Absolute PE)'] = models.SmallTransformer()
        if args.model in ['lstm', 'all']: models_to_run['LSTM'] = models.SmallLSTM()
        if args.model in ['rope', 'all']: models_to_run['Transformer (RoPE)'] = models.RoPETransformer()
    elif args.task == 'kadane':
        if args.model in ['lstm', 'all']: models_to_run['LSTM'] = models.KadaneLSTM()
    elif args.task in ['bfs', 'dijkstra']:
        
        # --- THE ARENA MODE ---
        if args.model == 'gnn': 
            # 1. Build YOUR dynamic model from the terminal inputs
            name_tag = f"Custom GNN ({args.num_layers}L, {args.hidden_dim}D)"
            if args.use_vn: 
                name_tag += " (+Virtual Node)"
                
            models_to_run[name_tag] = models.SimpleMPNN(hidden_dim=args.hidden_dim) if args.task == 'bfs' else models.DijkstraGNN(hidden_dim=args.hidden_dim, num_layers=args.num_layers)

            # 2. Inject the Champion to race against it!
            if args.compare_champion:
                models_to_run['Champion dry-cherry-25 (+Virtual Node)'] = models.SimpleMPNN(hidden_dim=128) if args.task == 'bfs' else models.DijkstraGNN(hidden_dim=128, num_layers=12)
        
        # --- THE PUBLICATION MODE ---
        elif args.model == 'all':
            models_to_run['Shallow GNN (Baseline)'] = models.SimpleMPNN() if args.task == 'bfs' else models.DijkstraGNN(hidden_dim=64, num_layers=3)
            models_to_run['Deep GNN (+Virtual Node)'] = models.SimpleMPNN(hidden_dim=128) if args.task == 'bfs' else models.DijkstraGNN(hidden_dim=128, num_layers=12)

    if not models_to_run:
        raise ValueError(f"No valid models found for task '{args.task}' and model '{args.model}'.")

    all_results = []
    metric_name = "Accuracy" if args.task == 'xor' else "MAE"

    # 2. The Grand Loop
    for model_name, model in models_to_run.items():
        print(f"\n" + "="*50)
        print(f" EXPERIMENT: {args.task.upper()} on {model_name.upper()}")
        print("="*50)
        
        # --- THE TOGGLE ---
        # Check if this specific model name requests the Virtual Node
        use_vn_flag = "(+Virtual Node)" in model_name
        
        # Generate fresh training data with or without the Virtual Node
        if args.task == 'xor': train_data = datasets.CumulativeXORDataset(args.samples, args.train_size)
        elif args.task == 'kadane': train_data = datasets.MaxSubarrayDataset(args.samples, args.train_size)
        elif args.task == 'bfs': train_data = datasets.BFSDataset(args.samples, args.train_size, use_vn=use_vn_flag)
        elif args.task == 'dijkstra': train_data = datasets.DijkstraDataset(args.samples, args.train_size, use_vn=use_vn_flag)
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        
        model = model.to(device)
        
        # --- NEW: THE CHAMPION SHIELD ---
        current_lr = args.lr
        if "dry-cherry-25" in model_name or "Deep GNN" in model_name:
            current_lr = 0.00112  # Lock in the champion's exact DNA
            
        optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)

        print(f"--- PHASE 1: TRAINING (Size: {args.train_size} | LR: {current_lr}) ---")
        model = train_loop(model, train_loader, optimizer, args.task, device, args.epochs, args.patience)

        print("\n--- PHASE 2: OOD EVALUATION ---")
        for size in args.eval_sizes:
            if args.task == 'xor': test_data = datasets.CumulativeXORDataset(500, size)
            elif args.task == 'kadane': test_data = datasets.MaxSubarrayDataset(500, size)
            elif args.task == 'bfs': test_data = datasets.BFSDataset(300, size, use_vn=use_vn_flag)
            elif args.task == 'dijkstra': test_data = datasets.DijkstraDataset(300, size, use_vn=use_vn_flag)
                
            test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
            score = evaluate_loop(model, test_loader, args.task, device)
            
            label = "IN-DIST" if size <= args.train_size else "OOD"
            print(f"Size: {size:<6} | {label:<8} | {metric_name}: {score:.4f}")
            
            all_results.append({'model': model_name, 'eval_size': size, 'metric_value': score})

    # 3. Artifacts
    print("\n--- PHASE 3: ARTIFACT GENERATION ---")
    save_and_plot_results(all_results, args.task, "all_models", metric_name)

if __name__ == "__main__":
    main()