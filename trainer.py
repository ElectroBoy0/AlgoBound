# trainer.py
import os
import torch
import torch.nn as nn
import wandb 

def _forward_pass(model, batch, task, device):
    """Dynamically handles data routing for sequence vs graph models."""
    if task in ['xor', 'kadane']:
        batch_x, batch_y = batch[0].to(device), batch[1].to(device)
        if task == 'kadane': batch_y = batch_y.squeeze(-1)
        logits = model(batch_x)
        return logits, batch_y
    else: # Graph tasks
        batch = batch.to(device)
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        logits = model(batch.x, batch.edge_index, edge_attr)
        return logits, batch.y

def train_loop(model, dataloader, optimizer, task, device, max_epochs, patience=20):
    is_classification = (task == 'xor')
    criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()
    
    best_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/best_model_{task}.pt"

    model.train()
    for epoch in range(max_epochs):
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            logits, targets = _forward_pass(model, batch, task, device)
            
            if is_classification:
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                loss = criterion(logits, targets)
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        
        # ==========================================
        # 2. THE CLOUD HOOK (THIS IS WHAT DRAWS THE GRAPH)
        # ==========================================
        if wandb.run is not None:
            wandb.log({"train/loss": avg_loss, "train/epoch": epoch})
            
        # Early Stopping Logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>3}/{max_epochs} | Loss: {avg_loss:.4f} | Early Stop Count: {epochs_no_improve}/{patience}")
            
        if epochs_no_improve >= patience:
            print(f"[INFO] Converged! Early stopping triggered at epoch {epoch+1}.")
            break
            
    # Load best weights before returning
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model

def evaluate_loop(model, dataloader, task, device):
    is_classification = (task == 'xor')
    model.eval()
    
    total_error = 0
    total_elements = 0
    
    with torch.no_grad():
        for batch in dataloader:
            logits, targets = _forward_pass(model, batch, task, device)
            
            if is_classification:
                preds = torch.argmax(logits.view(-1, logits.size(-1)), dim=1)
                correct = (preds == targets.view(-1)).sum().item()
                total_error += correct 
            else:
                mae = torch.abs(logits - targets).sum().item()
                total_error += mae
                
            total_elements += targets.numel()
            
    if is_classification:
        return (total_error / total_elements) * 100.0 
    else:
        return total_error / total_elements