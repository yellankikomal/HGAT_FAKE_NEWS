import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from hgat import HGAT
from dataset import FakeNewsDataset, collate_fn
from tqdm import tqdm

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Dataset & DataLoader
    train_dataset = FakeNewsDataset()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    # Initialize Model
    model = HGAT().to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    epochs = 3
    
    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # For demonstration we just run a few batches
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            x_graph = batch['graph_x'].to(device)
            edge_index = batch['graph_edge_index'].to(device)
            graph_batch = batch['graph_batch'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, beta = model(input_ids, attention_mask, x_graph, edge_index, graph_batch)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")

    print("Training Complete. Saving model...")
    # In a real scenario, uncomment below:
    # torch.save(model.state_dict(), 'hgat_model.pth')

if __name__ == '__main__':
    train()
