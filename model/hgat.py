import torch
import torch.nn as nn
from transformers import RobertaModel
from torch_geometric.nn import GATConv, global_mean_pool

class HGAT(nn.Module):
    def __init__(self, roberta_model_name='roberta-base', num_classes=2, 
                 gat_hidden=256, gat_heads=8, fused_dim=512):
        super(HGAT, self).__init__()
        
        # RoBERTa Text Encoder (outputs 768-dim)
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        text_hidden_size = self.roberta.config.hidden_size # 768
        
        # GAT Graph Encoder (input 768 from embeddings, outputs 256)
        self.gat1 = GATConv(in_channels=768, out_channels=64, heads=gat_heads, concat=True) # 64 * 8 = 512
        self.gat2 = GATConv(in_channels=512, out_channels=gat_hidden, heads=1, concat=False) # 256
        
        # Project RoBERTa and GAT to the same dimension for attention fusion
        self.text_proj = nn.Linear(text_hidden_size, fused_dim)
        self.graph_proj = nn.Linear(gat_hidden, fused_dim)
        
        # Attention Fusion Weight (Beta)
        self.attention_net = nn.Sequential(
            nn.Linear(fused_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # MLP Classifier
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, x_graph, edge_index, batch):
        # 1. Text Representation (h_text)
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding
        h_text = roberta_outputs.last_hidden_state[:, 0, :] # Shape: (batch_size, 768)
        
        # 2. Graph Representation (h_graph)
        h_g = self.gat1(x_graph, edge_index)
        h_g = torch.relu(h_g)
        h_g = self.gat2(h_g, edge_index)
        # Pool graph node embeddings into a single graph vector per article
        h_graph = global_mean_pool(h_g, batch) # Shape: (batch_size, 256)
        
        # 3. Projection
        h_text_proj = self.text_proj(h_text)   # Shape: (batch_size, 512)
        h_graph_proj = self.graph_proj(h_graph) # Shape: (batch_size, 512)
        
        # 4. Attention Fusion (compute beta)
        concat_feats = torch.cat([h_text_proj, h_graph_proj], dim=1) # Shape: (batch_size, 1024)
        beta = self.attention_net(concat_feats) # Shape: (batch_size, 1)
        
        # Fuse representations
        h_fused = beta * h_text_proj + (1 - beta) * h_graph_proj # Shape: (batch_size, 512)
        
        # 5. Classification
        logits = self.mlp(h_fused) # Shape: (batch_size, 2)
        
        return logits, beta
