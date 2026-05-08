import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from torch_geometric.data import Data, Batch
import pandas as pd

class FakeNewsDataset(Dataset):
    def __init__(self, data_path=None, max_length=128):
        # Placeholder loading logic
        # In a real scenario, this would load a CSV with 'text' and 'label', 
        # and pre-compute dependency or entity graphs.
        if data_path:
            try:
                self.df = pd.read_csv(data_path)
            except Exception:
                self._load_dummy_data()
        else:
            self._load_dummy_data()
            
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_length = max_length

    def _load_dummy_data(self):
        self.df = pd.DataFrame({
            'text': ['This is real news about economy.', 'Fake news: aliens landed on earth!'] * 50,
            'label': [0, 1] * 50
        })

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = self.df.iloc[idx]['label']
        
        # Tokenize Text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Dummy Graph Construction
        # A real implementation would parse the text to extract entities as nodes
        # and relationships as edges to form the graph.
        num_nodes = 5
        x_graph = torch.randn(num_nodes, 768) # 5 nodes, 768 features
        
        # Dummy edge index (fully connected for demo)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 3, 4],
            [1, 0, 2, 1, 4, 3]
        ], dtype=torch.long)
        
        graph_data = Data(x=x_graph, edge_index=edge_index)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'graph': graph_data,
            'label': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Batch PyTorch Geometric graphs
    graphs = [item['graph'] for item in batch]
    batched_graph = Batch.from_data_list(graphs)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'graph_x': batched_graph.x,
        'graph_edge_index': batched_graph.edge_index,
        'graph_batch': batched_graph.batch,
        'labels': labels
    }
