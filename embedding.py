import torch
import numpy as np
from bert_model import get_tokenizer_model

# Load tokenizer and model
tokenizer, model = get_tokenizer_model()

# Attention Pooling Layer
class AttentionPooling(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = torch.nn.Linear(hidden_dim, 1)

    def forward(self, token_embeddings, attention_mask):
        attention_scores = self.attention_weights(token_embeddings).squeeze(-1)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)
        sentence_embedding = torch.sum(token_embeddings * attention_weights.unsqueeze(-1), dim=1)
        return sentence_embedding

attention_pooling = AttentionPooling(hidden_dim=768)

# Function to get attention-pooled embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    return attention_pooling(token_embeddings, attention_mask).squeeze(0).detach().numpy()

# Function to handle long text
def get_long_text_embedding(text, chunk_size=512):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    chunk_embeddings = [torch.tensor(get_embedding(chunk)) for chunk in chunks]

    if chunk_embeddings:
        return torch.mean(torch.stack(chunk_embeddings), dim=0).detach().numpy()
    else:
        return np.zeros(768)  # Return a zero vector if empty
