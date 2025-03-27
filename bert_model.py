from transformers import AutoTokenizer, AutoModel

# Load BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def get_tokenizer_model():
    return tokenizer, model
