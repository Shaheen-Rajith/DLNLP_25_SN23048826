from transformers import MarianMTModel, MarianTokenizer
import torch

def get_marianMT_mdl_tknzr():
    model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.config.dropout = 0.3 # default is 0.1
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device
