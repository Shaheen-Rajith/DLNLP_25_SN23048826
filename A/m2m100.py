from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

def get_M2M100_mdl_tknzr():
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    # Setting input language since M2M is multilingual
    tokenizer.src_lang = "en"
    tokenizer.tgt_lang = "de"
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device