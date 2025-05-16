import matplotlib.pyplot as plt
from tqdm import tqdm
import evaluate
import os

bleu = evaluate.load("bleu")
ter = evaluate.load("ter")


def translate_single(model, tokenizer, device, sentence, selector):
    model.eval()
    # Tokenizing Input sentence and moving to GPU
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
    if selector == "marian":
        translated = model.generate(**inputs)
    elif selector == "m2m100":
        translated = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("de"))
    else:
        print("Invalid selector value, CHECK CODE")
        return
    result = tokenizer.decode(translated[0], skip_special_tokens=True)
    return result


def translate_set(model, tokenizer, device, sentences, selector):
    model.eval()
    results = []
    for text in tqdm(sentences, desc="Translating", leave=False):
        # Tokenizing Input sentence and moving to GPU
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        if selector == "marian":
            translated = model.generate(**inputs)
        elif selector == "m2m100":
            translated = model.generate(**inputs,forced_bos_token_id=tokenizer.get_lang_id("de"))
        else:
            print("Invalid selector value, CHECK CODE")
            return
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        results.append(result)
    return results


def bleu_score(model, tokenizer, device, dataset, selector):
    source = [item["translation"]["en"] for item in dataset]
    target = [item["translation"]["de"] for item in dataset]
    target = [[sentence] for sentence in target]
    preds = translate_set(model, tokenizer, device, source, selector)
    score = bleu.compute(predictions=preds, references=target)["bleu"]
    print(f"\nBLEU score: {score:.4f}")
    return score

def ter_score(model, tokenizer, device, dataset, selector):
    source = [item["translation"]["en"] for item in dataset]
    target = [item["translation"]["de"] for item in dataset]
    target = [[sentence] for sentence in target]
    preds = translate_set(model, tokenizer, device, source, selector)
    score = ter.compute(predictions=preds, references=target)["score"]
    print(f"\nTER score: {score:.4f}")
    return score


