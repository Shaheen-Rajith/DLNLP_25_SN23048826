import matplotlib.pyplot as plt
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from tqdm import tqdm
import os


def tokenize(sample, tokenizer, max_length=128):
    source_text = sample["translation"]["en"]
    target_text = sample["translation"]["de"]
    tokenized = tokenizer(
        source_text,
        text_target=target_text,
        padding="max_length",
        truncation=True, #maybe
        max_length=max_length,
    )
    return tokenized

def preprocess_tensor(dataset, tokenizer, max_length=128):
    def wrap(sample):  
        return tokenize(sample, tokenizer, max_length)
    tokenized_dataset = dataset.map(wrap, batched=False)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_dataset

def train_model(model,
                tokenizer,
                train_dataset,
                val_dataset,
                selector,           
                num_epochs,
                learning_rate, 
                batch_size=16,
                weight_decay = 0.1
                ):
    save_path = os.path.join("A", "Models", f"{selector}_finetuned")
    os.makedirs(save_path, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=save_path,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        save_strategy="no",
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        logging_dir=os.path.join(save_path, "logs"),
        predict_with_generate=True,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model + tokenizer saved to {save_path}")
    log_history = trainer.state.log_history
    train_losses = [entry["loss"] for entry in log_history if "loss" in entry]
    val_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]
    return train_losses, val_losses

def training_plots(train_losses,val_losses,title):
    plt.figure(figsize=(10,6))
    plt.plot([0.8 * i for i in range(1, len(train_losses) + 1)], train_losses, label='Training Loss', marker='o')
    plt.plot(range(1,len(val_losses)+1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training And Validation Losses over Epochs')
    plt.legend()
    plt.grid(True)
    save_path = "Results/"+title+".png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Training Loss plot obtained and saved at: {save_path}")