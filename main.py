import gc
import torch

from A.preproc import get_dataset
from A.marian import get_marianMT_mdl_tknzr
from A.m2m100 import get_M2M100_mdl_tknzr
from A.utils import bleu_score,ter_score
from A.train import preprocess_tensor, train_model,training_plots
# ======================================================================================================================
# Data preprocessing
data = get_dataset()
print("Dataset Loaded Successfully")
data_train, data_val, data_test = data["train"] , data["validation"], data["test"]

# Cutting down dataset size further due to Resource Constraints
data_train = data_train.train_test_split(train_size=10000)["train"]
data_val = data_val.train_test_split(train_size=1000)["train"] 
data_test = data_test.train_test_split(train_size=1000)["train"]



sample_in = data_train[6]["translation"]["en"]
sample_out = data_train[6]["translation"]["de"]
print("Sample Input: ",sample_in)
print("Sample Output: ",sample_out)

print("\n--- Using MarianMT Model ---")
model, tokenizer, device = get_marianMT_mdl_tknzr()

print("Evaluating Before Fine Tuning")
bleu_init = bleu_score(model, tokenizer, device, data_test, selector="marian")

ter_init = ter_score(model, tokenizer, device, data_test, selector="marian")

print("Fine Tuning MarianMT")

data_train_tensor = preprocess_tensor(data_train, tokenizer)
data_val_tensor = preprocess_tensor(data_val, tokenizer)

train_losses, val_losses = train_model(
    model = model,
    tokenizer = tokenizer,
    train_dataset = data_train_tensor,
    val_dataset = data_val_tensor,
    selector="marian",
    num_epochs=5,
    learning_rate=1e-5,
    batch_size=16,
    weight_decay = 0.00001
)


print(train_losses)
print(val_losses)

training_plots(train_losses,val_losses,title="Marian")

print("Evaluating After Fine Tuning")
bleu_score(model,
           tokenizer, 
           device, 
           data_test, 
           selector="marian"
           )

ter_score(model,
           tokenizer, 
           device, 
           data_test, 
           selector="marian"
           )


# ======================================================================================================================
# Task A
# model_A = A(args...)                 # Build model object.
# acc_A_train = model_A.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
# acc_A_test = model_A.test(args...)   # Test model based on the test set.
# Clean up memory/GPU etc...             # Some code to free memory if necessary.


# # ======================================================================================================================

bleu_acc_A_train = bleu_score(model, tokenizer, device, data_train.train_test_split(train_size=5000)["train"], selector="marian")

bleu_acc_A_val = bleu_score(model, tokenizer, device, data_val, selector="marian")

bleu_acc_A_test = bleu_score(model, tokenizer, device, data_test, selector="marian")

ter_acc_A_train = ter_score(model, tokenizer, device, data_train.train_test_split(train_size=5000)["train"], selector="marian")

ter_acc_A_val = ter_score(model, tokenizer, device, data_test, selector="marian")

ter_acc_A_test = ter_score(model, tokenizer, device, data_test, selector="marian")


# # ======================================================================================================================
# ## Print out your results with following format:
print('TA (MarianMT BLEU Scores):{},{};'.format(bleu_acc_A_train, bleu_acc_A_test))
print('TA (MarianMT TER Scores):{},{};'.format(ter_acc_A_train, ter_acc_A_test))

print('BLEU Score Gain = ',bleu_acc_A_test - bleu_init)
print('TER Score Drop = ',ter_init - ter_acc_A_test)

# # If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# # acc_A_train = 'TBD'
# # acc_B_test = 'TBD'