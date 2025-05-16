# English to German Neural Machine Translation using MarianMT (DL-NLP Project)

This project implements a transformer-based model for English-to-German neural machine translation, the model was a finetune of a prebuilt transformer model MarianMT by Helsinki-NLP. The models are fine-tuned on a random sample of the combination of OPUS Books and News Commentary datasets, preprocessed with custom filtering. BLEU Scores and TER and evaluated using BLEU and TER (translation error rate) were used for evaluating performance. The fine tuned version showed an increase in BLEU scores of 0.02 (2%) which is considered a solid improvement in the field of Machine Translation.

## Project Structure
 - main.py , Main File, running it will train and evaluate the MT model.
 - A/ , All Source Code:
    + marian.py , Sets up Model and Tokenizer for MarianMT
    + m2m100.py , Sets up Model and Tokenizer for M2M100 (UNUSED)
    + preproc.py , Contains functions that handle dataset download and preprocessing
    + train.py , Contains code for training loop and plotting training curves
    + utils.py , Contains code for translating batch inputs and evaluation metrics like BLEU and TRE.
- Datasets/
    + final_dataset.pkl , Contains the entire dataset in the form of .pkl file.

- Results , Used for storing loss curves and other images
- env/
    + environment.yml , Code to create a new conda env called "DLNLP" with all necessary modules
    + requirements.txt, All needed modules
- README.md , This file

## All Packages needed to run the code
- numpy
- pandas
- matplotlib
- torch
- scikit-learn
- tqdm
- transformers
- datasets
- sentencepiece
- sacrebleu
- evaluate
- accelerate

## Instructions
git clone the repo, go into the project root folder and open terminal.
Run the following code to create a new conda environment named "DLNLP" with all the necessary modules.
```bash
conda env create -f env/environment.yml
```
Go back to the project root folder and run the following code to train the MarianMT model, obtain loss plots, BLEU and TRE accuracy scores (takes over 2 hours on my laptop with an Nvidia GPU)
```bash
python3 main.py
```