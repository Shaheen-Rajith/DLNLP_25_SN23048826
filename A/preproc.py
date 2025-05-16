import os
import pickle
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("Datasets", "final_dataset.pkl")

def keep(sample):
    src = sample["translation"]["en"]
    tgt = sample["translation"]["de"]
    #discard if either is empty
    if not src.strip() or not tgt.strip():
        return False
    #discard if either is too short (below 3 words)
    if len(src.split()) < 3 or len(tgt.split()) < 3:
        return False
    #discard if either is too long (over 100 words)
    if len(src.split()) > 100 or len(tgt.split()) > 100:
        return False
    #discard if source and target are identical (faulty data sample)
    if src.strip().lower() == tgt.strip().lower():
        return False
    return True


def get_dataset():
    if os.path.exists(DATA_PATH):
        print("Loading cached dataset from Datasets/final_dataset.pkl")
        with open(DATA_PATH, "rb") as f:
            return pickle.load(f)
    else:
        print("Downloading and Cleaning Datasets")
        # Loading OPUS Books (originally de > en)
        books = load_dataset("opus_books", "de-en")["train"]
        # Loading News Commentary (originally de > en)
        news = load_dataset("Helsinki-NLP/news_commentary", "de-en")["train"]

        #Removing unwanted samples as per mentioned criteria
        books_clean = [s for s in books if keep(s)]
        news_clean = [s for s in news if keep(s)]

        #concatenating both datasets and splitting 
        #into train/val/test set in a 70/15/15 ratio
        combined = books_clean + news_clean

        train, temp = train_test_split(combined, train_size=0.7, random_state=37)
        val, test = train_test_split(temp, test_size=0.5, random_state=37)
        final_dataset = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test)
        })

        #Saving final dataset in Datasets Folder
        os.makedirs("../Datasets", exist_ok=True)
        with open(DATA_PATH, "wb") as f:
            pickle.dump(final_dataset, f)
        print("Dataset saved at Datasets/final_dataset.pkl")
        return final_dataset
