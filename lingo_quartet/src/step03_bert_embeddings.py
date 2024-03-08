"""
Working from Lisette's initial BERT embeddings code.
This was run on Google CoLab

Authored by Megan (with work from Lisette)
"""

from google.colab import drive
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel  # kills kernel if run with others


# need to set device globally
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def read_data(
    path="/content/gdrive/MyDrive/AdvancedML/final_project_data/metadata_w_2020articles_cleaned.csv",
):
    """
    Read and check dataframe.

    Returns:
        df(pd.DataFrame): dataframe of 2020 articles
    """
    df = pd.read_csv(path, index_col=0)

    # print test article
    test_article = df.loc[
        df.loc[:, "uuid"] == "bcbc6bb2-406e-11ee-a96e-33dec8f414a2", :
    ]
    test_text = test_article["title_text"][0]
    print(test_text)

    return df


def get_cls_sentence(sentence, tokenizer, model):
    """
    create embeddings for each sentence inn a given text

    Inputs:
        sentence(str): sentence to embed
        tokenizer(BertTokenizer): BERT tokenizer to tokenize the sentences
        model(BertModel): BERT model to create embeddings from

    Returns:
        cls_embedding: embedding for the document/title
    """
    # Tokenize input sentence and convert to tensor
    input_ids = torch.tensor(
        [tokenizer.encode(sentence, add_special_tokens=True, max_length=512)]
    ).to(device)

    # Pass input through BERT model and extract embeddings for [CLS] token
    with torch.no_grad():
        outputs = model(input_ids)
        cls_embedding = outputs[0][:, 0, :]

    return cls_embedding.flatten()


def tokenize(df: pd.DataFrame):
    """
    Tokenize the rows in a dataframe

    Inputs:
        df(pd.DataFrame) = dataframe of documents to create embeddings for
    """
    # Load the pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model = BertModel.from_pretrained("bert-base-uncased")
    model.to(device)

    # starting with title only embeddings
    df["bert_sentence_embedding_title_only"] = df["title"].apply(
        lambda sentence: get_cls_sentence(sentence, tokenizer, model)
    )

    return df


def write_embeddings(
    df,
    filepath: str = "/content/gdrive/MyDrive/AdvancedML/final_project_data/clean_2020articles_w_title_embeddings.pkl",
):
    """
    Write out embeddings to pickle filepath since the embedding process
    can take a very long time.

    Inputs:
        df(pd.DataFrame): dataframe to write out
        filepath(str): pickle filepath to write to
    """
    # pickling so that we don't have to rerun the embedding process every time
    df.to_pickle(filepath)


if __name__ == "__main__":
    # Mount Google Drive
    drive.mount("/content/gdrive")
    df = read_data()
    df = tokenize(df)
    write_embeddings(df)
