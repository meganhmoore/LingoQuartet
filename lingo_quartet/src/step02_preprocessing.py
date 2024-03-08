"""
Preprocessign to clean and prep docs for embeddings.

Authored by Megan, building off of Jackie's preprocessing script
"""

import chardet
import pandas as pd
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


def get_data(filepath: str = "../data/metadata_w_2020articles.json"):
    """
    Read in the 2020 article data:

    Inputs:
        filepath(str): filepath to read data from

    Returns:
        df(pd.DataFrame): cleaned up pandas dataframe
    """
    with open(filepath, "rb") as f:
        result = chardet.detect(f.read())
        print(result)

    df = pd.read_json(filepath)

    df = df.T.reset_index().rename(columns={"index": "uuid"})
    return df


def process_data(df: pd.DataFrame):
    """
    Take the data and process it in preparation for creating embeddings.

    Inputs:
        df(pd.DataFrame): main dataframe of articles
    """
    # lowercase text
    df["title"] = df["title"].str.lower()
    df["article_text"] = df["article_text"].str.lower()

    # remove certain characters from title
    df["title"] = df["title"].apply(lambda x: re.sub(r"[\n\t\r]", "", x))

    # concatenate text and title and reshorten
    df["title_text"] = (df["title"] + " " + df["article_text"]).apply(lambda x: x[:512])

    # checking that they were shortened on the right dimension
    test_val = df.loc[df.loc[:, "uuid"] == "bcbc6bb2-406e-11ee-a96e-33dec8f414a2", :]
    len(test_val["title_text"][0]) == 512

    return df


def write_cleaned_df(
    df: pd.DataFrame, filepath: str = "../data/metadata_w_2020articles_cleaned.csv"
):
    """
    Write out the processed data

    Inputs:
        df(pd.DataFrame): main dataframe of articles
        filepath(str): filepath to write cleaned dataframe to
    """
    # write out cleaned version without lemmatization in case that made things weird
    df.to_csv(filepath)


# Jackie's lemmatizer
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(token) for token in word_tokens]

    return " ".join(lemmatized_text)


def create_lemmatized_df(
    df: pd.DataFrame, filepath: str = "../data/metadata_w_2020articles_lemmatized.csv"
):
    lemmatized_df = df.copy()
    lemmatized_df["title_text"] = lemmatized_df["title_text"].apply(lemmatize)
    lemmatized_df["article_text"] = lemmatized_df["article_text"].apply(lemmatize)
    lemmatized_df["title"] = lemmatized_df["title"].apply(lemmatize)

    # write out lemmatized version to save time
    lemmatized_df.to_csv(filepath)


if __name__ == "__main__":
    df = get_data()
    clean_df = process_data(df)
    write_cleaned_df(df)
    create_lemmatized_df(clean_df)
