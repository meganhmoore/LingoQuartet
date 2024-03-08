"""
This script conducted exploratory analysis on the initial news corpus to help
inform our modeling strategy and scaling considerations.

Authored by Megan
"""

import json
import matplotlib.pytplot as plt
from types import Dict


def get_data(filepath: str = "../data/articles.json"):
    """
    By default this pulls in the 2GB file containing the politics news
    articles, titles and images for 500,000+ news articles gathered by
    James Turk, Katheryn and JP.

    Inputs:
        filepath (str): file path to data source

    Returns:
        data (Dict): dictionary of articles by id
    """
    with open(filepath) as json_file:
        data = json.load(json_file)

    print(data[0].keys())

    # 512,000 articles
    print(len(data))

    # to get a sense of the structure and an example article
    print(data[0])

    return data


def url_evaluation(data: Dict):
    """
    Evaluate the sources by gathering urls from the images associated with
    each article and processing them. First need to check that there are urls
    for each article.

    Input:
        data(Dict): dictionary from read in json dataset.
    """
    # map article ids to the urls within them
    ids_to_urls_all = {}
    num_urls = []
    for article in data:
        ids_to_urls_all[article["uuid"]] = []
        num_urls.append(len(article["images"]))
        for image in article["images"]:
            ids_to_urls_all[article["uuid"]].append(image["url"])

    # Checking if articles have no images associated
    # (and therefore no urls for source detection)
    print(f"Min url count: {min(num_urls)}")
    print(f"Max url count: {max(num_urls)}")
    uuids = list(ids_to_urls_all.keys())
    print(uuids[245009])  # this article has 174 images/urls
    # ids_to_urls['cf701143-3d5e-11ee-a96e-33dec8f414a2']

    # Check for imageless articles
    imageless_articles = []
    for key in ids_to_urls_all:
        if len(ids_to_urls_all[key]) == 0:
            imageless_articles.append(key)

    print(
        f"There are {len(imageless_articles)} articles with no images"
        " and therefore no urls associated so we cannot detect the source."
    )

    percent_sourceless_articles = len(imageless_articles) / len(num_urls)
    print(
        f"{percent_sourceless_articles}% of the artcles do not have "
        "identifiable urls and therefore we cannot identify the source"
        " or the year that it was published"
    )

    return ids_to_urls_all


def plot_sources(source_options: Dict):
    """
    Plot the sources that we found articles for and the number of articles
    for each source.

    Inputs:
        source_options(Dict): dictionary of sources and the count of articles
        we found associated with them
    """
    values = list(source_options.values())
    labels = list(source_options.keys())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color="skyblue")
    plt.xlabel("News Sources")
    plt.ylabel("Number of Articles")
    plt.title("Number of Articles per News Source")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_years(year_counts: Dict):
    """
    Plot the years that we found articles for and the number of articles
    for each year.

    Inputs:
        year_counts(Dict): dictionary of years and the count of articles
            we found associated with that year
    """
    values = list(year_counts.values())
    labels = list(year_counts.keys())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color="skyblue")
    plt.xlabel("Year")
    plt.ylabel("Number of Articles")
    plt.title("Number of Articles per Year")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def parse_urls(data: Dict):
    """
    Evaluate the articles that have urls to parse useful source and date
    information from them.

    Input:
        data(Dict): dictionary from read in json dataset.
    """
    # collect all of the articles that do have urls connected so that we can classify their sources
    ids_with_urls = {}
    for article in data:
        if len(article["images"]) > 0:
            ids_with_urls[article["uuid"]] = []
            for image in article["images"]:
                ids_with_urls[article["uuid"]].append(image["url"])

    print(f"Working with {len(list(ids_with_urls.keys()))} usable articles")

    # getting a sense of the url formats and which sources we are working with
    source_options = {
        "breitbart": 0,
        "wp": 0,
        "thehill": 0,
        "fox": 0,
        "dailycaller": 0,
        "cnn": 0,
        "bbc": 0,
        "politico": 0,
        "washtimes": 0,
        "npr": 0,
        "us-east-1.prod.boltdns": 0,
        "apnews": 0,
        "timesofisrael": 0,
    }
    year_options = [
        "2012",
        "2013",
        "2014",
        "2015",
        "2016",
        "2017",
        "2018",
        "2019",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
    ]

    ids_and_sources = {}
    year_counts = {year: 0 for year in year_options}
    count_bad_urls = 0
    count_articles = 0

    def split_for_year(url_val: str):
        """
        Given a string value for the url, parse apart the year that it should
        be associated with.

        Inputs:
            url_val(str): url from an article that helps to identify the source

        Returns:
            str: string corresponding to the year, otherwise an empty string
        """
        url_els = url_val.split("/")
        for el in url_els:
            if el in year_options:
                year_counts[el] += 1
                return el
        return ""

    for key in ids_with_urls:
        first_source = ids_with_urls[key][0]
        if "://" not in first_source:
            count_bad_urls += 1
        else:
            count_articles += 1
            source = first_source.split(".com")[0].split("://")[1]
            year = split_for_year(first_source)
            found = False
            for opt in source_options.keys():
                if opt in source:
                    ids_and_sources[key] = {"source": opt, "year": year}
                    source_options[opt] += 1
                    found = True
                    break
            if not found:
                print(source)  # sources that are unaccounted for in list

    print(f"Found the source for: {count_articles} articles")
    print(f"Could not find the source for: {count_bad_urls} articles")
    print(f"Count of articles by source {print(source_options)}")
    print(f"Count of articles by year: {year_counts}")

    plot_sources(source_options)


def assess_article_lengths(data: Dict, ids_and_sources: Dict):
    """
    Assess the lengths of the articles to get a sense of how long and how much
    text we are working with and too check they aren't empty

    Input:
        data(Dict): dictionary from read in json dataset.
        ids_and_sources(Dict): dictionary of the ids and the associated
            articles
    """
    article_lens = []
    articles_with_no_text = []
    all_article_lens = []
    for article in data:
        art_uuid = article["uuid"]
        if art_uuid in ids_and_sources.keys():
            ids_and_sources[art_uuid]["article_text"] = article["article_text"]
            ids_and_sources[art_uuid]["title"] = article["title"]
        article_length = len(article["article_text"])
        all_article_lens.append(article_length)
        if article_length == 0:
            articles_with_no_text.append(article["uuid"])
        else:
            if article_length > 100000:
                print(article_length)  # make note of extemely long articles
            else:
                article_lens.append(article_length)
    print(f"Number of articles with no text {len(articles_with_no_text)}")
    print(f"Average article lengths: " f"{sum(all_article_lens)/len(all_article_lens)}")
    print(f"Maximum length of the articles: {max(all_article_lens)}")


def plot_article_lengths(article_lens: Dict):
    """
    Plot the distribution of article lengths

    Input:
        article_lens(Dict): length of the articles
    """
    plt.hist(article_lens, bins=100, color="skyblue", edgecolor="black")
    plt.xlim(right=40000)
    plt.xlabel("Length of Article")
    plt.ylabel("Number of articles of this length")
    plt.title("Histogram of Article Lengths")
    plt.show()


def get_articles_from_2020(ids_and_sources: Dict):
    """
    Subset articles for only 2020

    Input:
        ids_and_sources(Dict): dictionary of the ids and the associated
            articles
    """
    articles_from_2020 = {}
    for uuid_ind in ids_and_sources.keys():
        if "year" in ids_and_sources[uuid_ind].keys():
            if ids_and_sources[uuid_ind]["year"] == "2020":
                articles_from_2020[uuid_ind] = ids_and_sources[uuid_ind]
                articles_from_2020[uuid_ind]["full_article_text"] = articles_from_2020[
                    uuid_ind
                ]["article_text"]
                if len(articles_from_2020[uuid_ind]["article_text"]) > 512:
                    articles_from_2020[uuid_ind]["article_text"] = articles_from_2020[
                        uuid_ind
                    ]["article_text"][:512]


def write_out_dataset(
    ids_and_sources: Dict,
    articles_from_2020: Dict,
    filepath_all: str = "../data/cleaned_articles.json",
    filepath_2020: str = "../data/metadata_w_2020articles.json",
):
    """
    Write out cleaned up datasets.

    Inputs:
        ids_and_sources(Dict): dictionary of all article metadata
        articles_from_2020(Dict): dictionary of the 2020 articles
        filepath_all(str): filepath to write the article metadata to
        filepath_2020(str): filepath to write the 2020 articles to
    """
    with open(filepath_all, "w") as fp:
        json.dump(ids_and_sources, fp)

    with open(filepath_2020, "w") as fp:
        json.dump(articles_from_2020, fp)
