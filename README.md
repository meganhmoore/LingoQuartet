# LingoQuartet
LingoQuartet: Unraveling political and linguistic themes using unsupervised learning clustering and other NLP techniques

# Background
News sources are the main medium through which people receive information on current events; therefore, the amount of coverage on an issue and positive or negative sentiment in coverage of different stories can change public opinion. This project leveraged a large dataset containing articles from the politics section of the main US news sources to implement a series of embedding processes, clustering methods, and sentiment evaluation methods in order to understand effective methods for measuring topics and sentiment in the news. 

For more information on methods and to see our results, see [our final presentation](./presentation/final_presentation.pdf).

# Code Breakdown
The work that each person did can be found in the `notebooks/` folder with each team member's name as the beginning of the files they owned. Cleaned up documented scripts can be found in the `lingo_quartet/src filepath`. Additionally, branches are prepended with names where more in-progress work may be located

Brief Notebook Descriptions: 

megans_notebook.ipynb - Exploratory data analysis and cleaning  
megans_preprocessing.ipynb – Data prepossessing for clustering  
megans_embeddings.ipynb – Creating embeddings for hierarchical clustering  
megan_k_means.ipynb - Implementation of early k-means model  
megans_hierarchical_part_2.ipynb - Implementation, tuning, and evaluation of hierarchical clustering  
jackies_notebook.ipynb – Exploratory data analysis and data prepossessing for clustering  
jackie_sentiment_preprocessing.ipynb - Data prepossessing for sentiment analysis  
jackie_sentiment_analysis.ipynb – Sentiment analysis on titles, full text, abbreviated text, etc. by source and regex derived topics  
jackie_berttopicmodel.ipynb – Implementation, tuning, and evaluation of BERTopics  



# Environment Usage
The dependencies for this project were based off of the CAPP30255 conda environment used for othe assignments in the course. In order to track dependencies that were not included in the original capp30255 base environment we use the following process from the root of the repository with the current environment activated:

To capture all conda and pip dependencies:

```
$ conda env export > ./env_yamls/<ENVIRONMENT_NAME>.yml 
```
And to create a new environment that starts with the recorded dependences
```
# make sure you dont have any environment currently activated by running
$ conda deactivate

# then create a new environment
$ conda env create -n <NEW_ENV_NAME> -f env_yamls/<OLD_ENV_NAME>.yml

# and then start using it
$ conda activate <NEW_ENV_NAME>
```
