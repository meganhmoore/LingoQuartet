# LingoQuartet
LingoQuartet: Unraveling political and linguistic themes using unsupervised learning clustering and other NLP techniques

# Background
News sources are the main medium through which people receive information on current events; therefore, the amount of coverage on an issue and positive or negative sentiment in coverage of different stories can change public opinion. This project leveraged a large dataset containing articles from the politics section of the main US news sources to implement a series of embedding processes, clustering methods, and sentiment evaluation methods in order to understand effective methods for measuring topics and sentiment in the news. 

For more information on methods and to see our results, see [our final presentation](./presentation/final_presentation.pdf).

# Code Breakdown





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