# LingoQuartet
LingoQuartet: Unraveling political and linguistic themes using unsupervised learning clustering and other NLP techniques



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