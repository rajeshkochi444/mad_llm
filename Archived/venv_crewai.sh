#!/bin/bash

module --force purge
#module load cuda/11.2
module load python/3.10

env_name='crewai'
python -m virtualenv ~/.virtualenvs/$env_name

source ~/.virtualenvs/$env_name/bin/activate
# Update pip
python -m pip install --upgrade pip
# Requirements to run
pip install crewai  ipywidgets ipykernel ollama vllm
#If you want to also install crewai-tools, which is a package with tools that can be used by the agents
pip install 'crewai[tools]'
