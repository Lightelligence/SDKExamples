#!/bin/bash

eval "$(conda shell.bash hook)"

if [[ "$1" == "update" ]]
then
    # Update the environment
    conda env update -f environment.yml --prune

else
    if [[ "$1" == "clean" ]]
    then
        # Remove if already exists
        conda remove --name lightelligence-sdk --all
    fi
    # Install conda environment
    conda env create -f environment.yml
fi

# Always update to newest version of our sdk
conda activate lightelligence-sdk

# Add SDKExamples to our python path
python setup.py develop

# Install npm dependencies
npm_deps=("prettier@1.19.1")

for package in ${npm_deps[@]}; do
    npm install -g $package;
done
