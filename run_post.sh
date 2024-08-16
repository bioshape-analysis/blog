#!/bin/bash

# Select the directory of the post to run. This directory should have :
#       - a package requirement file requirements.txt
#       - one or several quarto files 
#       - if you have a virtual environment, it should be named .venv/
#       - this code create or update the environment and then run the quarto in the folder post_path

# Name of Directories
current_path=`pwd`
post_path=posts/AFM-data

# Going to the post directory
cd $post_path

# Checking if the environment exists, or creating it, and activating it
if test -d .venv
then
    source .venv/bin/activate
else
    python -m venv .venv
    source .venv/bin/activate
fi
# Installing required packages
pip install -r requirements.txt
# Running quarto
quarto render
# Comming back to the main directory
cd $current_path
# Deactivating environment
deactivate