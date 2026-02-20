#!/usr/bin/env bash

# Check if the script is run with an argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <version_string>"
  echo "Example: $0 v3"
  exit 1
fi

# Assign the argument to a variable
VERSION=$1

# Define source and destination paths
SRC_PATH1="~/.ssh/ adidishe@spartan.hpc.unimelb.edu.au:/data/projects/punim2039/RevealedLLM/res/batch_process/*${VERSION}*"
SRC_PATH2="~/.ssh/ adidishe@spartan.hpc.unimelb.edu.au:/data/projects/punim2039/RevealedLLM/res/batch_process_secondary/*${VERSION}*"
DST_PATH1="/Users/adidisheim/Dropbox/Melbourne/research/RevealedLLM/res/batch_process/"
DST_PATH2="/Users/adidisheim/Dropbox/Melbourne/research/RevealedLLM/res/batch_process_secondary/"

# Execute the scp commands
scp -r $SRC_PATH1 $DST_PATH1
scp -r $SRC_PATH2 $DST_PATH2
