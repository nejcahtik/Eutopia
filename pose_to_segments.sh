#!/bin/bash

# Define the directory (change this to your target directory)
directory="./dgs_corpus/poses"

# Loop through all files in the directory
for file in "$directory"/*
do
  # Check if it's a file (not a directory)
  if [ -f "$file" ]; then
    
    echo "Displaying contents of $file:"
    
    filename=$(basename "$file")
    filename_no_ext="${filename%.*}"
    
    extension=".eaf"
    
    pose_to_segments --pose $file --elan="${filename_no_ext}${extension}"

    echo
  fi
done
