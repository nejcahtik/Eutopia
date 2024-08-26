#!/bin/bash

# Define the directory (change this to your target directory)
directory="./videos"

# Loop through all files in the directory
for file in "$directory"/*
do
  # Check if it's a file (not a directory)
  if [ -f "$file" ]; then
  
  
    echo $file
    
    filename=$(basename "$file")
    filename_no_ext="${filename%.*}"
    
    extension=".pose"
    
    if [ -f "$file" ] && [[ "$file" == *.mov ]]; then
    	mov="_mov"
    	filename_no_ext=$filename_no_ext$mov
    fi
    
    video_to_pose -i $file --format mediapipe -o "${filename_no_ext}${extension}"

  fi
done
