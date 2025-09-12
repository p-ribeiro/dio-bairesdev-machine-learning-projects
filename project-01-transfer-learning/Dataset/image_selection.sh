#!/bin/bash

count=0
for file in *; do
  # Check if it's a file
  [ -f "$file" ] || continue
   
  count=$((count + 1)) 
  # Get file type (MIME-like description)
  type=$(file --brief --mime-type "$file")

  # Get file extension
  extension="${file##*.}"

  printf -v padded_count "%03d" "$count"

  # Print basic file type
  echo "File: $file"
  echo "Type: $type"
  echo "Extension: $extension"
  
  # If it's an image, get its dimensions
  if [[ $type == image/* ]]; then
    size=$(identify -format "%wx%h" "$file" 2>/dev/null)
    echo "Image size: $size"
    mv -- "$file" "jdm_${padded_count}_${size}.${extension}"
  fi
  echo $count
  echo "---"
done

