#!/bin/bash

# Define the destination directory
DEST_DIR="all_wavs"

# Loop through all directories in the current location that start with "Bird"
for dir in Bird*/Wave/; do
    
    # Check if it is an actual directory (in case no "Bird*" folders exist)
    [ -d "$dir" ] || continue
    
    # Remove the trailing slash from the directory name for cleaner file naming
    dir_name="$(echo "$dir" | cut -d'/' -f1)"

    # Loop through all .wav files in the current "Bird" directory
    for file in "$dir"*.wav; do
        
        # Check if it is an actual file (in case the directory has no .wav files)
        [ -f "$file" ] || continue
        
        # Extract just the file name without the path
        base_name=$(basename "$file")
        
        # Define the new filename: DirectoryName_OriginalFileName.wav
        new_name="${dir_name}_${base_name}"
        
        # Copy the file to the destination directory with the new name
        cp "$file" "$DEST_DIR/$new_name"
        
        echo "Copied: '$file' -> '$DEST_DIR/$new_name'"
    done
done

echo "Done! All .wav files have been copied to '$DEST_DIR'."
