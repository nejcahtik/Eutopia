videos_directory="./augmented_videos/"

# Iterate over each .mp4 file in the directory
for file in "$videos_directory"*.mp4; do
    # Use ffmpeg to extract the first 160 frames and overwrite the original video file
    ffmpeg -y -i "$file" -vf "select=lt(n\,160)" -vsync vfr "$file.tmp.mp4" && mv "$file.tmp.mp4" "$file"
done

echo "Trimming and overwriting complete."
