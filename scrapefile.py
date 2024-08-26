import os

def rename_files_to_lowercase(directory):
    for filename in os.listdir(directory):
        # Construct full file path
        old_file_path = os.path.join(directory, filename)

        # Skip if it's not a file
        if os.path.isfile(old_file_path):
            # Create new file name with all lower case letters
            new_filename = filename.lower()
            new_file_path = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(old_file_path, new_file_path)

rename_files_to_lowercase("./data_SSL/augmented_videos/")