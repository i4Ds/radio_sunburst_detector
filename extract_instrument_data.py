import os
import shutil

# Function to create a new directory
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

def move_images(instrument_name, folder_name):
    
    # Define the source and destination directories
    current_directory = os.getcwd()
    source_directory = os.path.join(current_directory, "data", folder_name)
    destination_directory = os.path.join(current_directory, "data", f"{instrument_name}_{folder_name}")

    # Create the destination directory
    create_directory(destination_directory)

    print("debug 1")
    # Iterate through files in the source directory
    for file_name in os.listdir(source_directory):
        # Check if the file contains the instrument name
        if instrument_name in file_name:
            # Construct full file paths
            source_file_path = os.path.join(source_directory, file_name)
            destination_file_path = os.path.join(destination_directory, file_name)
            
            # Move the file
            shutil.move(source_file_path, destination_file_path)

    print(f"Images have been moved to {destination_directory}")

def main():
    # Enter the instrument name (e.g., australia_assa_02)
    instrument_name = "australia_assa_02"
    # Enter the folder name where the images are kept (1, 2, 3, 4, 5, 6, or no_burst)
    folder_name = "no_burst"
    move_images(instrument_name, folder_name)


if __name__ == "__main__":
    main()