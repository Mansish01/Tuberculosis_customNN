#!/bin/bash

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python and run the script again."
    exit 1
fi

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. Installing gdown..."
    pip install gdown
fi

# Set the file ID to download
file_id="1Rz3mV7xTxC7rapDAxoozrLQKiHlTBRfT"

# Set the destination folder
destination_folder="datasets"

# Create the destination folder if it doesn't exist
if [ ! -d "$destination_folder" ]; then
    mkdir "$destination_folder"
fi

# Download the file using gdown
echo "Downloading the file..."
gdown "https://drive.google.com/uc?id=$file_id" -O "$destination_folder/file.zip"

# Unzip the file to the dataset folder
echo "Unzipping the file..."
unzip "$destination_folder/file.zip" -d "$destination_folder"


# Remove the downloaded zip file
echo "Removing the zip file..."
rm "$destination_folder/file.zip"

echo "File downloaded, unzipped, and zip file removed successfully."