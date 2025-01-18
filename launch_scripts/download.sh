apt update && apt install -y unzip;
pip install gdown;

mkdir -p downloads;
gdown --folder https://drive.google.com/drive/folders/1BiK9sYyTslcxQQoS3vZGF02hRg6OQpR4 --output downloads;

# First create the data directory if it doesn't exist
mkdir -p data

# Unzip all zip files and move contents to data folder
find . -name "*.zip" -exec unzip -o {} -d data \;