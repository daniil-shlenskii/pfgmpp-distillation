# Usage: bash idmd/download_checkpoint.sh <Google_Drive_Share_Link> <ckpt_name> 

# Create directory if it doesn't exist
CHECKPOINTS_DIR="pfgmpp_checkpoints"
mkdir -p $CHECKPOINTS_DIR

# Extract file ID from the share link
SHARE_LINK="$1"
FILE_ID=$(echo "$SHARE_LINK" | sed -n 's/.*\/d\/\([^\/]*\)\/.*/\1/p')
echo $FILE_ID

if [ -z "$FILE_ID" ]; then
    echo "Error: Invalid Google Drive share link!"
    exit 1
fi
#
# Download the file using gdown
CKPT_NAME="$2"
SAVE_PATH="${CHECKPOINTS_DIR}/${CKPT_NAME}"
echo "Downloading checkpoint..."
gdown "https://drive.google.com/uc?export=download&id=$FILE_ID" -O $SAVE_PATH

if [ $? -eq 0 ]; then
    echo "Checkpoint saved to: $SAVE_PATH
else
    echo "Download failed! Ensure the link is correct and accessible."
    exit 1
fi
