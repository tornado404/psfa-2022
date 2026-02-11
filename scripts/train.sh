#!/bin/bash
set -e

# * ---------------------------------------------------------------------------------------------------------------- * #
# * Usage: ./scripts/train.sh <VIDEO_PATH> <SPEAKER_ID>
# * Example: ./scripts/train.sh /path/to/user_video.mp4 my_user
# * ---------------------------------------------------------------------------------------------------------------- * #

VIDEO_PATH=$1
SPEAKER_ID=$2

if [ -z "$VIDEO_PATH" ] || [ -z "$SPEAKER_ID" ]; then
  echo "Usage: $0 <VIDEO_PATH> <SPEAKER_ID>"
  echo "  VIDEO_PATH: Path to the user's input video file"
  echo "  SPEAKER_ID: Unique ID for the user (e.g., user01)"
  exit 1
fi

PROJECT_ROOT="/mnt/d/Downloads/psfa-2022-main"
DATA_SRC="celebtalk" # Using celebtalk structure as default
DATA_ROOT="$PROJECT_ROOT/assets/datasets/talk_video/$DATA_SRC/data/$SPEAKER_ID"

echo "=========================================================="
echo "Start Pipeline for Speaker: $SPEAKER_ID"
echo "Video Input: $VIDEO_PATH"
echo "=========================================================="

# 1. Prepare Data Directory
echo "[Step 1] Preparing Data Directory..."
if [ ! -d "$DATA_ROOT" ]; then
    echo "Creating directory: $DATA_ROOT"
    mkdir -p "$DATA_ROOT"
fi

# Copy video to the expected location (assuming structure required by tracking/training)
# Usually training expects a specific folder structure. 
# We place the video in the root of the speaker data folder for reference.
cp "$VIDEO_PATH" "$DATA_ROOT/video.mp4"

# 2. 3D Face Reconstruction (FLAME + tracking)
echo "[Step 2] 3D Face Reconstruction (FLAME + tracking)..."
FITTED_DIR="$DATA_ROOT/fitted"

if [ ! -d "$FITTED_DIR" ]; then
    echo "[WARNING] 'fitted' directory not found at $FITTED_DIR"
    echo "The 3D face reconstruction (tracking) step requires an external tool or script."
    echo "Please ensure FLAME parameters are generated and placed in '$FITTED_DIR'."
    echo "Expected structure: $FITTED_DIR/<clip_name>/<frame_index>.npz (containing 'cam', 'rot', 'tsl', etc.)"
    echo "Stopping pipeline here as training cannot proceed without tracked data."
    exit 1
else
    echo "Found 'fitted' data at $FITTED_DIR. Proceeding..."
fi

# 3. Data Processing & Joint Model Training
# Logic: User Data (Target Style) + VOCASET (Content Supplement)
# This is handled by TrainAnimNetDecmp which uses 'talk_video' (User) and 'vocaset' (Content) by default.
echo "[Step 3] Training Joint Model (User Style + VOCASET Content)..."

# Source environment
source scripts/functions.sh

# Check/Activate Conda
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "psfa" ]; then
    echo "Activating conda environment 'psfa'..."
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/home/sdk/miniconda3/etc/profile.d/conda.sh" ]; then
        source "/home/sdk/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    conda activate psfa || echo "Warning: Failed to activate psfa"
fi

# Execute Training
# We use TrainAnimNetDecmp as it trains the decomposition network (AnimNet)
# Arguments:
# --speaker=$SPEAKER_ID: Sets the target speaker
# --data_src=$DATA_SRC: Sets the data source
# Defaults in functions.sh ensure VOCASET is included as content supplement.
TrainAnimNetDecmp --speaker="$SPEAKER_ID" --data_src="$DATA_SRC"

# 4. Extract Style Code / Output Model
echo "[Step 4] Pipeline Completed."
CHECKPOINT_DIR="$PROJECT_ROOT/runs/anime/$SPEAKER_ID/animnet-decmp/checkpoints"

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Training finished."
    echo "The trained model (containing the user's style code) is available at:"
    ls -t "$CHECKPOINT_DIR"/*.pth | head -n 1
    
    BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/*.pth | head -n 1)
    echo ""
    echo "To generate video using this model, run:"
    echo "./scripts/generate.sh --load='$(basename "$BEST_CKPT")' --speaker='$SPEAKER_ID'"
else
    echo "[ERROR] Checkpoint directory not found. Training might have failed."
    exit 1
fi
