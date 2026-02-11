#!/bin/bash
set -e

# Arguments received from the caller
SPEAKER_ID=$1
AUDIO_PATH=$2
TASK_ID=$3

echo "==============================================="
echo "[WSL] generate_animnet.sh started"
echo "[WSL] Received Arguments:"
echo "  - Speaker ID: $SPEAKER_ID"
echo "  - Audio Path: $AUDIO_PATH"
echo "  - Task ID:    $TASK_ID"
echo "==============================================="

# 1. Define Project Root
PROJECT_ROOT="/mnt/d/Downloads/psfa-2022-main"

# 2. Check Project Root
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "[WSL] Error: Project root directory not found at: $PROJECT_ROOT"
    exit 10
fi

cd "$PROJECT_ROOT" || { echo "[WSL] Error: Failed to cd into $PROJECT_ROOT"; exit 11; }
echo "[WSL] Changed working directory to: $(pwd)"

# 3. Activate Conda Environment
CONDA_BASE=""
if [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
elif [ -d "/home/sdk/miniconda3" ]; then
    CONDA_BASE="/home/sdk/miniconda3"
elif [ -d "/home/sdk/anaconda3" ]; then
    CONDA_BASE="/home/sdk/anaconda3"
else
    CONDA_EXE=$(which conda 2>/dev/null)
    if [ -n "$CONDA_EXE" ]; then
        CONDA_BIN_DIR=$(dirname "$CONDA_EXE")
        CONDA_BASE=$(dirname "$CONDA_BIN_DIR")
    fi
fi

if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "[WSL] Found conda at $CONDA_BASE"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "[WSL] Warning: Could not find conda.sh. Hoping conda is in PATH and initialized."
fi

echo "[WSL] Activating conda environment 'psfa'..."
conda activate psfa || { echo "[WSL] Error: Failed to activate environment 'psfa'"; exit 13; }

source scripts/functions.sh

# Disable wgpu in docker WGPU FILAMENT VULKAN SKIP
export MIKU_RENDER_BACKEND=WGPU

# Speaker & exp.
speaker=$SPEAKER_ID
exp=decmp
exp_name=animnet-decmp

# Use the speaker-specific config
test_media_config=""
if [ "$speaker" = "m001_trump" ]; then
    test_media_config="gen_$speaker"
else
    test_media_config="regen_$speaker"
fi
    # gen_m001_trump
    # regen_f000_watson
    # regen_f001_clinton
    # regen_m000_obama
    # regen_m002_taruvinga
    # regen_m003_iphoneXm0

echo "[WSL] Running GenAnimNet..."

# Generate 3D results.
# We override model.test_media.misc to process the input audio
# Structure: [[folder_name, item_name, audio_path]]
# We use TASK_ID as folder name to isolate outputs
GenAnimNet --generating --data_src=celebtalk --wanna_fps=25 \
  --exp=$exp \
  --exp_name=$exp_name \
  --load='epoch_50.pth' \
  --speaker="$speaker" \
  --test_media="$test_media_config" \
  --dump_offsets \
  --dump_audio \
  model.visualizer.video_grid_size=512 \
  "model.test_media.misc=[['$TASK_ID','input','$AUDIO_PATH']]" \
;

# Check and Output Result Path
GENERATED_BASE="runs/anime/$speaker/$exp_name/generated"

# Find the directory that ends with the Task ID
PARENT_DIR=$(find "$GENERATED_BASE" -type d -name "*$TASK_ID" -print -quit)

if [ -n "$PARENT_DIR" ]; then
    OUTPUT_DIR="$PROJECT_ROOT/$PARENT_DIR/input"
else
    # Fallback to what we tried before, though likely to fail if find failed
    OUTPUT_DIR="$PROJECT_ROOT/$GENERATED_BASE/$TASK_ID/input"
fi

if [ -d "$OUTPUT_DIR" ]; then
    echo "==============================================="
    echo "[WSL] Success: Animation Network Generation Completed."
    echo "[WSL] Output Directory: $OUTPUT_DIR"
    echo "[WSL] Key Files:"
    if [ -f "$OUTPUT_DIR/dump-offsets-final.npy" ]; then
        echo "  - Offsets: $OUTPUT_DIR/dump-offsets-final.npy"
    fi
    if [ -f "$OUTPUT_DIR/audio.wav" ]; then
        echo "  - Audio:   $OUTPUT_DIR/audio.wav"
    fi
    echo "==============================================="

    # Prepare and print JSON result for Python caller
    OFFSETS_PATH=""
    if [ -f "$OUTPUT_DIR/dump-offsets-final.npy" ]; then
        OFFSETS_PATH="$OUTPUT_DIR/dump-offsets-final.npy"
    fi
    AUDIO_PATH_OUT=""
    if [ -f "$OUTPUT_DIR/audio.wav" ]; then
        AUDIO_PATH_OUT="$OUTPUT_DIR/audio.wav"
    fi

    export RES_OUTPUT_DIR="$OUTPUT_DIR"
    export RES_OFFSETS_PATH="$OFFSETS_PATH"
    export RES_AUDIO_PATH="$AUDIO_PATH_OUT"

    python -c "import json, os; print('__RESULT__ ' + json.dumps({
        'output_dir': os.environ.get('RES_OUTPUT_DIR'),
        'key_files': {
            'offsets': os.environ.get('RES_OFFSETS_PATH'),
            'audio': os.environ.get('RES_AUDIO_PATH')
        }
    }))"
else
    echo "==============================================="
    echo "[WSL] Warning: Output directory not found at expected location:"
    echo "[WSL] $OUTPUT_DIR"
    echo "[WSL] Please check the logs above for potential errors."
    echo "==============================================="
    # We don't exit with error here because sometimes structure might differ, 
    # but strictly speaking if it failed it should have exit code != 0 above.
fi

echo "[WSL] generate_animnet.sh finished successfully."
