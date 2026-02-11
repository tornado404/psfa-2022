#!/bin/bash
set -e
source scripts/functions.sh

# * ---------------------------------------------------------------------------------------------------------------- * #
# * Usage: ./scripts/train_animnet.sh <SPEAKER_ID> [DATA_SRC]
# * Example: ./scripts/train_animnet.sh obama celebtalk
# * ---------------------------------------------------------------------------------------------------------------- * #

SPEAKER_ID=$1
DATA_SRC=${2:-celebtalk}

if [ -z "$SPEAKER_ID" ]; then
  echo "Usage: $0 <SPEAKER_ID> [DATA_SRC]"
  echo "  SPEAKER_ID: The ID of the speaker to train (e.g., obama)"
  echo "  DATA_SRC:   The data source name (default: celebtalk)"
  exit 1
fi

echo "=========================================================="
echo "Start training AnimNet (Decmp) for speaker: $SPEAKER_ID"
echo "Data Source: $DATA_SRC"
echo "=========================================================="

# Check/Activate Conda Environment (Optional, consistent with other scripts)
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "psfa" ]; then
    echo "Warning: Conda environment 'psfa' is not active. Attempting to activate..."
    # Try common paths
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/home/sdk/miniconda3/etc/profile.d/conda.sh" ]; then
        source "/home/sdk/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    conda activate psfa || echo "Failed to activate psfa, assuming environment is handled externally."
fi

# Set Project Root if not set (Assuming script is run from project root or scripts dir)
# Adjust PROJECT_ROOT logic if needed, similar to other scripts if they have it.
# generate_animnet.sh had explicit PROJECT_ROOT. Let's see if we need it.
# The functions.sh relies on relative paths or python module execution.
# It seems running from project root is expected as 'python3 -m src ...'

# Call the training function
TrainAnimNetDecmp --speaker="$SPEAKER_ID" --data_src="$DATA_SRC"
