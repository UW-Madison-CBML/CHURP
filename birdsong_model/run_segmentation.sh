#!/bin/bash

PROCESS_ID=$1

set -e

# 2. Define the bird array
birdnames=(
    "Bird0"
    "Bird1"
    "Bird2"
    "Bird3"
    "Bird4"
    "Bird5"
    "Bird6"
    "Bird7"
    "Bird8"
    "Bird9"
    "Bird10"
)

# 3. Pick exactly ONE bird based on the process ID
bird="${birdnames[$PROCESS_ID]}"
echo "Starting parallel job for bird: ${bird}"

echo "Starting package installation..."

# at one point I had to install specific torchaudio wheel to work with chtc 
# -- this might not be necessary anymore for the current code
pip install --upgrade pip setuptools wheel cython
pip install numpy
pip install hdbscan iisignature --no-build-isolation
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install scipy librosa pandas soundfile scikit-learn matplotlib seaborn ipython umap-learn wandb

echo "unzip"
tar -zxf birdsong_inference_out.tar.gz
tar -zxf 3470165.tar.gz

pklname="birdsong_inference_out/${bird}.pkl"

python cluster_and_seg.py --audio_path "3470165/${bird}/Wave" --pkl "${pklname}" --hop_length 86 --bird_name_prefix "${bird}"

echo "${bird} processed."

# store outputs in a results directory and zip it
mkdir birdsong_segmentation_out
mv *.html birdsong_segmentation_out
mv *.png birdsong_segmentation_out
mv *.pkl birdsong_segmentation_out
tar -czf "birdsong_segmentation_out_${PROCESS_ID}.tar.gz" ./birdsong_segmentation_out

echo "DONE"

