#!/bin/bash

echo "Starting package installation..."

# at one point I had to install specific torchaudio wheel to work with chtc 
# -- this might not be necessary anymore for the current code
pip install --upgrade pip setuptools wheel cython
pip install numpy
pip install hdbscan iisignature --no-build-isolation
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install scipy librosa pandas soundfile scikit-learn matplotlib seaborn ipython umap-learn wandb

echo "unzip"
tar -zxf birdsong_train.tar.gz
tar -zxf 3470165.tar.gz

modelname="./birdsong_train/birdsong.pth"

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

# run inference on each bird
for bird in "${birdnames[@]}"; do

       pklname="./${bird%/}.pkl"

       python model_inference_batched.py --audio_path "3470165/$bird/Wave" --model_path "$modelname" --save_pickle "$pklname" --hop_length 86

        echo "$bird processed."

done

# store outputs in a results directory and zip it
mkdir birdsong_inference_out
mv *.pkl birdsong_inference_out
tar -czf birdsong_inference_out.tar.gz ./birdsong_inference_out

echo "DONE"

