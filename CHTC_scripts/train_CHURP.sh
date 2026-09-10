#!/bin/bash

conda env create -f environment.yml

conda activate churp

wget -P files/ https://figshare.com/ndownloader/articles/3470165/versions/1

tar -zxf files/3470165.tar.gz -C files/

bash scripts/move_wavs.sh

# training
cd CHURP/birdsong_model
python src/train.py --audio_path "files/3470165/all_wavs" --save_name "models/churp.pth" --hop_length 86 --epochs 50

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

echo "INFERENCE STARTED"

# run inference on each bird
for bird in "${birdnames[@]}"; do

    pklname="./files/${bird%/}.pkl"

    python src/model_inference_batched.py --audio_path "files/3470165/$bird/Wave" --model_path "models/churp.pth" --save_pickle "$pklname" --hop_length 86

done


# clustering embedding songs
for bird in "${birdnames[@]}"; do

    pklname="./files/${bird}.pkl"

    python src/cluster_and_seg.py --audio_path "files/3470165/${bird}/Wave" --out_dir "files" --pkl "${pklname}" --hop_length 86 --bird_name_prefix "${bird}" --min_length 20 --max_length 50 --max_clusters 20 --fst_threshold 0.2 --sec_threshold 0.75 &

done

wait