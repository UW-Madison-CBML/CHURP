CHURP Usage guide

tar -zxf 3470165.tar.gz

git clone https://github.com/C-P-R-DON/CHURP.git
cd birdsong_model 

conda env create -f environment.yml -n churp

conda activate churp

echo "TRAINING STARTED"

python train.py --audio_path "../3470165/all_wavs" --save_name "churp.pth" --hop_length 86

echo "TRAINING DONE"

birdnames=(
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

    pklname="./${bird%/}.pkl"

    python model_inference_batched.py --audio_path "../3470165/$bird/Wave" --model_path "churp.pth" --save_pickle "$pklname" --hop_length 86

done

echo "INFERENCE DONE"

echo "CLUSTERING STARTED"

for bird in "${birdnames[@]}"; do

    pklname="birdsong_inference_out/${bird}.pkl"

    python cluster_and_seg.py --audio_path "../3470165/${bird}/Wave" --pkl "${pklname}" --hop_length 86 --bird_name_prefix "${bird}"

done

echo "CLUSTERING DONE"

mkdir CHURP_outputs
mv *.html CHURP_outputs
mv *.png CHURP_outputs
mv *.pkl CHURP_outputs
tar -czf "CHURP_outputs.tar.gz" ./CHURP_outputs
