#!/bin/bash

pip install torch==2.10.0
pip install librosa

echo "unzip"
tar -zxf 3470165.tar.gz 

echo "Running"

savename="birdsong.pth"

wavfiles="./3470165/all_wavs/"

echo $savename
echo ""
echo $wavfiles

# train the model
python train.py --audio_path "$wavfiles" --save_name "$savename" --hop_length 86

# move results to a directory and zip them
mkdir birdsong_train
mv $savename birdsong_train
tar -zcvf birdsong_train.tar.gz birdsong_train 

echo "DONE"
