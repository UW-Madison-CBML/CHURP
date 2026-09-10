#!/bin/bash

# cant download the files from figshare via cli
tar -zxf 3470165.tar.gz -C files/

bash scripts/move_wavs.sh

export PYTHONPATH=$PWD:$PYTHONPATH

# training
python src/train.py --audio_path "files/3470165/all_wavs" --save_name "models/churp.pth" --hop_length 86 --epochs 50

tar -czf models.tar.gz models/

tar -czf files.tar.gz files/
