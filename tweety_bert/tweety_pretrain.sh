#!/bin/bash
tar -xzf 3470165.tar.gz

mkdir tweety_bert_results

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 1. Create and activate a new Conda environment
conda create -n tweetybert python=3.11
source $(conda info --base)/etc/profile.d/conda.sh 
conda activate tweetybert

# 2. Install core scientific packages (including librosa)
conda install -c conda-forge numpy matplotlib tqdm umap-learn hdbscan scikit-learn pandas librosa seaborn jupyter ipykernel

# 3. Install additional dependencies via pip
pip install soundfile shutil-extra glasbey pyqtgraph PyQt5 hmmlearn

# 4. install torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Clone this repository
git clone https://github.com/georgevenven/tweety_bert.git
cd tweety_bert

# pretrain network on all birds using default settings, except step_size 86 for 2.7ms bins
python pretrain.py --input_dir "../3470165/all_wavs" --experiment_name "MyTweetyBERTModel" --test_percentage 20 --batch_size 32 --learning_rate 3e-4 --context 1000 --m 250 --multi_thread --step_size 86 

cd ..

tar -zcvf tweety_bert.tar.gz tweety_bert

