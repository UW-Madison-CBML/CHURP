#!/bin/bash
tar -xzf 3470165.tar.gz

# unzip tweety_bert directory after pretraining step
tar -xzf tweety_bert.tar.gz

cd tweety_bert

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 1. Create and activate a new Conda environment
conda create -y -n tweetybert python=3.11
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tweetybert

# 2. Install core scientific packages (including librosa)
conda install -y -c conda-forge numpy matplotlib tqdm umap-learn hdbscan scikit-learn pandas librosa seaborn jupyter ipykernel

# 3. Install additional dependencies via pip
pip install soundfile shutil-extra glasbey pyqtgraph PyQt5 hmmlearn

# 4. install torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


# change path error in inference.py
sed -i "s|'python', '/home/george-vengrovski/Documents/projects/tweety_net_song_detector/src/inference.py'|'python','./src/inference.py'|g" src/inference.py

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


# fix hardcoded hop-length
myhop=86

perl -pi -e 's/("--output", str\(temp_detector_output_dir\))/$1,\n        "--step_size", "86"/g' detect_song.py

perl -pi.bak -e 's/\b(step_size\s*=\s*)119\b/${1}'"$myhop"'/g' src/spectogram_generator.py

sed -i "s/'--step_size', type=int, default=119/'--step_size', type=int, default=86/g" src/spectogram_generator.py

perl -pi.bak -e 's/\bhop_length\s*=\s*119\b/hop_length = '"$myhop"'/g' src/inference.py

# make empty npz file for decoding.py to work
mkdir files

for bird in "${birdnames[@]}"; do
	touch files/$bird.npz

	# detect songs for individual bird so that spectrogram naming is not messed up
	python detect_song.py --input_dir "../3470165/$bird/Wave/"


        # generate UMAP embeddings and train decoder on all birds with default arguments
	python decoding.py --mode single --bird_name "${bird}_decoder" --model_name "MyTweetyBERTModel" --wav_folder "../3470165/$bird/Wave/" --num_random_files_spec 100 --num_samples_umap 5e5 --song_detection_json_path "files/${bird}_song_detection.json"

	mv files/song_detection.json files/${bird}_song_detection.json

	# run inference on specific bird
	python run_inference.py --bird_name "${bird}_decoder" --wav_folder "../3470165/$bird/Wave/" --apply_post_processing True --visualize --song_detection_json "files/${bird}_song_detection.json"

done

cd ..

tar -zcvf tweety_bert2.tar.gz tweety_bert
