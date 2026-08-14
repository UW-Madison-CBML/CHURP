# CHURP Usage Guide

## 1. Setup and Installation

First, extract the dataset. The dataset used below comes from https://figshare.com/articles/media/BirdsongRecognition/3470165 and the "all_wavs" directory was created by adding all the wav files from the individual birds to a single directory (see the move_wavs.sh script). Note than any wav files can be used as model inputs. Next, clone the repository:

```bash
tar -zxf 3470165.tar.gz
git clone https://github.com/UW-Madison-CBML/CHURP
cd CHURP/birdsong_model
```

## 2. Environemnt Configuration
### Create an environment for running CHURP and install the necessary programs. Note that the pipeline uses torch 2.10.0, so be sure to change that and match the correct CUDA version to your system and GPU drivers if needed:

```bash
conda create -n churp python=3.10
source $(conda info --base)/etc/profile.d/conda.sh 
conda activate churp

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda install -y -c conda-forge numpy matplotlib tqdm umap-learn hdbscan scikit-learn pandas librosa seaborn jupyter ipykernel
pip install --upgrade pip setuptools wheel cython
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu121
echo "Installing hdbscan and iisignature..."
pip install hdbscan iisignature --no-build-isolation
pip install scipy soundfile ipython wandb shutil-extra glasbey pyqtgraph PyQt5 hmmlearn
```

## 3. Model Training -- outputs "churp.pth" weights file
### Start the training pipeline to generate the churp.pth model weights:

```bash
python train.py --audio_path "../../3470165/all_wavs" --save_name "churp.pth" --hop_length 86 --epochs 50
```

## 4. Inference -- outputs "<bird name>.pkl" files containing dictionary mapping each .wav file (the keys) to the corresponding list of 3d normalized embeddings for each time bin in the recording (the values)
### Define the target subjects and run batched inference to generate pickle (.pkl) files for each bird:

```bash
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

for bird in "${birdnames[@]}"; do
    pklname="./${bird%/}.pkl"
    python model_inference_batched.py --audio_path "../../3470165/$bird/Wave" --model_path "churp.pth" --save_pickle "$pklname" --hop_length 86
done
```

## 5. Clustering and Segmentation -- outputs <bird name>.html summary file and .png files containing visualizations
### Process the inferred pickle files to group and segment the audio data (in parallel, for speed):

```bash
for bird in "${birdnames[@]}"; do
    pklname="${bird}.pkl"
    python cluster_and_seg.py --audio_path "../../3470165/${bird}/Wave" --pkl "${pklname}" --hop_length 86 --bird_name_prefix "${bird}" --min_length 20 --max_length 50 --max_clusters 20 --fst_threshold 0.2 --sec_threshold 0.75 &
done
# wait for background processes running in parallel to end
wait
```

## 6. Packaging Outputs
### Finally, organize all generated visualizations and data files into a single compressed archive for easy sharing or storage:

```bash
mkdir CHURP_outputs
mv *.html CHURP_outputs
mv *.png CHURP_outputs
mv *.pkl CHURP_outputs
tar -czf "CHURP_outputs.tar.gz" ./CHURP_outputs
```
# TweetyBERT implementation
TweetyBERT was implemented according to the github https://github.com/georgevenven/tweety_bert/tree/main. Some minor changes were made to the code in order to correct path errors and change the hop length of the spectrogram generation steps. All code used for training and inference with the TweetyBERT model is included in the code block below. This includes all parameters used, as well as the commands that were executed to change some of the original code. 

```bash
# 1. Create and activate a new Conda environment
conda create -n tweetybert python=3.11
conda activate tweetybert

# 2. Install core scientific packages (including librosa)
conda install -c conda-forge \
    numpy \
    matplotlib \
    tqdm \
    umap-learn \
    hdbscan \
    scikit-learn \
    pandas \
    seaborn \
    jupyter \
    ipykernel \
    librosa

# 3. Install additional dependencies via pip
pip install soundfile shutil-extra glasbey pyqtgraph PyQt5 hmmlearn

# clone tweetyBERT repository
git clone https://github.com/georgevenven/tweety_bert.git
cd tweety_bert

# pretrain network on all birds using default settings, except step_size 86 for 2.7ms bins
python pretrain.py --input_dir "../3470165/all_wavs" --experiment_name "MyTweetyBERTModel" --test_percentage 20 --batch_size 32 --learning_rate 3e-4 --context 1000 --m 250 --multi_thread --step_size 86

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

perl -pi.bak -e 's/\bhop_length\s*=\s*119\b/hop_length = '"$myhop"'/g' src/inference.py

# make empty npz file for decoding.py to work
mkdir files

for bird in "${birdnames[@]}"; do
        touch files/$bird.npz

        # detect songs for individual bird
        python detect_song.py --input_dir "../3470165/$bird/Wave/"

        # generate UMAP embeddings and train decoder on all birds with default arguments
        python decoding.py --mode single --bird_name "${bird}_decoder" --model_name "MyTweetyBERTModel" --wav_folder "../3470165/$bird/Wave/" --num_random_files_spec 100 --num_samples_umap 5e5 --song_detection_json_path "files/${bird}_song_detection.json"

        mv files/song_detection.json files/${bird}_song_detection.json

        # run inference on specific bird
        python run_inference.py --bird_name "${bird}_decoder" --wav_folder "../3470165/$bird/Wave/" --apply_post_processing True --visualize --song_detection_json "files/${bird}_song_detection.json"

done
```

# Metrics and comparison calculations





