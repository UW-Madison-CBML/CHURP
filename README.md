# CHURP Overview

Brief description of method, good figure/animation, and link to preprint go here

# CHURP Usage Guide
The following usage guide works through the process for training and inference with the CHURP model using the dataset from the manuscript, which can be downloaded at https://figshare.com/articles/media/BirdsongRecognition/3470165

## 1. Setup and Installation

First, activate the docker image for CHURP. Next, clone the CHURP repository and extract the data set that was downloaded from the link included above. For training, all the recordings must be in the same subdirectory -- the "move_wavs.sh" script handles this.

```bash
# run docker
docker run -it cdonahue6/churp:latest

# clone the repo
git clone https://github.com/UW-Madison-CBML/CHURP

# unzip the data, deposited in to the files directory
tar -zxf 3470165.tar.gz -C files/

# copy wav files from all birds into single directory
bash scripts/move_wavs.sh
```

## 2. Model Training -- outputs "churp.pth" weights into "models" directory
```bash
python src/train.py --audio_path "files/3470165/all_wavs" --save_name "models/churp.pth" --hop_length 86 --epochs 50
```

## 3. Inference -- outputs "<bird name>.pkl" files (into the "files" directory), containing dictionary mapping each .wav file (the keys) to the corresponding list of 3d normalized embeddings for each time bin in the recording (the values)
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

echo "INFERENCE STARTED"

# run inference on each bird
for bird in "${birdnames[@]}"; do

    pklname="./files/${bird%/}.pkl"

    python src/model_inference_batched.py --audio_path "files/3470165/$bird/Wave" --model_path "models/churp.pth" --save_pickle "$pklname" --hop_length 86

done

wait
```

## 4. Clustering and Segmentation -- outputs <bird name>.html summary into "files" folder, as well as .png files containing visualizations
### Note that the code is run in parallel for all birds here due to the '&' after the python command.
```bash
# clustering embedding songs
for bird in "${birdnames[@]}"; do

    pklname="./files/${bird}.pkl"

    python src/cluster_and_seg.py --audio_path "files/3470165/${bird}/Wave" --out_dir "files" --pkl "${pklname}" --hop_length 86 --bird_name_prefix "${bird}" --min_length 20 --max_length 50 --max_clusters 20 --fst_threshold 0.2 --sec_threshold 0.75 &

done

wait
```
# TweetyBERT implementation
TweetyBERT was implemented according to the github https://github.com/georgevenven/tweety_bert/tree/main as of July 2026. Some minor changes were made to the code in order to correct path errors and change the hop length of the spectrogram generation steps. All code used for training and inference with the TweetyBERT model is included in the code block below. This includes all parameters used, as well as the commands that were executed to change some of the original code.
```bash
# run docker
docker run -it cdonahue6/churp:latest

tar -xzf 3470165.tar.gz

mkdir files
mv 3470165/ files

git clone https://github.com/georgevenven/tweety_bert.git
cd tweety_bert

# pretrain network on all birds using default settings, except step_size 86 for 2.7ms bins
python pretrain.py --input_dir "../files/3470165/all_wavs" --experiment_name "MyTweetyBERTModel" --test_percentage 20 --batch_size 32 --learning_rate 3e-4 --context 1000 --m 250 --multi_thread --step_size 86 

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

        # generate UMAP embeddings and train decoder on all birds with default arguments
        python decoding.py --mode single --bird_name "${bird}_decoder" --model_name "MyTweetyBERTModel" --wav_folder "../files/3470165/$bird/Wave/" --num_random_files_spec 100 --num_samples_umap 5e5

	    # detect songs for individual bird so that spectrogram naming is not messed up
	    python detect_song.py --input_dir "../files/3470165/$bird/Wave/"

        mv files/song_detection.json files/${bird}_song_detection.json

        # run inference on specific bird
        python run_inference.py --bird_name "${bird}_decoder" --wav_folder "../files/3470165/$bird/Wave/" --apply_post_processing True --visualize --song_detection_json "files/${bird}_song_detection.json"

done

cd ..
```

# Metrics and comparison calculations
### This code assumes that both the above CHURP and TweetyBERT code was executed in the same directory and that the CHURP docker image is running
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

mkdir files/comparison_outputs

# run comparisons for each bird
for bird in "${birdnames[@]}"; do

        python src/comparing.py \
                --xml "files/3470165/${bird}/Annotation.xml" \
                --pkl "files/annotated_bins_${bird}.pkl" \
                --json "tweety_bert/files/${bird}_decoder_decoded_database.json" \
                --wav_dir "files/3470165/${bird}/Wave" \
                --sr 32000 \
                --hop 86 \
                --out_dir "files/comparison_outputs" \
                --regions "tweety_bert/files/${bird}_song_detection.json" \
                --bird "${bird}" > "files/comparison_outputs/${bird}.out" &

done

wait

touch metrics.csv

sed -n '1p' "files/comparison_outputs/Bird0_comparison_data.csv" > files/comparison_outputs/metrics.csv

for bird in "${birdnames[@]}"; do
        sed -n '2p' "files/comparison_outputs/${bird}_comparison_data.csv" >> files/comparison_outputs/metrics.csv
done

# plot results
python src/plot_comparisons.py "files/comparison_outputs"

mkdir final_outputs
mv files/comparison_outputs/* final_outputs/
```



