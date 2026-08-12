# CHURP Usage Guide

## 1. Setup and Installation

First, extract the dataset and clone the repository:

```bash
tar -zxf 3470165.tar.gz
git clone https://github.com/UW-Madison-CBML/CHURP
cd CHURP/birdsong_model
```

## 2. Container Configuration
### Pull and run the docker image:

```bash
docker pull cdonahue6/churp
docker run cdonahue6/churp
```

## 3. Model Training
### Start the training pipeline to generate the churp.pth model weights:

```bash
python train.py --audio_path "../../3470165/all_wavs" --save_name "churp.pth" --hop_length 86 --epochs 50
```

## 4. Inference
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

## 5. Clustering and Segmentation
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
TweetyBERT was implemented according to the github https://github.com/georgevenven/tweety_bert/tree/main. Some minor changes were made to the code in order to correct path errors and change the hop length of the spectrogram generation steps. All code used for training and inference with the TweetyBERT model is included in the scripts in the "tweety_bert" subdirectory. These scripts include all parameters used, as well as the commands that were executed to change some of the original code.

# Metrics and comparison calculations





