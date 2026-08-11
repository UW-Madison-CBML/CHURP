Markdown
# CHURP Usage Guide

## 1. Setup and Installation

First, extract the dataset and clone the repository:

```bash
tar -zxf 3470165.tar.gz
git clone git@github.com:/UW-Madison-CBML/tda-birdsong
cd birdsong_model 
```

## 2. Environment Configuration
### Create and activate the required Conda environment:

```bash
conda env create -f environment.yml -n churp
conda activate churp
```

## 3. Model Training
### Start the training pipeline to generate the churp.pth model weights:

```bash
echo "TRAINING STARTED"
python train.py --audio_path "../3470165/all_wavs" --save_name "churp.pth" --hop_length 86
echo "TRAINING DONE"
```

## 4. Inference
### Define the target subjects and run batched inference to generate pickle (.pkl) files for each bird:

```bash
birdnames=(
    "Bird1" "Bird2" "Bird3" "Bird4" "Bird5" 
    "Bird6" "Bird7" "Bird8" "Bird9" "Bird10"
)

echo "INFERENCE STARTED"
for bird in "${birdnames[@]}"; do
    pklname="./${bird%/}.pkl"
    python model_inference_batched.py \
        --audio_path "../3470165/$bird/Wave" \
        --model_path "churp.pth" \
        --save_pickle "$pklname" \
        --hop_length 86
done
echo "INFERENCE DONE"
```

## 5. Clustering and Segmentation
### Process the inferred pickle files to group and segment the audio data:

```bash
echo "CLUSTERING STARTED"
for bird in "${birdnames[@]}"; do
    pklname="birdsong_inference_out/${bird}.pkl"
    python cluster_and_seg.py \
        --audio_path "../3470165/${bird}/Wave" \
        --pkl "${pklname}" \
        --hop_length 86 \
        --bird_name_prefix "${bird}"
done
echo "CLUSTERING DONE"
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
