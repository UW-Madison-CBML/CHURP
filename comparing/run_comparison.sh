#!/bin/bash
#capture the unique parallel process ID (0 through 9)
PROCESS_ID=$1

# 2. Define the bird array
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

# 3. Pick exactly ONE bird based on the process ID
bird="${birdnames[$PROCESS_ID]}"
echo "Starting parallel job for bird: $bird"

# 4. Install dependencies (standard python packages)
pip install numpy librosa soundfile matplotlib scipy

# 5. Extract input packages 
tar -xzf "birdsong_segmentation_out_${PROCESS_ID}.tar.gz"
tar -xzf tweety_bert2.tar.gz
tar -xzf 3470165.tar.gz

# 6. Create directories localized to this specific job
mkdir -p comparison/mapping
mkdir -p "${bird}_annotations"

# 7. Execute scripts for this SINGLE bird
python comparing.py \
    --xml "3470165/${bird}/Annotation.xml" \
    --pkl "birdsong_segmentation_out/annotated_bins_${bird}.pkl" \
    --json "tweety_bert/files/${bird}_decoder_decoded_database.json" \
    --wav_dir "3470165/${bird}/Wave" \
    --sr 32000 \
    --hop 86 \
    --out_dir "comparison/mapping" \
    --regions "tweety_bert/files/${bird}_song_detection.json" > "comparison/${bird}.out"

python plots.py --out_dir "${bird}_annotations" --wav_dir "3470165/${bird}/Wave" --hop 86 --dicts "comparison/mapping/expanded_json_annotations.pkl"
python plots.py --out_dir "${bird}_annotations" --wav_dir "3470165/${bird}/Wave" --hop 86 --dicts "comparison/mapping/expanded_pkl_annotations.pkl"
python plots.py --out_dir "${bird}_annotations" --wav_dir "3470165/${bird}/Wave" --hop 86 --dicts "comparison/mapping/processed_xml_annotations.pkl"

# 8. Clean up structure and move files into the main output folder
mv "${bird}_annotations" comparison/
mv *.csv comparison/


# 9. Pack everything up into the unique tarball HTCondor is expecting
tar -czf "comparison_${PROCESS_ID}.tar.gz" comparison/

echo "Job for $bird complete!"
