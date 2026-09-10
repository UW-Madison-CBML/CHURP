#!/bin/bash
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

