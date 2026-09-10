#!/bin/bash
tar -xzf 3470165.tar.gz

mkdir files
mv 3470165/ files

export PYTHONPATH=$PWD:$PYTHONPATH

git clone https://github.com/georgevenven/tweety_bert.git
cd tweety_bert

export PYTHONPATH=$PWD:$PYTHONPATH

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

tar -zcvf tweety_bert.tar.gz tweety_bert
