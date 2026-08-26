import argparse
from collections import Counter
import json
import os
import pickle
import re
import wave
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import csv

def get_run_lengths(annot_list):
    """Collapses continuous identical labels into (label, duration_in_bins) tuples."""
    if not annot_list:
        return []
    runs = []
    current_label = annot_list[0]
    current_len = 1
    for label in annot_list[1:]:
        if label == current_label:
            current_len += 1
        else:
            runs.append((current_label, current_len))
            current_label = label
            current_len = 1
    runs.append((current_label, current_len))
    return runs

def compute_phrase_metrics(annot_dict, bg_labels=("-1", -1)):
    """Computes occurrence frequency, average duration, and transition entropy per phrase."""
    durations = Counter()
    run_counts = Counter()
    transitions = {}
    frequencies = Counter()
    
    # Force background labels to strings for safe comparison
    bg_labels_str = {str(b) for b in bg_labels}

    for filename, annots in annot_dict.items():
        # Force all annotations to strings to match the mapping matrix
        str_annots = [str(x) for x in annots]
        runs = get_run_lengths(str_annots)
        
        # Compute Durations and Frequencies
        for label, length in runs:
            if label not in bg_labels_str:
                durations[label] += length
                run_counts[label] += 1
                frequencies[label] += 1  # Weight based on how often the phrase occurs
                
        # Compute Transitions (ignoring background gaps to find actual syllable syntax)
        phrase_seq = [label for label, length in runs if label not in bg_labels_str]
        for i in range(len(phrase_seq) - 1):
            curr_label = phrase_seq[i]
            next_label = phrase_seq[i+1]
            if curr_label not in transitions:
                transitions[curr_label] = Counter()
            transitions[curr_label][next_label] += 1

    # Calculate mean duration per phrase
    mean_durations = {k: durations[k] / run_counts[k] for k in durations}
    
    # Calculate Shannon Entropy per phrase and transition probability per phase transition
    entropies = {}
    transition_probs = {}
    for label in frequencies.keys():
        syb_transitions = {}
        trans_counts = transitions.get(label, {})
        total_trans = sum(trans_counts.values())
        ent = 0.0
        if total_trans > 0:
            for nxt, count in trans_counts.items():
                p = count / total_trans
                syb_transitions[nxt] = p
                ent -= p * np.log2(p)
        entropies[label] = ent
        transition_probs[label] = syb_transitions
        
    return frequencies, mean_durations, entropies, transition_probs

def weighted_pearson(x, y, w):
    """Calculates the weighted Pearson correlation coefficient."""
    x, y, w = np.array(x), np.array(y), np.array(w)
    if np.sum(w) == 0:
        return 0.0
        
    w_norm = w / np.sum(w)
    mean_x = np.sum(w_norm * x)
    mean_y = np.sum(w_norm * y)
    
    cov = np.sum(w_norm * (x - mean_x) * (y - mean_y))
    var_x = np.sum(w_norm * (x - mean_x)**2)
    var_y = np.sum(w_norm * (y - mean_y)**2)
    
    if var_x == 0 or var_y == 0:
        return 0.0
        
    return cov / np.sqrt(var_x * var_y)

def evaluate_mapped_metrics(gt_dict, pred_dict, mapping, bg_labels=("-1", -1)):
    """Extracts metrics and calculates weighted Pearson correlations for mapped labels."""
    gt_freq, gt_dur, gt_ent, _ = compute_phrase_metrics(gt_dict, bg_labels)
    pr_freq, pr_dur, pr_ent, _ = compute_phrase_metrics(pred_dict, bg_labels)
    
    bg_labels_str = {str(b) for b in bg_labels}
    
    x_dur, y_dur, w_dur = [], [], []
    x_ent, y_ent, w_ent = [], [], []
    
    for gt_label, pred_label in mapping.items():
        # Keys are already strings from the Hungarian mapping
        if gt_label in bg_labels_str or pred_label in bg_labels_str:
            continue
            
        weight = gt_freq.get(gt_label, 0)
        if weight == 0:
            continue
            
        w_dur.append(weight)
        x_dur.append(gt_dur.get(gt_label, 0.0))
        y_dur.append(pr_dur.get(pred_label, 0.0)) # Safe now because pr_dur has string keys
        
        w_ent.append(weight)
        x_ent.append(gt_ent.get(gt_label, 0.0))
        y_ent.append(pr_ent.get(pred_label, 0.0)) # Safe now because pr_ent has string keys
        
    corr_dur = weighted_pearson(x_dur, y_dur, w_dur)
    corr_ent = weighted_pearson(x_ent, y_ent, w_ent)
    
    print("\n--- Debugging Duration Mappings ---")
    for gt_label, pred_label in mapping.items():
        if gt_label in bg_labels_str or pred_label in bg_labels_str:
            continue
            
        weight = gt_freq.get(gt_label, 0)
        if weight == 0:
            continue
            
        gt_length = gt_dur.get(gt_label, 0.0)
        pr_length = pr_dur.get(pred_label, 0.0)
        
        print(f"Weight: {weight:<4} | GT Label '{gt_label}' (Len: {gt_length:.1f}) mapped to Pred Label '{pred_label}' (Len: {pr_length:.1f})")

    print("\n--- Debugging Entropy Mappings ---")
    for gt_label, pred_label in mapping.items():
        if gt_label in bg_labels_str or pred_label in bg_labels_str:
            continue
            
        weight = gt_freq.get(gt_label, 0)
        if weight == 0:
            continue
            
        gt_entropy = gt_ent.get(gt_label, 0.0)
        pr_entropy = pr_ent.get(pred_label, 0.0)
        
        print(f"Weight: {weight:<4} | GT Label '{gt_label}' (Entropy: {gt_entropy:.1f}) mapped to Pred Label '{pred_label}' (Entropy: {pr_entropy:.1f})")

    return corr_dur, corr_ent

def get_wav_length_samples(wav_path):
    """Opens a wav file and returns its total length in samples."""
    with wave.open(wav_path, "rb") as wf:
        return wf.getnframes()


def normalize_wav_name(filename):
    """Strips whitespace and ensures .wav extension exists."""
    filename = str(filename).strip()
    if not filename.endswith(".wav"):
        filename += ".wav"
    return filename

def pad_missing_wav_files(parsed_dict, wav_dir, default_val=-1):
    """
    Checks a parsed dictionary against the actual .wav files in the directory.
    If any .wav files are missing from the dictionary, it creates an entry
    filled with the default background label matching the audio's sample length.
    """
    print(f"\n[Padding] Scanning '{wav_dir}' for missing files...")
    
    all_wav_files = [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")]
    missing_files = []
    
    for wav_file in all_wav_files:
        wave_filename = normalize_wav_name(wav_file)
        
        if wave_filename not in parsed_dict:
            missing_files.append(wave_filename)
            full_wav_path = os.path.join(wav_dir, wave_filename)
            wav_length = get_wav_length_samples(full_wav_path)
            
            parsed_dict[wave_filename] = [default_val] * wav_length
            
    if missing_files:
        print(f"[Padding] Warning: Added {len(missing_files)} missing files with default '{default_val}' annotations.")
        print(f"[Padding] Missing files patched: {sorted(missing_files)}")
    else:
        print("[Padding] Success: All WAV files in the directory were already present.")
        
    return parsed_dict

def xml_to_position_dict(xml_path, wav_dir):
    """
    Parses XML annotation files and maps syllable labels to a sample-level 
    list matching the exact length of the corresponding WAV audio files.
    """
    print(f"\n[XML] Loading {xml_path}...")

    # Load XML content as a string
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_string = f.read()

    # Handle potentially malformed XML files missing the closing root tag
    if not xml_string.strip().endswith("</Sequences>"):
        xml_string += "\n</Sequences>"

    root = ET.fromstring(xml_string)
    xml_dict = {}

    # Extract uniquely referenced WAV filenames from the XML sequences
    raw_wave_files = set(
        seq.find("WaveFileName").text
        for seq in root.findall("Sequence")
        if seq.find("WaveFileName") is not None
    )

    wave_files = {normalize_wav_name(w) for w in raw_wave_files}
    print(f"[XML] Detected WAV files: {sorted(list(wave_files))}")

    # Pre-allocate annotation arrays filled with "-1" (background/unlabeled) sized to the exact sample length of each target WAV file
    for wave_file in wave_files:
        wav_path = os.path.join(wav_dir, wave_file)
        if os.path.exists(wav_path):
            wav_length = get_wav_length_samples(wav_path)
            xml_dict[wave_file] = ["-1"] * wav_length
        else:
            raise FileNotFoundError(
                f"[XML Error] Required audio file '{wave_file}' not found in '{wav_dir}'"
            )

    # Map XML Note positions to the pre-allocated sample arrays
    for sequence in root.findall("Sequence"):
        raw_name = sequence.find("WaveFileName").text
        wave_file = normalize_wav_name(raw_name)
        
        # Sequence position acts as the global offset for notes within it
        seq_pos = int(sequence.find("Position").text)

        for note in sequence.findall("Note"):
            note_pos = int(note.find("Position").text)
            note_len = int(note.find("Length").text)
            label = note.find("Label").text.strip()

            # Calculate absolute sample indices for the note
            start_idx = seq_pos + note_pos
            end_idx = start_idx + note_len

            # Apply the label to the calculated sample range
            if note_len > 0:
                xml_dict[wave_file][start_idx:end_idx] = [label] * note_len

    return xml_dict


def process_pkl_dict(pkl_path, hop, wav_dir):
    """
    Loads Pickle annotation data and expands 
    time-binned annotations into sample-level arrays based on a hop length.
    """
    print(f"\n[PKL] Loading {pkl_path}...")

    with open(pkl_path, "rb") as f:
        raw_pkl = pickle.load(f)

    pkl_dict = {}

    # Extract entries mapping wav filenames to their binned annotations
    if isinstance(raw_pkl, dict) and "wav_file" in raw_pkl:
        ann_key = (
            "annotated_times" if "annotated_times" in raw_pkl else "annotations"
        )

        if ann_key in raw_pkl:
            print(
                f"[PKL] Correctly identified columnar structure ('wav_file' & '{ann_key}')."
            )
            wav_names = raw_pkl["wav_file"]
            annotations = raw_pkl[ann_key]

            # Normalize filenames and ensure annotations are standard Python lists
            for name, ann in zip(wav_names, annotations):
                if hasattr(ann, "tolist"):
                    ann = ann.tolist()
                pkl_dict[normalize_wav_name(name)] = ann
        else:
            raise ValueError(
                f"[PKL Error] Found 'wav_file' key but could not find annotation array in {list(raw_pkl.keys())}"
            )

    print(
        f"[PKL] Successfully loaded WAV records: {sorted(list(pkl_dict.keys()))}"
    )

    expanded_dict = {}
    hop_multiplier = int(hop)

    # Upsample binned annotations to match actual WAV sample rates
    for wave_file, time_bins in pkl_dict.items():
        wave_filename = os.path.basename(wave_file)
        full_wav_path = os.path.join(wav_dir, wave_filename)

        if not os.path.exists(full_wav_path):
            raise FileNotFoundError(
                f"[PKL Error] Required audio file '{wave_filename}' not found in '{wav_dir}'"
            )

        wav_length = get_wav_length_samples(full_wav_path)
        
        # Determine placeholder type based on the first annotation (string or int)
        placeholder = (
            "-1" if (time_bins and isinstance(time_bins[0], str)) else -1
        )
        expanded_list = [placeholder] * wav_length

        # Expand each bin by 'hop' samples to reconstruct the full audio timeline
        for bin_idx, val in enumerate(time_bins):
            start_idx = bin_idx * hop_multiplier
            end_idx = start_idx + hop_multiplier

            expanded_list[start_idx:end_idx] = [val] * (end_idx - start_idx)

        expanded_dict[wave_filename] = expanded_list

    # Ensure any WAV files in the directory without PKL annotations get blank arrays
    expanded_dict = pad_missing_wav_files(expanded_dict, wav_dir, default_val=-1)

    return expanded_dict


def parse_cluster_json(
    data_input, samplerate: float, hop_length: float, wav_dir: str, regions_json_path: str = None
):
    """
    Parses cluster/syllable JSON outputs, applies optional regional time offsets, 
    and aligns the cluster IDs to a sample-level array matching the WAV files.
    """
    print(f"\n[JSON] Loading {data_input}...")

    # Flexible loading: handle direct filepath or raw JSON string/dict
    if isinstance(data_input, str):
        if os.path.isfile(data_input):
            with open(data_input, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(data_input)
    else:
        data = data_input

    # Load region map because the JSON targets isolated segments rather than full audio (tweetyBERT uses song_detection.py)
    region_offsets = {}
    if regions_json_path and os.path.exists(regions_json_path):
        with open(regions_json_path, 'r') as f:
            regions_data = json.load(f)
        for item in regions_data:
            fname = normalize_wav_name(item["filename"])
            region_offsets[fname] = item.get("segments", [])

    expanded_dict = {}

    for result in data.get("results", []):
        raw_file_name = result.get("file_name", "")
        if not raw_file_name:
            continue

        # Extract base file identifier and segment index (e.g. '012_segment_2' -> 012, 2)
        match = re.match(r"^(\d+)(?:_segment_(\d+))?", raw_file_name)
        if not match:
            continue
        
        file_num = match.group(1)
        segment_idx = int(match.group(2)) if match.group(2) is not None else 0
        
        wave_filename = normalize_wav_name(file_num)
        full_wav_path = os.path.join(wav_dir, wave_filename)

        if not os.path.exists(full_wav_path):
            raise FileNotFoundError(
                f"[JSON Error] Required audio file '{wave_filename}' not found in '{wav_dir}'"
            )

        # Look up global offset if this segment needs shifting back into the full recording timeline
        segment_offset_bin = 0.0
        if wave_filename in region_offsets:
            segs = region_offsets[wave_filename]
            if segment_idx < len(segs):
                segment_offset_bin = segs[segment_idx].get("onset_timebin", 0.0)

        # Allocate the full-length array for this WAV file exactly once
        if wave_filename not in expanded_dict:
            wav_length = get_wav_length_samples(full_wav_path)
            expanded_dict[wave_filename] = [-1] * wav_length

        expanded_list = expanded_dict[wave_filename]
        onsets_offsets = result.get("syllable_onsets_offsets_timebins", {})

        # Map cluster IDs onto the sample array using provided time intervals
        for cluster_str, intervals in onsets_offsets.items():
            cluster_id = int(cluster_str)
            
            for start_bin, end_bin in intervals:
                # Apply global region offset (if any) to align segment back to master WAV
                start_bin = start_bin + segment_offset_bin
                end_bin = end_bin + segment_offset_bin

                # Convert time bins to raw sample indices
                start_idx = int(start_bin * hop_length)
                end_idx = int(end_bin * hop_length)

                # Boundary safety checks
                if start_idx >= len(expanded_list):
                    continue
                if end_idx > len(expanded_list):
                    end_idx = len(expanded_list)

                # Fill the array slice with the cluster ID
                if end_idx > start_idx:
                    expanded_list[start_idx:end_idx] = [cluster_id] * (
                        end_idx - start_idx
                    )

    print(
        f"[JSON] Successfully loaded WAV records: {sorted(list(expanded_dict.keys()))}"
    )

    # Ensure unrepresented WAV files in the dir are populated with default arrays
    expanded_dict = pad_missing_wav_files(expanded_dict, wav_dir, default_val=-1)

    return expanded_dict

def apply_region_mask(annot_dict, regions_json_path, hop):
    """
    Filters the annotations dict by ONLY keeping samples that fall inside the
    segments specified in the regions JSON file. Samples outside these regions
    (or in files with song_present=False) are completely removed from evaluation.
    """
    if not regions_json_path or not os.path.exists(regions_json_path):
        return annot_dict

    print(f"\n[Regions] Trimming data to ONLY include specified song regions from {regions_json_path}...")
    with open(regions_json_path, 'r') as f:
        regions_data = json.load(f)

    # Create a lookup table for region parameters per file
    region_lookup = {normalize_wav_name(item["filename"]): item for item in regions_data}
    filtered_dict = {}
    hop_multiplier = int(hop)

    for filename, original_annots in annot_dict.items():
        # If file is not in the JSON regions list, drop its contents
        if filename not in region_lookup:
            filtered_dict[filename] = []
            continue

        item = region_lookup[filename]
        
        # If song is marked as not present, completely drop frames for this file
        if not item.get("song_present", False):
            filtered_dict[filename] = []
            continue

        bg_val = "-1" if (original_annots and isinstance(original_annots[0], str)) else -1
        kept_samples = []

        segments = item.get("segments", [])
        for i, seg in enumerate(segments):
            onset_bin = seg.get("onset_timebin", 0)
            offset_bin = seg.get("offset_timebin", 0)
            
            # filter out sigments less than 250 time bins long 
            if (offset_bin - onset_bin) < 250:
                continue
            
            # Convert timebins directly to sample indices
            onset_sample = max(0, min(onset_bin * hop_multiplier, len(original_annots)))
            offset_sample = max(0, min(offset_bin * hop_multiplier, len(original_annots)))

            if offset_sample > onset_sample:
                # Insert a single background tag between disconnected segments 
                # to prevent artificial phrase transitions across segment boundaries
                if i > 0 and kept_samples:
                    kept_samples.append(bg_val)
                
                # Slice and keep ONLY the valid song region
                kept_samples.extend(original_annots[onset_sample:offset_sample])

        filtered_dict[filename] = kept_samples

    return filtered_dict

def compute_hungarian_mapping(xml_dict, target_dict, bg_label="-1"):
    """
    Finds the optimal 1-to-1 mapping between ground truth (XML) labels and 
    predicted/clustered (Target) labels.
    Forces the background label to map to itself, and uses the Hungarian 
    algorithm for the remaining syllables.
    """
    print("\n[Hungarian] Computing co-occurrence matrix and 1-1 mapping...")

    # Build a raw co-occurrence frequency matrix between GT and Target labels
    pair_counts = Counter()
    
    # Iterate through all audio files that exist in both datasets
    for wave_file in xml_dict.keys():
        if wave_file in target_dict:
            # Convert all labels to strings for consistent matching
            xml_str_list = (str(x) for x in xml_dict[wave_file])
            target_str_list = (str(p) for p in target_dict[wave_file])
            
            # align the two arrays sample-by-sample. 
            pair_counts.update(zip(xml_str_list, target_str_list))

    # Extract globally unique, sorted labels for both Ground Truth (XML) and Predictions (Target)
    xml_labels = sorted(list({pair[0] for pair in pair_counts.keys()}))
    target_labels = sorted(list({pair[1] for pair in pair_counts.keys()}))

    # Create mapping dictionaries to convert string labels into integer matrix indices
    xml_idx = {lbl: i for i, lbl in enumerate(xml_labels)}
    target_idx = {lbl: i for i, lbl in enumerate(target_labels)}

    # Initialize an empty 2D numpy array: Rows = GT labels, Columns = Pred labels
    co_matrix = np.zeros((len(xml_labels), len(target_labels)), dtype=int)
    
    # Populate the matrix with paired sample counts
    for (x_lbl, t_lbl), count in pair_counts.items():
        if x_lbl in xml_idx and t_lbl in target_idx:
            co_matrix[xml_idx[x_lbl]][target_idx[t_lbl]] = count

    # Wrap the raw counts in a pandas DataFrame
    co_df = pd.DataFrame(co_matrix, index=xml_labels, columns=target_labels)
    co_df.index.name = "XML \\ Target"

    # Isolate Foreground Labels for Hungarian Algorithm -- background labels automatically map to one another
    bg_str = str(bg_label)
    xml_labels_sub = [lbl for lbl in xml_labels if lbl != bg_str]
    target_labels_sub = [lbl for lbl in target_labels if lbl != bg_str]
    
    mapping = {}
    
    # Force the background label mapping if it exists in both
    if bg_str in xml_labels and bg_str in target_labels:
        mapping[bg_str] = bg_str

    # Only run the Hungarian algorithm if there are actual foreground syllables to map
    if xml_labels_sub and target_labels_sub:
        # Extract the sub-matrix of only foreground labels
        sub_matrix_float = co_df.loc[xml_labels_sub, target_labels_sub].to_numpy().astype(float)
        
        # Normalize by columns to convert raw counts into probabilities
        col_sums = sub_matrix_float.sum(axis=0, keepdims=True)
        # Prevent division by zero for any empty columns by replacing 0s with 1.0
        col_sums[col_sums == 0] = 1.0 
        
        # Divide raw counts by the column totals
        normalized_matrix = sub_matrix_float / col_sums

        # Use Hungarian algorithm to maximize total probability 
        cost_matrix = normalized_matrix.max() - normalized_matrix
        
        # row_ind contains the XML label indices, col_ind contains the Target label indices
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Translate the matched integer indices back into their original string labels
        for r, c in zip(row_ind, col_ind):
            mapping[xml_labels_sub[r]] = target_labels_sub[c]

    # Return the final 1-to-1 dictionary mapping and the raw count dataframe
    return mapping, co_df

def get_total_audio_minutes(directory_path):
    """
    Returns the number of audio minutes represented in the wav files 
    """
    total_seconds = 0.0
    
    # Create a Path object for the directory
    dir_path = Path(directory_path)
    
    for wav_file in dir_path.glob('*.wav'):
        try:
            with wave.open(str(wav_file), 'rb') as audio:
                frames = audio.getnframes()
                rate = audio.getframerate()
                
                # Duration in seconds = total frames / frame rate
                duration_seconds = frames / float(rate)
                total_seconds += duration_seconds
                
        except wave.Error as e:
            print(f"Skipping corrupted or unsupported file {wav_file.name}: {e}")
            
    # Convert total seconds to minutes
    total_minutes = total_seconds / 60.0
    return total_minutes

def strip_unmapped(annot_dict, mapping, bg_label="-1"):
    """
    Replaces any annotation in annot_dict that is NOT mapped with the background label.
    """

    bg_str = str(bg_label)
    
    # Isolate valid labels from values (Target/Predicted)
    valid_labels = {str(v) for v in mapping.values()}
        
    valid_labels.add(bg_str)

    cleaned_dict = {}
    for filename, annots in annot_dict.items():
        cleaned_dict[filename] = [
            x if str(x) in valid_labels else bg_label 
            for x in annots
        ]
        
    return cleaned_dict

def build_transition_matrix(transition_probs, label_mapping=None):
    """
    Converts a nested transition dictionary into a pandas DataFrame matrix.
    """
    
    # If a label mapping is provided, remap the transition probabilities
    if label_mapping is not None:
        remapped_probs = {}
        for from_state, to_states in transition_probs.items():
            # Map the 'from' state using the provided mapping, default to original if not in mapping
            mapped_from = label_mapping.get(str(from_state), str(from_state))
            
            if mapped_from not in remapped_probs:
                remapped_probs[mapped_from] = {}
            
            # Map each 'to' state and accumulate probabilities for the same mapped state
            for to_state, prob in to_states.items():
                mapped_to = label_mapping.get(str(to_state), str(to_state))
                
                remapped_probs[mapped_from][mapped_to] = prob
        
        transition_probs = remapped_probs
    
    # Identify all unique states (phrases) across both 'from' and 'to' transitions
    all_states = set(transition_probs.keys())
    for to_states in transition_probs.values():
        all_states.update(to_states.keys())
        
    # Sort them so the matrix has consistent row/column ordering
    sorted_states = sorted(list(all_states))
    
    # Initialize an empty matrix filled with zeros
    matrix = pd.DataFrame(0.0, index=sorted_states, columns=sorted_states)
    
    # Populate the matrix with the probability values
    for from_state, to_states in transition_probs.items():
        for to_state, prob in to_states.items():
            matrix.at[from_state, to_state] = prob
            
    return matrix

def main():
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Process XML/PKL/JSON annotations matching WAV durations."
    )
    parser.add_argument(
        "--xml", type=str, required=True, help="Path to input XML file"
    )
    parser.add_argument(
        "--pkl", type=str, required=True, help="Path to input PKL file"
    )
    parser.add_argument(
        "--json", type=str, required=True, help="Path to input JSON file"
    )
    parser.add_argument(
        "--regions", type=str, default=None, help="Optional JSON file specifying valid segment bounds per WAV"
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        required=True,
        help="Directory containing .wav files",
    )
    parser.add_argument(
        "--sr", type=float, required=True, help="Sample rate float (e.g., 16000.0)"
    )
    parser.add_argument(
        "--hop", type=float, required=True, help="Sample hop length"
    )
    parser.add_argument(
        "--out_dir", type=str, default=".", help="Directory to save outputs"
    )
    parser.add_argument(
        "--bird", type=str, default=".", help="Bird ID for csv naming"
    )

    args = parser.parse_args()

    # use arguments to create annotations dictonaries for each wav file according to the json (tweetyBERT), pkl (CHURP), and xml (ground truth) annotations
    processed_json = parse_cluster_json(args.json, args.sr, args.hop, args.wav_dir, args.regions)
    processed_xml = xml_to_position_dict(args.xml, args.wav_dir)
    processed_pkl = process_pkl_dict(args.pkl, args.hop, args.wav_dir)


    # save annotations to files for later plotting 
    os.makedirs(args.out_dir, exist_ok=True)
    xml_out_path = os.path.join(args.out_dir, "processed_xml_annotations.pkl")
    pkl_out_path = os.path.join(args.out_dir, "expanded_pkl_annotations.pkl")
    json_out_path = os.path.join(args.out_dir, "expanded_json_annotations.pkl")
    map_out_path_1 = os.path.join(
        args.out_dir, "hungarian_label_mapping_xml_pkl.json"
    )
    map_out_path_2 = os.path.join(
        args.out_dir, "hungarian_label_mapping_xml_json.json"
    )
    with open(xml_out_path, "wb") as f:
        pickle.dump(processed_xml, f)
    with open(pkl_out_path, "wb") as f:
        pickle.dump(processed_pkl, f)
    with open(json_out_path, "wb") as f:
        pickle.dump(processed_json, f) 

    # Apply regional segment constraint if JSON map provided -- this only calculates annotations where tweetyBERT pipeline detects song
    if args.regions:
        processed_json = apply_region_mask(processed_json, args.regions, args.hop)
        processed_xml = apply_region_mask(processed_xml, args.regions, args.hop)
        processed_pkl = apply_region_mask(processed_pkl, args.regions, args.hop)

    # make co-occurance matrix and 1-1 mapping of GT syllables and clusters 
    # for CHURP
    label_mapping_1, co_df_1 = compute_hungarian_mapping(processed_xml, processed_pkl)
    # for tweetyBERT
    label_mapping_2, co_df_2 = compute_hungarian_mapping(processed_xml, processed_json)

    # strip unmapped labels from the dictionaries
    processed_pkl = strip_unmapped(processed_pkl, label_mapping_1)
    processed_json = strip_unmapped(processed_json, label_mapping_2)

    # save mappings
    with open(map_out_path_1, "w", encoding="utf-8") as f:
        json.dump(label_mapping_1, f, indent=4)
    with open(map_out_path_2, "w", encoding="utf-8") as f:
        json.dump(label_mapping_2, f, indent=4)

    # verification print statements
    print(f"\n--- Output Summary ---")
    print(f"XML Annotations Saved -> {xml_out_path}")
    print(f"PKL Annotations Saved -> {pkl_out_path}")
    print(f"JSON Annotations Saved -> {json_out_path}")
    print(f"Hungarian Mapping (XML -> PKL) Saved -> {map_out_path_1}")
    print(f"Hungarian Mapping (XML -> JSON) Saved -> {map_out_path_2}")
    print("\nOptimal 1-1 Label Mapping (XML -> PKL):")
    print(json.dumps(label_mapping_1, indent=4))
    print("\nOptimal 1-1 Label Mapping (XML -> JSON):")
    print(json.dumps(label_mapping_2, indent=4))

    # get the total number of audio minutes represented in the wav files 
    minutes = get_total_audio_minutes(args.wav_dir)

    # calculate per_sample frame error rates
    fer_per_sample_birdsong = []

    for wave_file in processed_xml.keys():
        if wave_file in processed_pkl:
            gt_labels = [str(x) for x in processed_xml[wave_file]]
            pred_labels = [str(p) for p in processed_pkl[wave_file]]
            
            total_frames = len(gt_labels)
            if total_frames == 0:
                continue
                
            errors = 0
            for gt, pred in zip(gt_labels, pred_labels):
                # Check what the ground truth label was mapped to.
                # If the prediction doesn't match the mapped target, it's an error.
                if label_mapping_1.get(gt) != pred:
                    errors += 1
                    
            # Calculate error rate (0.0 to 1.0)
            fer_per_sample_birdsong.append(errors / total_frames)

    fer_per_sample_tweety = []

    for wave_file in processed_xml.keys():
        if wave_file in processed_json:
            gt_labels = [str(x) for x in processed_xml[wave_file]]
            pred_labels = [str(p) for p in processed_json[wave_file]]
            
            total_frames = len(gt_labels)
            if total_frames == 0:
                continue
                
            errors = 0
            for gt, pred in zip(gt_labels, pred_labels):
                # Check what the ground truth label was mapped to.
                # If the prediction doesn't match the mapped target, it's an error.
                if label_mapping_2.get(gt) != pred:
                    errors += 1
                    
            # Calculate error rate (0.0 to 1.0)
            fer_per_sample_tweety.append(errors / total_frames)

    # get the mean durations per syllable and the probabilites of each syllable transition pair
    freq_xml, mean_durations_xml, _, transition_probs_xml = compute_phrase_metrics(processed_xml, bg_labels=("-1", -1))
    freq_pkl, mean_durations_pkl, _, transition_probs_pkl = compute_phrase_metrics(processed_pkl, bg_labels=("-1", -1))
    freq_json, mean_durations_json, _, transition_probs_json = compute_phrase_metrics(processed_json, bg_labels=("-1", -1))

    # calculate number of syllables ground truth and learned
    num_xml_sybs = sum(freq_xml.values())
    num_pkl_sybs = sum(freq_pkl.values())
    num_json_sybs = sum(freq_json.values())
    
    # calculate min-max similarity for syllables
    similarity_birdsong = []
    mapped_keys = set(label_mapping_1.keys())
    
    for key in mapped_keys:
        if key == "-1":
            continue
        
        # Get the mean duration for each label
        val_a = mean_durations_xml.get(key)
        val_b = mean_durations_pkl.get(label_mapping_1.get(key))
        
        intersection = min(val_a, val_b)
        union = max(val_a, val_b)
        
        similarity_birdsong.append(intersection / union)

    similarity_tweety = []
    mapped_keys = set(label_mapping_2.keys())
        
    for key in mapped_keys:
        if key == "-1":
            continue
            
        # Get the mean duration for each label
        val_a = mean_durations_xml.get(key)
        val_b = mean_durations_json.get(label_mapping_2.get(key))
        
        intersection = min(val_a, val_b)
        union = max(val_a, val_b)
        
        similarity_tweety.append(intersection / union)

    # make transition probability matrices
    gt_matrix = build_transition_matrix(transition_probs_xml)
    reversed_map_1 = {v: k for k, v in label_mapping_1.items()}
    birdsong_matrix = build_transition_matrix(transition_probs_pkl, label_mapping=reversed_map_1)
    reversed_map_2 = {v: k for k, v in label_mapping_2.items()}
    tweety_matrix = build_transition_matrix(transition_probs_json, label_mapping=reversed_map_2)

    stats = {
        'per_record_fer_tweety' : fer_per_sample_tweety, 
        'per_record_fer_birdsong' : fer_per_sample_birdsong,
        'per_syb_similarity_tweety' : similarity_tweety,
        'per_syb_similarity_birdsong' : similarity_birdsong,
        'transition_matrix_tweety' : tweety_matrix,
        'transition_matrix_birdsong' : birdsong_matrix,
        'transition_matrix_gt' : gt_matrix
    }

    # save statistics as pkl file
    stats_path = os.path.join(args.out_dir, f"comparison_stats_{args.bird}.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    # Compute Phrase orrelations of entropy
    # Compare XML (ground truth) against CHURP (PKL) 
    birdsong_corr_dur, birdsong_corr_ent = evaluate_mapped_metrics(
        processed_xml, processed_pkl, label_mapping_1
    )
    
    # Compare XML (ground truth) against TweetyBERT (JSON) 
    tweety_corr_dur, tweety_corr_ent = evaluate_mapped_metrics(
        processed_xml, processed_json, label_mapping_2
    )

    # store all the comparison metrics in a csv file
    file_exists = os.path.exists(f"{args.bird}_comparison_data.csv")
    with open(f"{args.bird}_comparison_data.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
   
        if not file_exists:
            writer.writerow(['Syllables From Tweety', 'Syllables from Birdsong', 'Ground Truth Syllables', 'Tweety Entropy Cor', 'Birdsong Entropy Cor', 'Tweety Duration Cor', 'Birdsong Duration Cor', 'Audio Minutes'])

        raw_data = [
            num_json_sybs, num_pkl_sybs, num_xml_sybs, tweety_corr_ent, birdsong_corr_ent, 
            tweety_corr_dur, birdsong_corr_dur, minutes
        ]

        formatted_data = [f"{val:.2f}" for val in raw_data]

        writer.writerow(formatted_data)

if __name__ == "__main__":
    main()
