#!/bin/bash
# Experiment 1: Demucs 2-stem (vocals / no_vocals) on each paired mixture.
# Run from inside the folder that holds one sub-folder per song (each with drums_vocals.wav, bass_vocals.wav, ...).
# Output layout: ../separated_outputs_experiment1/<song>_<pair>/{vocals,no_vocals}.wav

# Output root
OUTPUT_ROOT="../separated_outputs_experiment1"

mkdir -p "$OUTPUT_ROOT"

for song in */; do
    echo "Processing song folder: $song"
    song_name=$(basename "$song")

    for stem_file in "$song"/*.wav; do
        stem_name=$(basename "$stem_file" .wav)
        
        # Example: drums_vocals.wav -> vocals_drums
        # Split the name
        IFS='_' read -r inst vocals <<< "$stem_name"
        output_folder="$OUTPUT_ROOT/${song_name}_${inst}"
        mkdir -p "$output_folder"

        echo "  Running Demucs on $stem_file, output: $output_folder"

        # Run Demucs 2-stem separation
        demucs --two-stems=vocals "$stem_file" -o "$output_folder"
        
        # Move files from demucs output to folder root
        demucs_out="$output_folder/htdemucs/$(basename "$stem_file" .wav)"
        if [ -d "$demucs_out" ]; then
            mv "$demucs_out"/* "$output_folder/"
            rm -rf "$output_folder/htdemucs"
        fi
    done
done

exit 0
