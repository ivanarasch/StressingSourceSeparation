#!/bin/bash

# Output root
OUTPUT_ROOT="sample/output/path"

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
