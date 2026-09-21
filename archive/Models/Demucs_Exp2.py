import os
import subprocess

data_root = "/Sample/Data/Path"
output_root = "/Sample/Output/Path"

os.makedirs(output_root, exist_ok=True)

for song_folder in os.listdir(data_root):
    song_path = os.path.join(data_root, song_folder)
    if not os.path.isdir(song_path):
        continue

    # Make output folder for this song
    song_output_folder = os.path.join(output_root, song_folder)
    os.makedirs(song_output_folder, exist_ok=True)

    # Loop through every FX version inside the song folder
    for audio_file in os.listdir(song_path):
        if not audio_file.lower().endswith((".wav", ".mp3", ".flac")):
            continue

        audio_path = os.path.join(song_path, audio_file)
        fx_name = os.path.splitext(audio_file)[0]

        fx_output_folder = os.path.join(song_output_folder, fx_name)
        os.makedirs(fx_output_folder, exist_ok=True)

        print(f"Processing: {audio_file}")
        print(f"Saving to: {fx_output_folder}")

        # Run Demucs
        subprocess.run([
            "demucs",
            audio_path,
            "-o", fx_output_folder
        ])

