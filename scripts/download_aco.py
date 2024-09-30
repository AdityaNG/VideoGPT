"""
Download the ACO dataset into the following tree structure
the code supports a dataset with the following directory structure:
```
video_dataset/
    train/
        video1.mp4
        video2.mp4
        ...
    test/
        video1.mp4
        video2.mp4
        ...
```

Randomly select 5% of the ACO videos for testing

The `aco_dataset.csv` is of the following format
```csv
title,id,hour,minute,second,continent,weather
,,48,4215,3321,,
Amsterdam 4K - Morning Drive - Canals & Bicycles,YQMPjBxM-hY,0,41,17,,
Amsterdam 4K - Driving Downtown - Sunset Drive,Ve4DvQNnDDQ,0,57,10,,
...
```

- You can use youtube-dlp `yt-dlp` shell script to download the videos.
- Show a progres bar
- Make sure to donwload all the videos at a maximum resolution of 480p
- Make sure the script is able to correctly continue downloading where it stopped
"""

import os
import csv
import random
from tqdm import tqdm
import subprocess

def download_video(video_id, output_path):
    command = [
        'yt-dlp',
        f'https://www.youtube.com/watch?v={video_id}',
        '-o', output_path,
        '-f', 'bv*[height<=360]+ba/b[height<=360]/wv*+ba/w',
        '--no-overwrites',
        '--newline',
        '--merge-output-format', 'mkv',
        '--ffmpeg-location', '/usr/bin/ffmpeg',
    ]
    print(' '.join(command))
    
    process = subprocess.Popen(
        command,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        # universal_newlines=True,
    )
    
    process.wait()

def main():
    csv_file = 'aco_dataset.csv'
    output_dir = 'video_dataset'
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        videos = list(reader)

    random.shuffle(videos)
    test_size = int(len(videos) * 0.05)
    test_videos = videos[:test_size]
    train_videos = videos[test_size:]

    for dataset, videos in [('test', test_videos), ('train', train_videos)]:
        output_folder = test_dir if dataset == 'test' else train_dir
        for video in tqdm(videos, desc=f"Processing {dataset} videos"):
            video_id = video['id']
            if not video_id:
                continue
            output_path = os.path.join(output_folder, f'{video_id}.%(ext)s')
            if not os.path.exists(os.path.join(output_folder, f'{video_id}.mkv')):
                download_video(video_id, output_path)

if __name__ == "__main__":
    main()
