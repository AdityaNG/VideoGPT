import os
import argparse
import torch

from videogpt import VideoData, VideoGPT, load_videogpt
from videogpt.utils import save_video_grid

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="video_dataset/")
    parser.add_argument('--ckpt', type=str, default='lightning_logs/version_1/checkpoints/last.ckpt')
    parser.add_argument('--batch_size', type=int, default=1)

    parser = VideoGPT.add_model_specific_args(parser)
    return parser.parse_args()

def generate_video_samples(gpt: VideoGPT, batch, batch_size: int):
    batch = {k: v.cuda() for k, v in batch.items()}

    return gpt.sample(batch_size, batch)

def main():
    args = parse_arguments()

    if not os.path.exists(args.ckpt):
        gpt = load_videogpt(args.ckpt)
    else:
        gpt = VideoGPT.load_from_checkpoint(args.ckpt)
    gpt = gpt.cuda()
    gpt.eval()
    args = gpt.hparams['args']
    
    data = VideoData(args)
    loader = data.test_dataloader()
    batch = next(iter(loader))

    samples = generate_video_samples(gpt, batch, args.batch_size)
    save_video_grid(samples, 'samples.mp4')

if __name__ == "__main__":
    main()
