import os
import argparse
import torch

import pytorch_lightning as pl

from videogpt import VideoData, VQVAE, load_vqvae
from videogpt.utils import save_video_grid

def parse_arguments():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQVAE.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, default='video_dataset')
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--ckpt', type=str, default='ckpts/vqvae_step_007899.ckpt')

    args = parser.parse_args()

    return args

@torch.no_grad()
def generate_video_samples(vqvae: VQVAE, batch):
    batch = {k: v.cuda() for k, v in batch.items()}
    x = batch['video']
    _, samples, _ = vqvae.forward(x)
    samples = torch.clamp(samples, -0.5, 0.5) + 0.5
    return samples

def main():
    args = parse_arguments()

    if not os.path.exists(args.ckpt):
        vqvae = load_vqvae(args.ckpt)
    else:
        vqvae = VQVAE.load_from_checkpoint(args.ckpt)
        vqvae.codebook._need_init = False
    vqvae = vqvae.cuda()
    # vqvae.eval()  # removed this because it causes a bug
    args = vqvae.hparams['args']
    
    data = VideoData(args)

    loader = data.test_dataloader()
    batch = next(iter(loader))

    with torch.no_grad():
        samples = generate_video_samples(vqvae, batch)
        save_video_grid(samples, 'samples.mp4')

if __name__ == "__main__":
    main()
