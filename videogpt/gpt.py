import os
import itertools
from datetime import datetime
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl

from .resnet import resnet34
from .attention import AttentionStack, LayerNorm, AddBroadcastPosEmbed
from .utils import shift_dim, save_video_grid
class VideoGPT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load VQ-VAE and set all parameters to no grad
        from .vqvae import VQVAE
        from .download import load_vqvae
        if not os.path.exists(args.vqvae):
            self.vqvae = load_vqvae(args.vqvae)
        else:
            print('vqvae from ckpt', args.vqvae)
            self.vqvae = VQVAE.load_from_checkpoint(args.vqvae)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        # self.vqvae.eval()  # removed this because it causes a bug

        # ResNet34 for frame conditioning
        self.use_frame_cond = args.n_cond_frames > 0
        if self.use_frame_cond:
            frame_cond_shape = (args.n_cond_frames,
                                args.resolution // 4,
                                args.resolution // 4,
                                240)
            self.resnet = resnet34(1, (1, 4, 4), resnet_dim=240)
            self.cond_pos_embd = AddBroadcastPosEmbed(
                shape=frame_cond_shape[:-1], embd_dim=frame_cond_shape[-1]
            )
        else:
            frame_cond_shape = None

        # VideoGPT transformer
        self.shape = self.vqvae.latent_shape

        self.fc_in = nn.Linear(self.vqvae.embedding_dim, args.hidden_dim, bias=False)
        self.fc_in.weight.data.normal_(std=0.02)

        self.attn_stack = AttentionStack(
            self.shape, args.hidden_dim, args.heads, args.layers, args.dropout,
            args.attn_type, args.attn_dropout, args.class_cond_dim, frame_cond_shape
        )

        self.norm = LayerNorm(args.hidden_dim, args.class_cond_dim)

        self.fc_out = nn.Linear(args.hidden_dim, self.vqvae.n_codes, bias=False)
        self.fc_out.weight.data.copy_(torch.zeros(self.vqvae.n_codes, args.hidden_dim))

        # caches for faster decoding (if necessary)
        self.frame_cond_cache = None

        self.save_hyperparameters()

    def get_reconstruction(self, videos):
        return self.vqvae.decode(self.vqvae.encode(videos))

    def sample(self, n, batch=None):
        device = self.fc_in.weight.device

        cond = dict()
        if self.use_frame_cond or self.args.class_cond:
            assert batch is not None
            video = batch['video']

            if self.args.class_cond:
                label = batch['label']
                cond['class_cond'] = F.one_hot(label, self.args.class_cond_dim).type_as(video)
            if self.use_frame_cond:
                cond['frame_cond'] = video[:, :, :self.args.n_cond_frames]

        samples = torch.zeros((n,) + self.shape).long().to(device)
        idxs = list(itertools.product(*[range(s) for s in self.shape]))

        with torch.no_grad():
            prev_idx = None
            for i, idx in enumerate(tqdm(idxs)):
                batch_idx_slice = (slice(None, None), *[slice(i, i + 1) for i in idx])
                batch_idx = (slice(None, None), *idx)
                embeddings = self.vqvae.codebook.dictionary_lookup(samples)

                if prev_idx is None:
                    # set arbitrary input values for the first token
                    # does not matter what value since it will be shifted anyways
                    embeddings_slice = embeddings[batch_idx_slice]
                    samples_slice = samples[batch_idx_slice]
                else:
                    embeddings_slice = embeddings[prev_idx]
                    samples_slice = samples[prev_idx]

                logits = self(embeddings_slice, samples_slice, cond,
                              decode_step=i, decode_idx=idx)[1]
                # squeeze all possible dim except batch dimension
                logits = logits.squeeze().unsqueeze(0) if logits.shape[0] == 1 else logits.squeeze()
                probs = F.softmax(logits, dim=-1)
                samples[batch_idx] = torch.multinomial(probs, 1).squeeze(-1)

                prev_idx = batch_idx_slice
            samples = self.vqvae.decode(samples)
            samples = torch.clamp(samples, -0.5, 0.5) + 0.5

        return samples # BCTHW in [0, 1]


    def forward(self, x, targets, cond, decode_step=None, decode_idx=None):
        if self.use_frame_cond:
            if decode_step is None:
                cond['frame_cond'] = self.cond_pos_embd(self.resnet(cond['frame_cond']))
            elif decode_step == 0:
                self.frame_cond_cache = self.cond_pos_embd(self.resnet(cond['frame_cond']))
                cond['frame_cond'] = self.frame_cond_cache
            else:
                cond['frame_cond'] = self.frame_cond_cache

        h = self.fc_in(x)
        h = self.attn_stack(h, cond, decode_step, decode_idx)
        h = self.norm(h, cond)
        logits = self.fc_out(h)

        loss = F.cross_entropy(shift_dim(logits, -1, 1), targets)

        return loss, logits

    def training_step(self, batch, batch_idx):
        # self.vqvae.eval()  vqvae.codebook._need_init = False
        x = batch['video'].clone()

        cond = dict()
        if self.args.class_cond:
            label = batch['label']
            cond['class_cond'] = F.one_hot(label, self.args.class_cond_dim).type_as(x)
        if self.use_frame_cond:
            cond['frame_cond'] = x[:, :, :self.args.n_cond_frames]

        with torch.no_grad():
            targets, x = self.vqvae.encode(x, include_embeddings=True)
            x = shift_dim(x, 1, -1)

        loss, _ = self(x, targets, cond)

        self.log('train/loss', loss)

        if batch_idx % 1000 == 0:
            with torch.no_grad():
                batch_size = x.shape[0]

                samples_gt = torch.clamp(batch['video'].clone(), -0.5, 0.5) + 0.5

                _, samples_vqvae, _ = self.vqvae.forward(batch['video'].clone())
                samples_vqvae = torch.clamp(samples_vqvae, -0.5, 0.5) + 0.5

                samples = self.sample(batch_size, batch)

                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_video = os.path.join(
                    self.logger.log_dir,
                    f"train_{current_time}_{str(batch_idx).zfill(8)}.mp4"
                )
                output_video_vqvae = os.path.join(
                    self.logger.log_dir,
                    f"train_vqvae_{current_time}_{str(batch_idx).zfill(8)}.mp4"
                )
                output_video_gt = os.path.join(
                    self.logger.log_dir,
                    f"train_gt_{current_time}_{str(batch_idx).zfill(8)}.mp4"
                )

                # samples -> (B, C, T, H, W)
                save_video_grid(samples, output_video)
                save_video_grid(samples_vqvae, output_video_vqvae)
                save_video_grid(samples_gt, output_video_gt)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--vqvae', type=str, default='kinetics_stride4x4x4',
                            help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--n_cond_frames', type=int, default=0)
        parser.add_argument('--class_cond', action='store_true')

        # VideoGPT hyperparmeters
        parser.add_argument('--hidden_dim', type=int, default=576)
        parser.add_argument('--heads', type=int, default=4)
        parser.add_argument('--layers', type=int, default=32)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--attn_type', type=str, default='full',
                            choices=['full', 'sparse'])
        parser.add_argument('--attn_dropout', type=float, default=0.3)

        return parser
