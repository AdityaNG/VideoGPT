import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from videogpt import VQVAE, VideoData


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQVAE.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, default='video_dataset')
    parser.add_argument('--sequence_length', type=int, default=8)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()  # 411883
    data.test_dataloader()
    model = VQVAE(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(
        monitor='val/recon_loss',
        mode='min',
        every_n_val_epochs=1,
        save_top_k=5,
    ))

    # Saving the last checkpoint
    callbacks.append(ModelCheckpoint(
        filename='last-{epoch:02d}-{step:06d}',
        every_n_train_steps=100,
        save_top_k=-1,  # Keep all checkpoints
        save_last=False,
    ))

    kwargs = dict()
    if args.gpus > 1:
        kwargs = dict(distributed_backend='ddp', gpus=args.gpus)
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, max_steps=400000, **kwargs
    )

    trainer.fit(model, data)


if __name__ == '__main__':
    main()

