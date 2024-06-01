from argparse import ArgumentParser

import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import BatchSizeFinder

from gan_module import AgingGAN

import fid_evaluasi

parser = ArgumentParser()
parser.add_argument('--config', default='configs/aging_gan.yaml', help='Config to use for training')


def main():
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)
    model = AgingGAN(config)
    trainer = Trainer(max_epochs=config['epochs'], accelerator='auto', callbacks=BatchSizeFinder(mode='binsearch'))
    trainer.fit(model)

    fid_score = fid_evaluasi.compute_fid(model.genA2B, model.train_dataloader())
    print(fid_score)


if __name__ == '__main__':
    main()
