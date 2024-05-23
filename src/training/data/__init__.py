import pdb
import logging
from src.training.data.datasets import InpaintingTrainDataset, InpaintingValDataset, InpaintingTrainSpixDataset, InpaintingValSpixDataset

from src.training.data.mask_generator import MixedMaskGenerator

from src.training.data.transforms import get_transforms 
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as spDataLoader
LOGGER = logging.getLogger(__name__)


def make_train_dataloader(cfg, kind='xxx',cfgspix=None):
    LOGGER.info(f'Make train dataloader {cfg.kind} from {cfg.train.indir}. Using mask generator={cfg.mask_generator_kind}')
    mask_generator = MixedMaskGenerator(**cfg.train.mask_generator_kwargs)
    transform = get_transforms(cfg.train.transform_variant, cfg.train.out_size)
    if kind.startswith('def_'):
        if cfg.kind == 'davis' or cfg.kind.startswith('places') or cfg.kind.startswith('celebA'):
            dataset = InpaintingTrainDataset(indir=cfg.train.indir,
                                            kind=cfg.kind,
                                            mask_generator=mask_generator,
                                            outsize=cfg.train.out_size,
                                            transform=transform)
            loader=DataLoader
    elif kind.startswith('c2f') or kind.startswith('graphFill') or kind.startswith('refine') or kind.startswith('finetune'):
        if cfg.kind == 'davis' or cfg.kind.startswith('places') or cfg.kind.startswith('celebA'):
            dataset = InpaintingTrainSpixDataset(indir=cfg.train.indir,
                                            kind=cfg.kind,
                                            mask_generator=mask_generator,
                                            transform=transform,
                                            cfgspix=cfgspix,
                                            outsize=cfg.train.out_size,
                                            pickle_data=cfg.train.pickle_data)
            loader=spDataLoader
    else:
        raise ValueError(f'Unknown model kind {kind}')

    dataloader = loader(dataset, **cfg.train.dataloader_kwargs)
    return dataloader

def make_val_dataloader(indir, img_suffix, kind='default',pickle_data=False, val_save=False, dataloader_kwargs=None, cfgspix=None):
    
    LOGGER.info(f'Make val dataloader from {indir}')
    if kind.startswith('def_'):
        dataset = InpaintingValDataset(datadir=indir,
                                        img_suffix=img_suffix)
        loader=DataLoader
    elif kind.startswith('c2f') or kind.startswith('graphFill') or kind.startswith('refine') or kind.startswith('finetune'):
        dataset = InpaintingValSpixDataset(datadir=indir,
                                        img_suffix=img_suffix,
                                        cfgspix=cfgspix,
                                        pickle_data=pickle_data,
                                        )
        loader=spDataLoader
    else:
        raise ValueError(f'Unknown model kind {kind}')
    
    dataloader = loader(dataset, **dataloader_kwargs)
    return dataloader