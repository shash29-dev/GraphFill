import logging
import torch
from src.training.trainers.gan import InpaintingTrainingModule
from src.training.trainers.c2f import C2FInpaintingTrainingModule

def get_training_model_class(kind):
    if kind.startswith('c2f'):
        return C2FInpaintingTrainingModule
    if kind.startswith('def_'):
        return InpaintingTrainingModule

    raise ValueError(f'Unknown model kind {kind}')

def make_training_model(cfg):
    kind = cfg.kind
    logging.info('Make Training Model\t {}'.format(kind))
    cls = get_training_model_class(kind)
    return cls(cfg)