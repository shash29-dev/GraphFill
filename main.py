
import os 
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import hydra
import sys
import numpy as np
from omegaconf import OmegaConf
import pdb
import logging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib
matplotlib.use('TkAgg')
import torch
from src.training.trainers import make_training_model
import time
LOGGER = logging.getLogger(__name__)

def main(cfg: OmegaConf) -> None:

    LOGGER.info(OmegaConf.to_yaml(cfg))
    metrics_logger = TensorBoardLogger(cfg.tb_dir, name=os.path.basename(os.getcwd()))
    metrics_logger.log_hyperparams(cfg)

    checkpoints_dir = cfg.save_model_dir
    os.makedirs(checkpoints_dir, exist_ok=True)
    training_model = make_training_model(cfg)

    trainer_kwargs = OmegaConf.to_container(cfg.trainer.kwargs, resolve=True)

    if cfg.util_args.eval_mode==True:
        # Load Model here and exempt lightning load to deal with extra_evaluator key mismatch. If added/removed any.
        print('Loading Model @ {}'.format(cfg.model_load))
        loaded_model=torch.load(cfg.model_load)['state_dict']
        weights = training_model.state_dict()
        for key in training_model.state_dict().keys():
            if key.startswith('coarse_model'):  
                weights[key]=loaded_model[key]
            elif key.startswith('refine_model'):
                weights[key]=loaded_model[key]
        training_model.load_state_dict(weights)
        trainer = Trainer(
            callbacks=ModelCheckpoint(dirpath=cfg.save_model_dir, **cfg.trainer.checkpoint_kwargs),
            logger= metrics_logger,
            **trainer_kwargs
        )
        trainer.validate(training_model)
    else:
        trainer = Trainer(
            callbacks=ModelCheckpoint(dirpath=cfg.save_model_dir, **cfg.trainer.checkpoint_kwargs),
            logger= metrics_logger,
            resume_from_checkpoint=cfg.save_model_dir+'/last.ckpt',  
            **trainer_kwargs
        )
        trainer.fit(training_model)
    

if __name__ == "__main__":
    config_path = "./configs/"
    if len(sys.argv) > 1 and sys.argv[1].startswith("config="):
        config_name = sys.argv[1].split("=")[-1]
        sys.argv.pop(1)

    main_wrapper = hydra.main(config_path=config_path, config_name=config_name,version_base=None)
    main_wrapper(main)()