import torch 
import glob
import cv2
import imutils
import PIL.Image as Image
from src.training.data.image2graph import Image2Graph
import hydra
import numpy as np
import os 
from omegaconf import OmegaConf
import pdb
import logging
import matplotlib
# matplotlib.use('TkAgg')
import torch
import matplotlib.pyplot as plt
import pickle
import tqdm
import sys
from multiprocessing import Pool
from src.training.data.mask_generator import MixedMaskGenerator

from src.training.data.transforms import get_transforms 

   
class InpaintingTrainSpixDataset(object):
    def __init__(self, indir, kind, mask_generator, transform, cfgspix,outsize=256):
        self.indir=indir
        
        ## Refactored code --- Load appropriate train/val splits---
        self.in_files = sorted(list(glob.glob(os.path.join(indir, '**' , '*.jpg'), recursive=True)))
        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0
        self.outsize=outsize
        if cfgspix.generate==True:
            self.spixgen= Image2Graph(**cfgspix)
        else:
            self.spixgen=None

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = imutils.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),height=self.outsize)
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        mask = self.mask_generator(img, iter_i=self.iter_i)
        self.iter_i += 1
        imname = "_".join(path.split(self.indir)[-1].split(os.sep)[1:])
        if self.spixgen is not None:
            spix_info, seg_dict=self.spixgen.get_data(img,mask)
            return dict(
                    image=torch.from_numpy(img),
                    mask=torch.from_numpy(mask), 
                    spix_info=spix_info,
                    seg= seg_dict,
                    imname=imname
                    )
        else:
            return dict(image=img,
                        mask=mask)

class Engine(object):
    def __init__(self,cfgA):
        self.cfg=cfgA
        self.save_fol = './pickleData/'+ cfgA.data.kind+os.sep
        if not os.path.exists(self.save_fol): os.makedirs(self.save_fol)
        cfg= cfgA.data
        mask_generator = MixedMaskGenerator(**cfg.train.mask_generator_kwargs)
        transform = get_transforms(cfg.train.transform_variant, cfg.train.out_size)
        self.dataset = InpaintingTrainSpixDataset(indir=cfg.train.indir,
                                                kind=cfg.kind,
                                                mask_generator=mask_generator,
                                                transform=transform,
                                                cfgspix=cfgA.spix_gen,
                                                outsize=cfg.train.out_size)
        self.dl = len(self.dataset)
        already_pickled= glob.glob(self.save_fol+'/*.pkl')
        self.indices = [int(x.split(os.sep)[-1].split('_')[0]) for x in already_pickled]


    def __call__(self, idx):
        if idx not in self.indices:
            data = self.dataset[idx]
            imname = str(idx+1)+'_'+ data['imname']
            pklname = self.save_fol + imname.split('.')[0]+'.pkl'
            with open(pklname, 'wb') as f: pickle.dump(data,f)
        else:
            print('Skipping {}\t already pickled'.format(idx))
    

LOGGER = logging.getLogger(__name__)

def main(cfgA: OmegaConf) -> None:
    LOGGER.info(OmegaConf.to_yaml(cfgA))

    # Change as per resources
    processes = 14
    chunksize=20
    pool= Pool(processes)
    engine = Engine(cfgA)
    print('Starting Pool...')
    for i, _ in enumerate(pool.imap(engine,range(engine.dl),chunksize=chunksize)):
        sys.stderr.write('\r***********Done {}/{}**************'.format(i,engine.dl))
   
       
if __name__ == "__main__":
    config_path = "./configs/"
    if len(sys.argv) > 1 and sys.argv[1].startswith("config="):
        config_name = sys.argv[1].split("=")[-1]
        sys.argv.pop(1)

    main_wrapper = hydra.main(config_path=config_path, config_name=config_name,version_base=None)
    main_wrapper(main)()