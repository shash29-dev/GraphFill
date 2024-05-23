import os

import cv2
import numpy as np

from src.training.visualizers.base import BaseVisualizer, visualize_mask_and_images_batch
from src.utils import check_and_warn_input_range
import pdb
import matplotlib.pyplot as plt

class DirectoryVisualizer(BaseVisualizer):
    DEFAULT_KEY_ORDER = 'image predicted_image inpainted'.split(' ')

    def __init__(self, outdir, key_order=DEFAULT_KEY_ORDER, max_items_in_batch=10,
                 last_without_mask=True, rescale_keys=None):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.key_order = key_order
        self.max_items_in_batch = max_items_in_batch
        self.last_without_mask = last_without_mask
        self.rescale_keys = rescale_keys

    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        check_and_warn_input_range(batch['image'], 0, 1, 'DirectoryVisualizer target image')
        vis_img = visualize_mask_and_images_batch(batch, self.key_order, max_items=self.max_items_in_batch,
                                                  last_without_mask=self.last_without_mask,
                                                  rescale_keys=self.rescale_keys)

        vis_img = np.clip(vis_img * 255, 0, 255).astype('uint8')

        curoutdir = os.path.join(self.outdir, f'epoch{epoch_i:04d}{suffix}')
        os.makedirs(curoutdir, exist_ok=True)
        rank_suffix = f'_r{rank}' if rank is not None else ''
        out_fname = os.path.join(curoutdir, f'batch{batch_i:07d}{rank_suffix}.jpg')

        cv2.imwrite(out_fname, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        return vis_img

class DirectoryVisualizerC2F(BaseVisualizer):
    DEFAULT_KEY_ORDER = 'image predicted_image inpainted'.split(' ')

    def __init__(self, outdir, key_order=DEFAULT_KEY_ORDER, max_items_in_batch=10,
                 mask_only_first=True, rescale_keys=None):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.key_order = key_order 
        self.max_items_in_batch = max_items_in_batch
        self.mask_only_first = mask_only_first
        self.rescale_keys = rescale_keys

    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        check_and_warn_input_range(batch['image'], 0, 1, 'DirectoryVisualizer target image')
        vis_batch ={ 'image': batch['image'],
                    'mask': batch['mask'],
                    'predicted_image': batch['predicted_image'],
                    'inpainted': batch['inpainted'],
                    'coarse_avg': batch['coarse_out']['coarse_avg'],
        }
        vis_img = visualize_mask_and_images_batch(vis_batch, self.key_order +['coarse_avg'], max_items=2,mask_only_first=self.mask_only_first,rescale_keys=self.rescale_keys)
        vis_coarse=[]
        for key in batch['coarse_out']['coarse_pred'].keys():
            img= batch['coarse_out']['coarse_gt'][key]
            predicted_img = batch['coarse_out']['coarse_pred'][key]
            original_mask = 1-batch['coarse_out']['coarse_mask'][key]
            vis_batch ={'image_rf': batch['image'], 
                        'image': img,
                        'mask': original_mask,
                        'predicted_image': predicted_img,
                        'inpainted': (1-original_mask) * predicted_img + original_mask * img,
                        }
            keys = list(vis_batch.keys())
            keys.remove('mask')
            vis_coarse.append(visualize_mask_and_images_batch(vis_batch, keys, max_items=2,mask_only_first=self.mask_only_first,rescale_keys=self.rescale_keys))
        
        coarse_im=np.vstack([x for x in vis_coarse])
        vis_img = np.clip(vis_img * 255, 0, 255).astype('uint8')
        coarse_im = np.clip(coarse_im * 255, 0, 255).astype('uint8')
        curoutdir = os.path.join(self.outdir, f'epoch{epoch_i:04d}{suffix}')
        os.makedirs(curoutdir, exist_ok=True)
        rank_suffix = f'_r{rank}' if rank is not None else ''
        out_fname = os.path.join(curoutdir, f'batch{batch_i:07d}{rank_suffix}_refine.jpg')
        cv2.imwrite(out_fname, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        out_fname = os.path.join(curoutdir, f'batch{batch_i:07d}{rank_suffix}_coarse.jpg')
        cv2.imwrite(out_fname, cv2.cvtColor(coarse_im, cv2.COLOR_RGB2BGR))
        return vis_img, coarse_im
