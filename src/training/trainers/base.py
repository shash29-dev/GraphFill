import logging 
import pytorch_lightning as ptl
import torch
import pdb
from src.training.data import make_train_dataloader, make_val_dataloader
import matplotlib.pyplot as plt
from src.utils import add_prefix_to_keys, average_dicts, flatten_dict, set_requires_grad
from src.training.modules import make_discriminator, make_generator
from src.training.losses.adversarial import make_discrim_loss
from src.training.visualizers import make_visualizer
from torch import nn
import pandas as pd
from src.evaluation import make_evaluator
import numpy as np
import lpips 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.segmentation import slic, mark_boundaries
import os

LOGGER = logging.getLogger(__name__)

def make_optimizer(parameters, kind='adamw', **kwargs):
    if kind == 'adam':
        optimizer_class = torch.optim.Adam
    elif kind == 'adamw':
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f'Unknown optimizer kind {kind}')
    return optimizer_class(parameters, **kwargs)

class BaseInpaintingModule(ptl.LightningModule):
    def __init__(self,cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        LOGGER.info('BaseInpaintingModule init called..')
        self.cfg=cfg

        if self.cfg.model.concat_mask:
            self.cfg.model.generator.coarse.indim = 4 
        
        self.coarse_model= make_generator(**self.cfg.model.generator.coarse)
        self.refine_model= make_generator(**self.cfg.model.generator.refine)
        if self.cfg.util_args.predict_only==False:
            self.discriminator = make_discriminator(**self.cfg.model.discriminator)
            self.adversarial_loss = make_discrim_loss(**self.cfg.losses.adversarial)
            self.visualizer = make_visualizer(**self.cfg.visualizer)
            self.val_evaluator = make_evaluator(**self.cfg.evaluator)
            set_requires_grad(self.val_evaluator.scores['lpips'],False)
            extra_val = self.cfg.data.get('extra_val', ())
            if extra_val:
                self.extra_val_titles = list(extra_val)
                self.extra_evaluators = nn.ModuleDict({k: make_evaluator(**self.cfg.evaluator)
                                                       for k in extra_val})
                for k,v in self.extra_evaluators.items(): set_requires_grad(v.scores['lpips'],False)
            else:
                self.extra_evaluators = {}

            if self.cfg.losses.get("l1", {"weight_known": 0})['weight_known'] > 0:
                self.loss_l1 = nn.L1Loss(reduction='none')

            if self.cfg.losses.get("mse", {"weight": 0})['weight'] > 0:
                self.loss_mse = nn.MSELoss(reduction='none')
            
            from src.training.losses.perceptual import PerceptualLoss, ResNetPL
            if self.cfg.losses.perceptual.weight > 0:
                self.loss_pl = PerceptualLoss()
                # set_requires_grad(self.loss_pl,False)

            if self.cfg.losses.get("resnet_pl", {"weight": 0})['weight'] > 0:
                self.loss_lpips = lpips.LPIPS(net='vgg')
                self.loss_resnet_pl = ResNetPL(**self.cfg.losses.resnet_pl)
                # set_requires_grad(self.loss_resnet_pl,False)
        

    def train_dataloader(self): 
        dataloader = make_train_dataloader(self.cfg.data, kind=self.cfg.kind, cfgspix=self.cfg.spix_gen)
        return dataloader
    
    def val_dataloader(self):
        res = [make_val_dataloader(**self.cfg.data.val, kind=self.cfg.kind,cfgspix=self.cfg.spix_gen)]
        extra_val = self.cfg.data.get('extra_val', ())
        if extra_val:
            res += [make_val_dataloader(**extra_val[k], kind=self.cfg.kind,cfgspix=self.cfg.spix_gen) for k in self.extra_val_titles]
        return res

    def configure_optimizers(self):
        discriminator_params = list(self.discriminator.parameters())
        return [
            dict(optimizer=make_optimizer(self.coarse_model.parameters(), **self.cfg.optimizers.coarse_model)),
            dict(optimizer=make_optimizer(self.refine_model.parameters(), **self.cfg.optimizers.generator)),
            dict(optimizer=make_optimizer(discriminator_params, **self.cfg.optimizers.discriminator)),
        ]

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self._is_training_step = True
        return self._do_step(batch, batch_idx, mode='train', optimizer_idx=optimizer_idx)

    def training_step_end(self, batch_parts_outputs):
        full_loss = (batch_parts_outputs['loss'].mean()
                     if torch.is_tensor(batch_parts_outputs['loss'])  # loss is not tensor when no discriminator used
                     else torch.tensor(batch_parts_outputs['loss']).float().requires_grad_(True))
        log_info = {k: v.mean() for k, v in batch_parts_outputs['log_info'].items()}
        self.log_dict(log_info, on_step=True, on_epoch=False)
        return full_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        extra_val_key = None
        if dataloader_idx == 0:
            mode = 'val'
        else:
            mode = 'extra_val'
            extra_val_key = self.extra_val_titles[dataloader_idx - 1]
        self._is_training_step = False
        return self._do_step(batch, batch_idx, mode=mode, extra_val_key=extra_val_key)

    def validation_epoch_end(self, outputs):
        if self.cfg.util_args.predict_only==False:
            if len(outputs)==1:
                outputs=[outputs]
            outputs = [step_out for out_group in outputs for step_out in out_group]
            averaged_logs = average_dicts(step_out['log_info'] for step_out in outputs)
            self.log_dict({k: v.mean() for k, v in averaged_logs.items()})

            pd.set_option('display.max_columns', 500)
            pd.set_option('display.width', 1000)

            # standard validation
            val_evaluator_states = [s['val_evaluator_state'] for s in outputs if 'val_evaluator_state' in s]
            val_evaluator_res = self.val_evaluator.evaluation_end(states=val_evaluator_states)
            val_evaluator_res_df = pd.DataFrame(val_evaluator_res).stack(1).unstack(0)
            val_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
            LOGGER.info(f'Validation metrics after epoch #{self.current_epoch}, '
                        f'total {self.global_step} iterations:\n{val_evaluator_res_df}')

            for k, v in flatten_dict(val_evaluator_res).items():
                self.log(f'val_{k}', v)

            # extra validations
            if self.extra_evaluators:
                for cur_eval_title, cur_evaluator in self.extra_evaluators.items():
                    cur_state_key = f'extra_val_{cur_eval_title}_evaluator_state'
                    cur_states = [s[cur_state_key] for s in outputs if cur_state_key in s]
                    cur_evaluator_res = cur_evaluator.evaluation_end(states=cur_states)
                    cur_evaluator_res_df = pd.DataFrame(cur_evaluator_res).stack(1).unstack(0)
                    cur_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
                    LOGGER.info(f'Extra val {cur_eval_title} metrics after epoch #{self.current_epoch}, '
                                f'total {self.global_step} iterations:\n{cur_evaluator_res_df}')
                    for k, v in flatten_dict(cur_evaluator_res).items():
                        self.log(f'extra_val_{cur_eval_title}_{k}', v)

    def _do_step(self, batch, batch_idx, mode='train', optimizer_idx=None, extra_val_key=None):
        # pdb.set_trace()
        if optimizer_idx == 0:  # step for coarse model
            set_requires_grad(self.coarse_model, True)
            set_requires_grad(self.refine_model, False)
            set_requires_grad(self.discriminator, False)
        elif optimizer_idx == 1:  # step for refine_model
            set_requires_grad(self.coarse_model, False)
            set_requires_grad(self.refine_model, True)
            set_requires_grad(self.discriminator, False)
        elif optimizer_idx == 2:  # step for discriminator
            set_requires_grad(self.coarse_model, False)
            set_requires_grad(self.refine_model, False)
            set_requires_grad(self.discriminator, True)

        
        total_loss = 0
        metrics = {}
        if optimizer_idx==0:   # step for coarse generator
            batch=self(batch, model='coarse')
            total_loss, metrics = self.coarse_generator_loss(batch)

        elif optimizer_idx == 1:  # step for refine generator
            batch=self(batch, model='refine')
            total_loss, metrics = self.refine_generator_loss(batch)
            if mode=='val' or mode =='extra_val':
                visiter=self.cfg.util_args.visualize_each_iters_val
            else:
                visiter=self.cfg.util_args.visualize_each_iters

            if batch_idx % visiter ==0:
                vis_suffix= f'_{mode}'
                if mode == 'extra_val':
                    vis_suffix += f'_{extra_val_key}'
                vis_img=self.visualizer(self.current_epoch, batch_idx, batch, suffix=vis_suffix)
                if type(vis_img)==tuple:
                    for name, vim in zip(['Refined', 'Coarse'], vis_img):
                        self.logger.experiment.add_image(name, vim, dataformats='HWC',global_step=self.global_step)
                else:
                    self.logger.experiment.add_image('ImageGrid', vis_img, dataformats='HWC',global_step=self.global_step)
        elif optimizer_idx == 2:  # step for discriminator
            if self.cfg.losses.adversarial.weight > 0:
                total_loss, metrics = self.discriminator_loss(batch)

        if mode=='val' or mode=='extra_val':
            batch=self(batch, model='coarse')
            batch=self(batch, model='refine')
            if self.cfg.util_args.predict_only==False:
                visiter=self.cfg.util_args.visualize_each_iters_val
                if batch_idx % visiter ==0:
                    vis_suffix= f'_{mode}'
                    if mode == 'extra_val':
                        vis_suffix += f'_{extra_val_key}'
                    vis_img=self.visualizer(self.current_epoch, batch_idx, batch, suffix=vis_suffix)
                    if type(vis_img)==tuple:
                        for name, vim in zip(['Refined', 'Coarse'], vis_img):
                            self.logger.experiment.add_image(name, vim, dataformats='HWC',global_step=self.global_step)
                    else:
                        self.logger.experiment.add_image('ImageGrid', vis_img, dataformats='HWC',global_step=self.global_step)

        if mode=='val' or mode=='extra_val':
            if self.cfg.data.val.get('val_save',None) is not None:
                res_folder = self.cfg.data.val.val_save + os.sep + self.cfg.kind 
                for bf in range(batch['inpainted'].shape[0]):
                    saveRelPath = res_folder + os.sep +batch['pkl_fname'][bf].split(os.sep)[-1]
                    saveRelPath = saveRelPath.replace('.pkl','.png')
                    if not os.path.exists(os.sep.join(saveRelPath.split(os.sep)[:-1])): os.makedirs(os.sep.join(saveRelPath.split(os.sep)[:-1]))
                    tosave = []
                    inim = batch['image'][bf].permute(1,2,0).cpu().numpy()
                    mim = (batch['image']*(1-batch['mask']))[bf].permute(1,2,0).cpu().numpy()
                    pred =batch['inpainted'][bf].permute(1,2,0).cpu().numpy()
                    tosave.append(inim)
                    tosave.append(mim)
                    tosave.append(np.ones((mim.shape[0],50,3)))
                    tosave.append(batch['coarse_out']['coarse_avg'][bf].permute(1,2,0).cpu().numpy())
                    tosave.append(pred)
                    # for key in batch['coarse_out']['coarse_pred'].keys():
                    #     tosave.append( batch['coarse_out']['coarse_pred'][key][bf].permute(1,2,0).cpu().numpy())
                    im2s = np.hstack(tosave)
                    plt.imsave(saveRelPath,im2s)

        metrics_prefix = f'{mode}_'
        if mode == 'extra_val':
            metrics_prefix += f'{extra_val_key}_'
        result = dict(loss=total_loss, log_info=add_prefix_to_keys(metrics, metrics_prefix))
        if self.cfg.util_args.predict_only==False:
            if mode == 'val':
                result['val_evaluator_state'] = self.val_evaluator.process_batch(batch)
            elif mode == 'extra_val':
                result[f'extra_val_{extra_val_key}_evaluator_state'] = self.extra_evaluators[extra_val_key].process_batch(batch)
        return result

    def forward(self):
        raise NotImplementedError()
