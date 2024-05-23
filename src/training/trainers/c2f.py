import logging
from src.training.trainers.base import BaseInpaintingModule
import pdb
from src.training.losses.feature_matching import feature_matching_loss, masked_l1_loss, masked_l2_loss
from src.utils import add_prefix_to_keys
from omegaconf import OmegaConf
import torch
from torch_sparse import SparseTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
#Coarser to Finer
import torchvision as tv 

class C2FInpaintingTrainingModule(BaseInpaintingModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.image_to_discriminator = cfg.model.image_to_discriminator
        self.concat_mask = cfg.model.concat_mask
        self.kind = cfg.model.generator.kind
        
    def forward(self,batch, model=None):
        img= batch['image']
        mask= batch['mask']

        if self.kind.startswith('c2f'):
            if model=='coarse':
                batch['coarse_out']=self.coarse_model(batch, concat_mask=self.concat_mask)
                coarse_avg=batch['coarse_out']['coarse_avg']
                coarse_avg=tv.transforms.functional.gaussian_blur(coarse_avg, (5,5))
                masked_img= img * (1-mask)
                coarse_pred = coarse_avg* mask
                coarse_pred= coarse_pred.detach()   # Test if detach or not
                refine_in = masked_img + coarse_pred
                if self.concat_mask:
                    refine_in = torch.cat([refine_in, mask], dim=1)
                batch['refine_in']=refine_in
            if model=='refine':
                batch['refine_out'] = self.refine_model(batch['refine_in'])
                batch['predicted_image']= batch['refine_out']
                batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']
        else:
            raise NotImplementedError()
        return batch
    
    def coarse_generator_loss(self,batch):
        f_metrics=dict()
        total_loss = 0
        img_all=[]
        predicted_img_all=[]
        original_mask_all=[]
        for key in batch['coarse_out']['coarse_pred'].keys():
            img= batch['coarse_out']['coarse_gt'][key]
            predicted_img = batch['coarse_out']['coarse_pred'][key]
            original_mask = batch['coarse_out']['coarse_mask'][key]   # 1-mask here
            img_all.append(img)
            predicted_img_all.append(predicted_img)
            original_mask_all.append(original_mask)
            tls,metrics = self.spix_genloss(batch,key)
            total_loss+=tls
            f_metrics.update(add_prefix_to_keys(metrics, 'coarse_{}'.format(key)))
            tlc,metrics = self.losses_generator(img,predicted_img, original_mask, adversarial=False)
            total_loss+=tlc
            f_metrics.update(add_prefix_to_keys(metrics, 'coarse_{}_'.format(key)))
        f_metrics['total_loss']=total_loss
        return total_loss, f_metrics
    
    def refine_generator_loss(self,batch):
        f_metrics=dict()
        total_loss = 0
        img= batch['image']
        predicted_img = batch['refine_out'] 
        original_mask = batch['mask']
        tlf, metrics = self.losses_generator(img,predicted_img, original_mask, adversarial=True)
        total_loss+=tlf
        f_metrics.update(add_prefix_to_keys(metrics, 'refine_'))
        f_metrics['total_loss']=total_loss
        return total_loss, f_metrics


    def spix_genloss(self, batch, key ):
        predspix,gtspix, maskspix=batch['coarse_out']['spixout'][key],batch['spix_info'][key].gtspix, batch['spix_info'][key].spix_mask
        unnfs, gtunnfs= predspix[(maskspix==1).squeeze()], gtspix[(maskspix==1).squeeze()]
        unn_loss=F.mse_loss(unnfs,gtunnfs)* self.cfg.losses.spix.unn

        knnfs, gtknnfs= predspix[(maskspix==0).squeeze()], gtspix[(maskspix==0).squeeze()]
        knn_loss=F.mse_loss(knnfs,gtknnfs)* self.cfg.losses.spix.knn

        total_node_loss= unn_loss+knn_loss
        metrics = dict(spix_mse=total_node_loss)
        return total_node_loss, metrics

    def losses_generator(self, img, predicted_img, original_mask, adversarial=True):
        metrics={}
        l1_value = masked_l1_loss(predicted_img, img, original_mask,self.cfg.losses.l1.weight_known,self.cfg.losses.l1.weight_missing)
        total_loss = l1_value
        metrics['l1_value'] = l1_value
        
        # mse_value = masked_l2_loss(predicted_img, img, original_mask,
        #                           self.cfg.losses.l2.weight_known,
        #                           self.cfg.losses.l2.weight_missing)
        # total_loss = mse_value
        # metrics = dict( mse_value=mse_value)
        # if self.cfg.losses.perceptual.weight > 0:
        #     pl_value = self.loss_pl(predicted_img, img, mask=original_mask).sum() * self.cfg.losses.perceptual.weight
        #     total_loss = total_loss + pl_value
        #     metrics['gen_pl'] = pl_value
        
        if adversarial:
            mask_for_discr= original_mask
            self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                    generator=self.refine_model, discriminator=self.discriminator)
            discr_real_pred, discr_real_features = self.discriminator(img)
            discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
            adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(real_batch=img,
                                                                            fake_batch=predicted_img,
                                                                            discr_real_pred=discr_real_pred,
                                                                            discr_fake_pred=discr_fake_pred,
                                                                            mask=mask_for_discr)
            total_loss = total_loss + adv_gen_loss
            metrics['gen_adv'] = adv_gen_loss
            metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

            if self.cfg.losses.feature_matching.weight > 0:
                need_mask_in_fm = OmegaConf.to_container(self.cfg.losses.feature_matching).get('pass_mask', False)
                mask_for_fm = original_mask if need_mask_in_fm else None
                fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                                mask=mask_for_fm) * self.cfg.losses.feature_matching.weight
                total_loss = total_loss + fm_value
                metrics['gen_fm'] = fm_value

        if self.cfg.losses.resnet_pl.weight> 0:
            # lpips_value=self.loss_lpips(predicted_img.clone(), img.clone()).mean() * self.cfg.losses.lpips.weight
            # total_loss = total_loss + lpips_value
            # metrics['lpips_value'] = lpips_value
            resnet_pl_value = self.loss_resnet_pl(predicted_img, img) 
            total_loss = total_loss + resnet_pl_value
            metrics['gen_resnet_pl'] = resnet_pl_value

        
        return total_loss, metrics

    def discriminator_loss(self, batch):        
        total_loss = 0
        f_metrics = {}

        # Uncomment if only training coarse graphFill
        # if 'coarse_out' in batch.keys():
        #     img_all=[]
        #     predicted_img_all=[]
        #     original_mask_all=[]
        #     for key in batch['coarse_out']['coarse_pred'].keys():
        #         img= batch['coarse_out']['coarse_gt'][key]
        #         predicted_img = batch['coarse_out']['coarse_pred'][key]
        #         original_mask = batch['coarse_out']['coarse_mask'][key]   
        #         img_all.append(img)
        #         predicted_img_all.append(predicted_img)
        #         original_mask_all.append(original_mask)

        #     img= torch.cat(img_all)
        #     predicted_img= torch.cat(predicted_img_all)
        #     original_mask= torch.cat(original_mask_all).float()
        #     self.adversarial_loss.pre_discriminator_step(real_batch=img, fake_batch=predicted_img,generator=self.coarse_model, discriminator=self.discriminator)
        #     discr_real_pred, discr_real_features = self.discriminator(img)
        #     discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        #     adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(real_batch=img,fake_batch=predicted_img,discr_real_pred=discr_real_pred,discr_fake_pred=discr_fake_pred, mask=original_mask)
        #     total_loss = total_loss + adv_discr_loss
        #     f_metrics.update(add_prefix_to_keys(adv_metrics, 'coarse_{}'.format(key)))

        if 'refine_out' in batch.keys():
            img= batch['image']
            predicted_img = batch['refine_out'].detach()
            original_mask = batch['mask']
            self.adversarial_loss.pre_discriminator_step(real_batch=img, fake_batch=predicted_img, generator=self.refine_model, discriminator=self.discriminator)
            discr_real_pred, discr_real_features = self.discriminator(img)
            discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
            adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(real_batch=img,fake_batch=predicted_img,discr_real_pred=discr_real_pred,discr_fake_pred=discr_fake_pred, mask=original_mask)
            total_loss = total_loss + adv_discr_loss
            f_metrics.update(add_prefix_to_keys(adv_metrics, 'disc_'))
        return total_loss, f_metrics

