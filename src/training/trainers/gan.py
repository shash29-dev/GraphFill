import logging
from src.training.trainers.base import BaseInpaintingModule
import pdb
from src.training.losses.feature_matching import feature_matching_loss, masked_l1_loss
from src.utils import add_prefix_to_keys
from omegaconf import OmegaConf
import torch

#Coarser to Finer
class InpaintingTrainingModule(BaseInpaintingModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.image_to_discriminator = cfg.model.image_to_discriminator
        self.concat_mask = cfg.model.concat_mask
        
    def forward(self,batch):
        img= batch['image']
        mask= batch['mask']

        masked_img= img * (1-mask)
        if self.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)
        batch['predicted_image']=self.generator(masked_img)
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']
        return batch

    def generator_loss(self,batch):
        img= batch['image']
        predicted_img = batch[self.image_to_discriminator]
        original_mask = batch['mask']
        l1_value = masked_l1_loss(predicted_img, img, original_mask,
                                  self.cfg.losses.l1.weight_known,
                                  self.cfg.losses.l1.weight_missing)
        total_loss = l1_value
        metrics = dict(gen_l1=l1_value)
        if self.cfg.losses.perceptual.weight > 0:
            pl_value = self.loss_pl(predicted_img, img, mask=original_mask).sum() * self.config.losses.perceptual.weight
            total_loss = total_loss + pl_value
            metrics['gen_pl'] = pl_value
        
        
        mask_for_discr= original_mask
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
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
            resnet_pl_value = self.loss_resnet_pl(predicted_img, img)
            total_loss = total_loss + resnet_pl_value
            metrics['gen_resnet_pl'] = resnet_pl_value
        return total_loss, metrics

    def discriminator_loss(self, batch):
        total_loss = 0
        metrics = {}

        predicted_img = batch[self.image_to_discriminator].detach()
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=predicted_img,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['image'])
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(real_batch=batch['image'],
                                                                               fake_batch=predicted_img,
                                                                               discr_real_pred=discr_real_pred,
                                                                               discr_fake_pred=discr_fake_pred,
                                                                               mask=batch['mask'])
        total_loss = total_loss + adv_discr_loss
        metrics['discr_adv'] = adv_discr_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))


        if batch.get('use_fake_fakes', False):
            fake_fakes = batch['fake_fakes']
            self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=fake_fakes,
                                                         generator=self.generator, discriminator=self.discriminator)
            discr_fake_fakes_pred, _ = self.discriminator(fake_fakes)
            fake_fakes_adv_discr_loss, fake_fakes_adv_metrics = self.adversarial_loss.discriminator_loss(
                real_batch=batch['image'],
                fake_batch=fake_fakes,
                discr_real_pred=discr_real_pred,
                discr_fake_pred=discr_fake_fakes_pred,
                mask=batch['mask']
            )
            total_loss = total_loss + fake_fakes_adv_discr_loss
            metrics['discr_adv_fake_fakes'] = fake_fakes_adv_discr_loss
            metrics.update(add_prefix_to_keys(fake_fakes_adv_metrics, 'adv_'))

        return total_loss, metrics

