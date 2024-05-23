import logging
from src.training.modules.ffc import FFCResNetGenerator
from src.training.modules.pix2pixhd import GlobalGenerator, MultiDilatedGlobalGenerator, \
    NLayerDiscriminator, MultidilatedNLayerDiscriminator
from src.training.modules.tmp import TmpModel
from src.training.modules.c2f import CoarseNet
from src.training.modules.c2f_iter import CoarseNetIter

def make_generator(kind, **kwargs):
    logging.info(f'Make generator {kind}')
    if kind == 'tmp':
        return TmpModel()
    if kind == 'pix2pixhd_multidilated':
        return MultiDilatedGlobalGenerator(**kwargs)
    if kind == 'ffc_resnet':
        return FFCResNetGenerator(**kwargs)
    # if kind.startswith('c2f_'):
    #     return C2FModel(kind,kwargs) 
    if kind=='gcn':
        logging.info(f'Make generator CoarseNet')
        return CoarseNet(kind, kwargs)
    if kind=='gcn_iter':
        logging.info(f'Make generator CoarseNetIter')
        return CoarseNetIter(kind, kwargs)
    

    raise ValueError(f'Unknown generator kind {kind}')


def make_discriminator(kind, **kwargs):
    logging.info(f'Make discriminator {kind}')
    if kind == 'pix2pixhd_nlayer_multidilated':
        return MultidilatedNLayerDiscriminator(**kwargs)

    if kind == 'pix2pixhd_nlayer':
        return NLayerDiscriminator(**kwargs)

    raise ValueError(f'Unknown discriminator kind {kind}')
