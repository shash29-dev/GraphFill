import random
from torch.utils.data import Dataset
import glob
import os 
import cv2
import numpy as np
import pdb
import imutils
import PIL.Image as Image
import torch
from src.training.data.image2graph import Image2Graph
import pickle

def load_image(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img

class InpaintingTrainDataset(Dataset):
    def __init__(self, indir, kind, mask_generator, transform, pickle_data=False):
        if not pickle_data:
            if kind=='davis':
                split_txt=indir+'/ImageSets/2017/train.txt'
                with open(split_txt, 'r') as f: 
                    fols=f.readlines()
                fols=[x.strip() for x in fols]
                self.in_files=[]
                for fol in fols:
                    self.in_files += list(glob.glob(os.path.join(indir,'JPEGImages', '*' ,fol, '*.jpg'), recursive=True))
            elif kind=='celebA':
                split_txt=indir+'/Eval/list_eval_partition.txt'
                with open(split_txt, 'r') as f: 
                    files=f.readlines()
                
                splittag=0  # for train
                self.in_files = [indir+'/Img/img_align_celeba/'+x.strip().split()[0] for x in files if x.strip().split()[1]==str(splittag) ]
            else:
                self.in_files = list(glob.glob(os.path.join(indir, '**' , '*.jpg'), recursive=True))
                # self.in_files= [self.in_files[19]]*20000
        else:
            self.in_files = list(glob.glob(os.path.join(indir, '**' , '*.pkl'), recursive=True))

        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = imutils.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),height=256)
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        mask = self.mask_generator(img, iter_i=self.iter_i)
        self.iter_i += 1
        return dict(image=img,
                    mask=mask)

class InpaintingValDataset(Dataset):
    def __init__(self, datadir, img_suffix='.jpg',pickle_data=False):
        self.datadir = datadir
        if not pickle_data:
            self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask.png'), recursive=True)))
            self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        else:
            self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**' , '*.pkl'), recursive=True)))

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        image = load_image(self.img_filenames[i], mode='RGB')
        mask = load_image(self.mask_filenames[i], mode='L')
        result = dict(image=image, mask=mask[None, ...])
        return result
    
class InpaintingTrainSpixDataset(InpaintingTrainDataset):
    def __init__(self, indir, kind, mask_generator, transform, cfgspix,outsize=256, pickle_data=False):
        super().__init__(indir, kind, mask_generator, transform, pickle_data)
        self.outsize=outsize
        if cfgspix.generate==True:
            self.spixgen= Image2Graph(**cfgspix)
        else:
            self.spixgen=None
        self.pickle_data = pickle_data

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        if not self.pickle_data:
            path = self.in_files[item]
            img = cv2.imread(path)
            img = imutils.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),height=self.outsize)
            img = self.transform(image=img)['image']
            img = np.transpose(img, (2, 0, 1))
            # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
            mask = self.mask_generator(img, iter_i=self.iter_i)
            self.iter_i += 1
            if self.spixgen is not None:
                spix_info, seg_dict=self.spixgen.get_data(img,mask)
                return dict(
                        image=torch.from_numpy(img),
                        mask=torch.from_numpy(mask), 
                        spix_info=spix_info,
                        seg= seg_dict,
                        )
            else:
                return dict(image=img,
                            mask=mask)
        else:
            path = self.in_files[item]
            try:
                with open(path,'rb') as f: data=pickle.load(f)
            except Exception as e:
                print(e)
                print('****Deleting {}:\t empty****'.format(path))
                if os.path.exists(path) : 
                    os.remove(path)
                else:
                    print('File Not found..')
                with open(self.in_files[0],'rb') as f: data=pickle.load(f)
            return data
            
class InpaintingValSpixDataset(InpaintingValDataset):
    def __init__(self, datadir, img_suffix, cfgspix, pickle_data=False):
        super().__init__(datadir,img_suffix,pickle_data)
        if cfgspix.generate==True:
            self.spixgen= Image2Graph(**cfgspix)
        else:
            self.spixgen=None
        self.pickle_data=pickle_data

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        if not self.pickle_data:
            img = load_image(self.img_filenames[i], mode='RGB')
            mask = load_image(self.mask_filenames[i], mode='L')
            mask=mask[None,:,:]
            # img = cv2.imread(self.img_filenames[i])
            # mask = cv2.imread(self.mask_filenames[i], cv2.IMREAD_GRAYSCALE)
            if self.spixgen is not None:
                spix_info, seg_dict=self.spixgen.get_data(img,mask)
                return dict(
                        image=torch.from_numpy(img),
                        mask=torch.from_numpy(mask), 
                        spix_info=spix_info,
                        seg= seg_dict,
                        pkl_fname = self.img_filenames[i]
                        )
            else:
                return dict(image=img,
                            mask=mask)
        else:
            path = self.mask_filenames[i]
            try:
                with open(path,'rb') as f: data=pickle.load(f)
                data['pkl_fname']= path
            except Exception as e:
                print(e)
                print('****Deleting {}:\t empty****'.format(path))
                if os.path.exists(path) : 
                    os.remove(path)
                else:
                    print('File Not found..')
                with open(self.mask_filenames[0],'rb') as f: data=pickle.load(f)
                data['pkl_fname']= self.mask_filenames[0]
            return data
