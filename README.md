# GraphFill

GraphFill: Deep Image Inpainting using Graphs




# Evaluation
We provide various settings in `run.sh`
- `python main.py config=main.yaml "util_args.predict_only=True" "util_args.eval_mode=True" "data=places_256"
` 
    - `predict_only`: Setting this flag to False will calculate losses and evaluate metrics. Set True to track performance on validation performance while training.
    - `eval_mode`: Sets mode for Inference/Training
    - `data`: change config accordingly at `config/data` with appropriate paths to training datasets, Validation datasets. 

- `python data_gen_pickle.py config=main.yaml "data=places_256" "data.train.indir=./Datasets/places365_standard/train"`
    - Pyramidal graph generation can be a bottleneck while loading data. Create pickled data for fast loading (Optional). 
    - If skipping pickling of data it is recommended to increase `num_workers` in dataloader kwargs.

- `python main.py config=main.yaml "util_args.eval_mode=False" "data=places_256" "data.train.pickle_data=True"`
    - Trains GraphFill.


Download trained models from [Here](https://drive.google.com/drive/folders/1Htcct72A2T9C5_p92LyAd226flIgu7qg?usp=sharing) 

Place downloaded models at as pointed by the key `model_load` in `main.yaml` config.
Note that shared model contains weights for discriminator, pre-trained model weights for perceptual loss calculation, etc. which are irrelevant in evaluation setting. 


# Requirements
```
pytorch_lightning==1.9.0
torch==1.13.1
networkx==2.6
torch_geometric==1.5.0
torch_scatter==2.1.1
torch_sparse==0.6.17 
```

Code in this repository is highly inspred from: LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions. Please follow there Instruction to setup `./models` folder, make random masks of sizes `medium,thin,thick`.

# Bib
```
@inproceedings{verma2024graphfill,
  title={GraphFill: Deep Image Inpainting Using Graphs},
  author={Verma, Shashikant and Sharma, Aman and Sheshadri, Roopa and Raman, Shanmuganathan},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={4996--5006},
  year={2024}
}
```