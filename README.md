# GraphFill

GraphFill: Deep Image Inpainting using Graphs


Download Model trained on places dataset from [Here]() and place in `./saved_models/c2f_iter/places256/last.ckpt`

Run `python data_gen_pickle.py config=places_256.yaml` first to generate pickles for faster data-loading. (Optional)

Run `./run.sh` or `python run_validation.py config=places_256.yaml` to obtain results for the shared data in `val_results` folder.


#TODO: More Details to be updated.

Note: Present shared model contains weights discriminator, pre-trained model weights for perceptual loss calculation, etc. which are irrelevant in evaluation setting. 
