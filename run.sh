

python main.py config=main.yaml "util_args.predict_only=True" "util_args.eval_mode=True" "data=celeb_256"
# python main.py config=main.yaml "util_args.predict_only=True" "util_args.eval_mode=True" "data=places_256"


# ### Evaluate metrics also
# python main.py config=main.yaml "util_args.predict_only=False" "util_args.eval_mode=True" "data=celeb_256"
# python main.py config=main.yaml "util_args.predict_only=False" "util_args.eval_mode=True" "data=places_256"

## Train Network
# python data_gen_pickle.py config=main.yaml "data=places_256" "data.train.indir=./Datasets/places365_standard/train"
# python main.py config=main.yaml "util_args.eval_mode=False" "data=places_256" "data.train.pickle_data=True"
# python main.py config=main.yaml "util_args.eval_mode=False" "data=places_256" "data.train.pickle_data=False" "data.train.indir=./Datasets/places365_standard/train"
