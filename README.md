# Attention Normalization Impacts Cardinality Generalization in Slot Attention
This repository contains training and model code for the publication ([https://openreview.net/forum?id=llQXLfbGOq](https://openreview.net/forum?id=llQXLfbGOq)).

Krimmel, Markus and Achterhold, Jan and Stueckler, Joerg: \
**Attention Normalization Impacts Cardinality Generalization in Slot Attention**\
Accepted for Transactions on Machine Learning Research (TMLR), 2024


If you use the code, data or models provided in this repository for your research, please cite our paper as:
```
@article{
    krimmel2024_sanormalization,
    title={Attention Normalization Impacts Cardinality Generalization in Slot Attention},
    author={Markus Krimmel and Jan Achterhold and Joerg Stueckler},
    journal={Accepted for Transactions on Machine Learning Research (TMLR)},
    year={2024},
    url={https://openreview.net/forum?id=llQXLfbGOq}
}
```

## Setup and General Project Structure
Install the project's dependencies:
```
conda env create -f environment.yml
```

This repository is organized into four submodules. Three submodules implement training on various datasets, while the implementation of the Slot Attention module resides in a shared submodule.
To run computational experiments, you will need to install various datasets. To this end, create a directory called `data` at the root of this directory structure (if necessary, a symbolic link to bulk storage).

```
├── sa_generalization
│   ├── slot_attention
│   │   └── *This contains the implementation of the Slot Attention module, used by the other components.*
│   ├── property_prediction
│   │   └── *Training and model code for property/set prediction on the CLEVR dataset.*
│   ├── object_discovery
│   │   └── *Training and model code for object discovery on CLEVR (Tetrominoes also available).*
│   └── dinosaur
│       └── *Training and model code for object discovery on MOVi dataset.*
└── data
    └── *Create this directory to install datasets.*
```

Hydra is used for configuring the computational experiments, the `train.py` files are entry points for training.
The code is adapted to run on shared cluster resources and will therefore periodically interrupt training with exit code 124. Either configure your cluster to reschedule training jobs upon this exit code, or configure `leap_timelimit_h` sufficiently high to avoid interruptions.

## Citation and License
If you make use of this code, please cite the following:

```
@article{DBLP:journals/corr/abs-2407-04170,
  author       = {Markus Krimmel and
                  Jan Achterhold and
                  Joerg Stueckler},
  title        = {Attention Normalization Impacts Cardinality Generalization in Slot
                  Attention},
  journal      = {CoRR},
  volume       = {abs/2407.04170},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2407.04170},
  doi          = {10.48550/ARXIV.2407.04170},
  eprinttype    = {arXiv},
  eprint       = {2407.04170},
  timestamp    = {Mon, 12 Aug 2024 20:53:40 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2407-04170.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

Some parts of this codebase are derived from other projects and therefore fall under specific licenses. Please refer to `LICENSE.md` and headers of the individual code files for details.

## Property Prediction on CLEVR
### Data Preparation
1. The datasets necessary for property/set prediction are installed into `data/property_prediction`. Create this directory if not present already.
2. Run `python -m sa_generalization.property_prediction.data.install_clevr` to download and pre-process the CLEVR dataset for property prediction

### Training
The entry point for training is `sa_generalization/property_prediction/train.py`.
To perform a training run with the weighted mean variant (baseline), use the following command:
```
python -m sa_generalization.property_prediction.train hydra.run.dir=<OUTPUT DIR> wandb.name=<WANDB NAME> device=cuda
```
You may configure the normalization variant by appending `sa_kwargs.update_normalization` to the hydra config.
Specifically you can add the following to the command above:
- `+sa_kwargs.update_normalization=layer_norm`: for layer normalization
- `+sa_kwargs.update_normalization=scalar_batch_norm_single`: for batch normalization
- `+sa_kwargs.update_normalization=constant`: for weighted sum normalization

The argument `leap_timelimit_h` controls periodic restarts, as explained above.

## Object Discovery on CLEVR
### Data Preparation
1. The datasets for object discovery on CLEVR are installed into `data/object_discovery`. Create this directory if not present already.
2. Run `python -m sa_generalization.object_discovery.data.get_clevr_data`.

### Training
The entry point for training is `sa_generalization/object_discovery/train.py`.
To perform a training run with the weighted mean variant (baseline), use the following command:
```
python -m sa_generalization.object_discovery.train hydra.run.dir=<OUTPUT DIR>
```
For configuring experiments, note:
- As in property prediction, you may choose alternative attention normalization variants. However, here you must set `+autoencoder_kwargs.sa_kwargs.update_normalization=<VARIANT>`.
- You may adjust the maximum number of objects seen during training by setting `dataset.max_objects=<MAX OBJECTS>`.
- You may change the number of slots used during training from the default (7) by setting `+autoencoder_kwargs.num_slots=<NUM SLOTS>`. 
- As the code is intended to be run in a cluster environment, by default the datasets will be copied into local node storage (`/tmp`) at the start of training. Set `copy_to_tmp=false` to avoid this.
- You may set `leap_timelimit_h` as discussed above.

## Object Discovery on MOVi
### Data Preparation
1. The datasets for object discovery on MOVi are installed into `data/dinosaur`. Create this directory if not present already.
2. Run `python -m sa_generalization.dinosaur.data` to install MOVi-C. Note that this will pre-compute ViT features and you will therefore need a GPU. The resulting file is approximately 500GB in size.

### Training
The entry point for training is `sa_generalization/dinosaur/train.py`.
To perform a training run with the weighted mean variant (baseline), use the following command:
```
python -m sa_generalization.dinosaur.train hydra.run.dir=<OUTPUT DIR>
```
For configuring experiments, note:
- You may choose alternative attention normalizations via `+sa_kwargs.update_normalization=<VARIANT>`
- You may adjust the maximum number of objects seen during training by setting `max_objects=<MAX OBJECTS>`
- You may adjust the slot count used during training by setting `n_slots=<NUM SLOTS>`
- Set `leap_time_h` to adjust the restarting behavior

## License
See file [LICENSE.md](LICENSE.md) in the root directory of this repository for license information. 
