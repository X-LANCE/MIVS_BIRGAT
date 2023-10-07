# A BiRGAT Model for Multi-intent Spoken Language Understanding with Hierarchical Semantic Frames

This repository contains the MIVS dataset and codes to train our BiRGAT model in the paper **A BiRGAT Model for Multi-intent Spoken Language Understanding with Hierarchical Semantic Frames**. 

## Requirements

The required python packages is listed in "requirements.txt". You can install them by

```
pip install -r requirements.txt
```

or

```
conda install --file requirements.txt
```


## Data Description

Please unzip the file in the `data` folder, i.e., `data/aispeech.zip` and `data/topv2.zip`, which correspond to our proposed MIVS dataset and the converted TOPV2 dataset, respectively.


## Training and Evaluation

### Pre-processing

First, we need to do preprocessing and aggregate all ontology information, then build the relationship matrix between different ontologies:

```
./run/run_preprocessing.sh
```

### Training

By running different scripts, we can train different models and evaluate on the validation and test sets during training.  `swv`, `small`, and `base` indicate using only static word vectors from pre-trained models, using small-series pre-trained models, and using base-series pre-trained models, respectively.

```
./run/run_train_and_eval_swv.sh
./run/run_train_and_eval_small.sh
./run/run_train_and_eval_base.sh
```

### Evaluation
Evaluate the model on the validation and test sets: (Specify the directory of the saved model)

```
./run/run_eval.sh [dir_to_saved_model]
```

### Some Parameters of `scripts/train_and_eval.py`
- `--files`: Used to specify input file names (excluding the .json extension) to be recursively searched for and read from the data directory, e.g., `--files 地图 天气 地图_cross_天气` means reading three files from the respective data directories: `地图.json`, `天气.json`, and `地图_cross_天气.json`. Additionally, there are special values such as `all` (read all data), `single_domain` (read all single-domain data), and `cross_domain` (read all cross-domain data) (see `process/dataset_utils.py` for more details).
- `--src_files` and `--tgt_files`: Similar to `--files`, but with higher priority. `src` represents data files used for training, and `tgt` represents files used for testing. Both can be used for transfer learning experiments. If neither parameter is specified, it defaults to using the data files specified by `--files`.
- `--domains`: Used to specify input domains, e.g., `--domains 地图 天气` means that all read data will use the two domains `地图` and `天气`, even if a data sample only involves the `地图` domain. If this parameter is not specified (default is `None`), the default domain for each sample will be automatically used (which could be one or two domains). If this parameter is specified, please ensure that it includes all the domains of the data being read (the program also checks this when reading data files).
- `--init_method [swv|plm]`: Indicates the initialization method for embedding vectors, either initializing from static word vectors of pre-trained models (`swv`) or using the complete pre-trained model (`plm`).
- `--ontology_encoding`: Specifies whether to use ontology encoding. If not used, all ontology items are initialized directly from the embedding matrix. Otherwise, semantic encoding initialization is done using text descriptions, such as the intent `播放音乐`. Furthermore, it can be combined with the `--use_value` parameter to enhance the encoding of semantic slots with additional sampled slot values.