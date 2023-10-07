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

### MIVS

The multi-intent MIVS dataset contains 5 different domains, namely `map`, `weather`, `phone`, `in-vehicle control` and `music`. The entire dataset can be split into two parts: single-domain and multi-domain. Single-domain examples contain both single-intent and multi-intent cases, which are collected and manually annotated from a realistic industrial in-vehicle environment. For cross-domain samples, we automatically synthesize them following MixATIS. Concretely, we extract two utterances from two different domains and concatenate them by conjunction words such as `and`.
<!-- The output tree can be serialized as a token sequence by inserting sentinel tokens such as brackets for clustering. This serialized format is exactly the output of our model. Annotated examples from both single-domain and multi-domain are given in the following table. -->

<p align="center">
  <img src="https://raw.githubusercontent.com/importpandas/MIVS_BIRGAT/main/assets/mivs_example.png" alt="MIVS examples" width="80%"/>
</p>

MIVS dataset contains $105,240$ samples in total. It can be further split by two parts: single-domain and multi-domain. The statistics of single-domain examples~($5$ domains) are listed in the below table. 

Domains | # Intents | # Slots | # Train | # Valid | # Test
-------| --- | --- |--- | --- | ---
in-vehicle control | 3 | 18 | 16000 | 2000 | 2000
map | 6 | 16 | 4000 | 500 | 500
music | 7 | 27 | 4000 | 500 | 500
weather | 18 | 24 | 4000 | 500 | 500
phone | 8 | 13 | 3249 | 500 | 491
**total** | 42 | 98 | 31249 | 4000 | 3991

For the multi-domain partition, we synthesize examples for all $C_5^2=10$ different domain combinations. The average domain, intent and slot mentioned per utterance over the entire dataset is provided in the following table.

Domains | avg. domain | avg. intent | avg. slot
-------| --- | --- |--- 
TOPv2 | 1 | 1.1 | 1.8
MIVS | 1.6 | 2.3 | 6.3

### Privacy and Anonymization

Preventing leakage of personal identifiable information (PII) is one major concern in practical deployment. To handle privacy constraints, we carry out a slot-based manual check followed by automatic substitution. For example, we discover that the sensitive information mainly lies in two slots "*contact person*" and  "*phone number*" in domain `phone`. For anonymity consideration, we replace the person name with one placeholder from a pool of popular names. Similarly, the phone number is also substituted with a randomly sampled digit sequence of the equivalent length.

### Data Format Conversion of TOPv2
The original outputs in TOPv2 is organized as a token sequence, including raw question words and ontology items, exemplified in following table. We convert the target semantic representation into the same format as MIVS based on the following criterion:
-  Names of intents and slots are simplified into meaningful lowercased English words. The prefix `IN:` or `SL:` is removed, and the underscore `_` is replaced with a whitespace.
- Only question words bounded with slots are preserved and treated as slot values, while words wrapped by intents are ignored. Notice that in TOPv2, intents can also be children of slots~(the second example in the following table). In this case, the word `tomorrows` in the scope of this child intent `IN:GET_TIME` is part of the slot value `tomorrows alarms` for the outer slot `SL:ALARM_NAME`.
- The nested intent and its children slots are extracted and treated as the right sibling of the current intent. For example, in the second case of the below table, the inner intent `IN:GET_TIME` and its child slot-value pair `SL:DATE_TIME=tomorrows` are treated as a separate sub-tree under the domain `alarm`.
- In total, $23$ data samples are removed which contain slots without slot values.

<p align="center">
  <img src="https://raw.githubusercontent.com/importpandas/MIVS_BIRGAT/main/assets/topv2_example.png" alt="TOPv2 examples" width="80%"/>
</p>



## Training and Evaluation

### Data Preparation

Please unzip the file in the `data` folder, i.e., `data/aispeech.zip` and `data/topv2.zip`, which correspond to our proposed MIVS dataset and the converted TOPV2 dataset, respectively.

After unzipping the data files, you will get the following folder structure:
- train:
    - cross_data: Each file in this directory contains samples from two domains
    - one_domain_data: Each file in this directory contains samples from a single domain
    - null_data: Contains only one file, null.json, with empty parsing results
    - cross_data_multi: An enhanced version of cross_data, where "车载控制_multi" is more challenging than "车载控制," and "车载控制_multi_5_10" is too complex and is not considered for now
- valid: Same structure as the train directory
- test: Same structure as the train directory
- ontology.json: 

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