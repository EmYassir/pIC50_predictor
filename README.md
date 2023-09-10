# pIC50_predictor
Epidermal Growth Factor Receptor (EGFR) kinase detection


## About
This folder contains the original code used to build a model selection and inference pipeline that can generate the most accurate predictor for pIC50. It uses the [EGFR_compounds_lipinski dataset](https://raw.githubusercontent.com/volkamerlab/teachopencadd/master/teachopencadd/talktorials/T002_compound_adme/data/EGFR_compounds_lipinski.csv) for training and evaluation.

## Setup
There are few specific dependencies to install before launching a distillation, you can install them with the command `pip install -r requirements.txt`.

## Dataset pre-processing
### Data cleaning
The dataset yields tabular data which consists in 10 columns. We get rid of the following columns for the sake of training:
- molecule_chembl_id: Ids of the molecules.
- IC50: related to `pIC50`, which we are trying to predict.
- units: useless in terms of training.
- ro5_fulfilled: same value (True) for all samples.

We hence kept the following features for the rest of the experiment:
- smiles: text data.
- pIC50: continuous data.
- molecular_weight: continuous data.
- n_hba: categorical data.
- n_hbd: categorical data.
- logp: continuous data.

### Data augmentation
To better make use of features and allow for performing classification tasks, we augment the data as follows:
- smiles_fp: embedding for `smiles` data. Calculated using functions provided by []().
- labels: True when `pIC50` > 8, false otherwise.

### Data visualization
We vizualise the data by plotting box-plots to better assess the spread of values. Below is the figure showing categorical data `n_hba` and `n_hbd`:

The molecular weight values are far greater than those of the remaining continuous values, so we plot them separately:


Finally, the box plots of `logp` and `pIC50`:

As we can see below, the values of `pIC50` are concentrated around the critical threshold (8) that we are using to classify our samples: 

Going with regression can be tricky, as we can achieve good regression performance and yet get a bad classification score: the regressor can approach the threshold (8.) from both sides (from under or above), which means it can classify correctly or incorrectly depending on which side of the threshold the prediction falls. Therefore, we are moving forward with **treating the problem as a classification task**.


## Models
We run experiments on one baseline and two models. The baseline we chose is the random forest ([Sklearn]() implementation) as suggested by this [article](). The two other models we chose are [CatBoostClassifier]() and [ChemBertA](). For the latter, we are using [simpletransformers implementation]().

## How to use the code

### Files organization
In the following, we explain how to finetune/pretrain/distil GPT2 architectures. Connecting to the huggingface repo is necessary while downloading the tokenizer, but this is not possible with due to the company's proxy. Users can follow these steps to deactivate the SSL verification from the environment side:
1- If you are using a `conda` environment, go to `~/.conda/envs/<NAMEOFYOURENVIRONEMNT>/lib/python3.8/site-packages/requests/sessions.py`.
2- Search for `self.verify = True`.
3- Change it to `self.verify = False`.


### The configuration file




**Note**: in order to reproduce our best model, please use the intial weights at `/media/data/yassir/truncated_models/gpt2-alt/`.

### Training and evaluation 
Evaluation scripts below provide examples to be inspired of:
- evaluate.sh
- evaluate_deepspeed.sh

## Results and discussion
The table below shows the results we obtained.
| Model | Accuracy | Sensitivity | Specificity | F1 |
|  :---: |  :---: |  :---:  |  :---: |  :---:  |  
| Random Forest (baseline) | 0.86 | 0.57 | 0.93 | 0.61 |
| CatBoostClassifier (all features) | 0.85 | 0.43 | **0.95** | 0.53 |
| CatBoostClassifier (smiles only) | 0.86 | 0.49 | **0.95** | 0.56 |
| ChemBertA (`seyonec/PubChem10M_SMILES_BPE_396_250`) | **0.87** | **0.60** | 0.93 | **0.63** |

As shown in the results, the baseline model is pretty strong, although it only uses smiles embeddings. The CatBoostClassifier performs better when fed with smiles embedding only, which might question the usefulness of other features (provided that CatBoost models handle categorical, continous and embedding data). However, since we did not explore this problem in depth, nor did we perform hyper-parameter finetuning, we cannot draw a conclusion yet. Despite only using the smiles features, the ChemBertA model outperforms all others, confirming the superiority of large pre-trained language models.

## Next steps
As next steps, we plan to propose an architecture that is able to both leverage pre-trained models (ChemBertA) and make use of other features (continuous and categorical). For that, we need to embed categorical data, rescale continuous variables and make use of dense layers to conveniently process the data. The figure below describes the full proposed architecture:

![archi](https://github.com/EmYassir/pIC50_predictor/assets/20977650/20bbe5b5-76cd-4104-bcbe-9635afcb73e3)

We also plan to put some effort in finetuning hyper-parameters, as the performance gains could be non-negligible. 

### E. Citation

```
@misc{https://doi.org/10.48550/arxiv.2110.08460,
  doi = {10.48550/ARXIV.2110.08460},
  url = {https://arxiv.org/abs/2110.08460},
  author = {Li, Tianda and Mesbahi, Yassir El and Kobyzev, Ivan and Rashid, Ahmad and Mahmud, Atif and Anchuri, Nithin and Hajimolahoseini, Habib and Liu, Yang and Rezagholizadeh, Mehdi},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {A Short Study on Compressing Decoder-Based Language Models},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

@misc{Simple transformers,
  url = {https://github.com/ThilinaRajapakse/simpletransformers/tree/master},
}

```
