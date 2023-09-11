# pIC50_predictor
Epidermal Growth Factor Receptor (EGFR) kinase detection

## About
This folder contains the original code used to build a model selection and inference pipeline that can generate the most accurate predictor for pIC50. It uses the [EGFR_compounds_lipinski dataset](https://raw.githubusercontent.com/volkamerlab/teachopencadd/master/teachopencadd/talktorials/T002_compound_adme/data/EGFR_compounds_lipinski.csv) for training and evaluation.

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
- smiles_fp: embedding for `smiles` data. Calculated using functions provided by [this tutorial](https://projects.volkamerlab.org/teachopencadd/talktorials/T007_compound_activity_machine_learning.html).
- labels: True when `pIC50` > 8, false otherwise.

### Data visualization
We vizualise the data by plotting box-plots to better assess the spread of values. Below is the figure showing categorical data `n_hba` and `n_hbd`:

![n_hba_n_hbd](https://github.com/EmYassir/pIC50_predictor/assets/20977650/0f8d8b7a-0637-4c70-aec5-ebea4f0bed6c)

Molecular weights' values are far greater than those of the remaining continuous values, so we plot them separately:

![molecular_weight](https://github.com/EmYassir/pIC50_predictor/assets/20977650/ec49262e-dc64-458b-b6ef-04256e7131c1)

Finally, the box plots of `logp` and `pIC50`:

![logp_pIC50](https://github.com/EmYassir/pIC50_predictor/assets/20977650/f69f6b79-84e8-482b-8e52-deb58f54d205)

As we can see below, the values of `pIC50` are concentrated around the critical threshold (8) that we are using to classify our samples: 

![pIC50](https://github.com/EmYassir/pIC50_predictor/assets/20977650/2dc76ec3-2c54-42f2-8949-b2f7bde76d85)

Going with regression can be tricky, as we can achieve good regression performance and yet get a bad classification score: the regressor can approach the threshold (8) from both sides (from under or above), which means it can classify correctly or incorrectly depending on which side of the threshold the prediction falls. Therefore, we are moving forward with **treating the problem as a classification task**.

### Data split
We split the dataset into two shards:
1- Training set (80% of the data, or X samples).
2- Validation set (20% of the data, or X samples).

## Models
We run experiments on one baseline and two models. The baseline we chose is the random forest ([Scikit-learn implementation](https://scikit-learn.org/stable/)) as it is suggested by [this article](https://projects.volkamerlab.org/teachopencadd/talktorials/T007_compound_activity_machine_learning.html). The two other models we chose are [CatBoostClassifier](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier) and [ChemBertA](https://arxiv.org/abs/2010.09885). For the latter, we are using [simpletransformers implementation](https://github.com/ThilinaRajapakse/simpletransformers). 

## How to run the code

### Setup
There are few specific dependencies to install before launching an execution, you can use the command `pip install -r requirements.txt`.

**Important note**: The code running ChemBertA makes use of GPUs. Make sure to have one if you are running this model.

### Running the main script

### Code organization
In the following, we explain how to finetune/pretrain/distil GPT2 architectures. Connecting to the huggingface repo is necessary while downloading the tokenizer, but this is not possible with due to the company's proxy. Users can follow these steps to deactivate the SSL verification from the environment side:
1- If you are using a `conda` environment, go to `~/.conda/envs/<NAMEOFYOURENVIRONEMNT>/lib/python3.8/site-packages/requests/sessions.py`.
2- Search for `self.verify = True`.
3- Change it to `self.verify = False`.

### Modules

### The configuration file


### Training and evaluation 
The tables below present the hyper-parameters used for each model:

1- Random Forest
| n_estimatos | Criterion | Max depth |
|  :---: |  :---: |  :---:  | 
| 100 | "entropy" | Unlimited | 
The rest of hyperparameters/parameters values are the default ones.

2- CatBoostClassifier (all features)
| iterations | Learning rate | Depth |  Minimum data in leaves |
|  :---: |  :---: |  :---:  |  :---:  | 
| 200 | 0.008 | 11 | 47 |
The rest of hyperparameters/parameters values are the default ones.

3- CatBoostClassifier (featurized smiles only)
| iterations | Learning rate | Depth |  Minimum data in leaves |
|  :---: |  :---: |  :---:  |  :---:  | 
| 166 | 0.05 | 11 | 47 |
The rest of hyperparameters/parameters values are the default ones.

4- ChemBertA
| Architecture | Layers | Hidden size | Inter. size | Att. heads | Vocab. size | Dropout prob. | epochs |
|  :---: |  :---: |  :---:  | :---: |  :---: |  :---:  |   :---:  |   :---:  | 
| "Roberta" | 6 | 768 | 3072 | 12 |  52000 |  0.1 | 15 |
The rest of hyperparameters/parameters values are the default ones.


## Results and discussion
The table below shows the results we obtained:
| Model | Accuracy | Sensitivity | Specificity | F1 |
|  :---: |  :---: |  :---:  |  :---: |  :---:  |  
| Random Forest (baseline) | 0.86 | 0.57 | 0.93 | 0.61 |
| CatBoostClassifier (all features) | 0.85 | 0.43 | **0.95** | 0.53 |
| CatBoostClassifier (smiles only) | 0.86 | 0.49 | **0.95** | 0.56 |
| ChemBertA (`seyonec/PubChem10M_SMILES_BPE_396_250`) | **0.87** | **0.60** | 0.93 | **0.63** |

As shown in the results, the baseline model is pretty strong, although it only uses smiles embeddings. The CatBoostClassifier performs better when fed with smiles embedding only, which might question the usefulness of other features (provided that CatBoost models handle categorical, continous and embedding data). However, as we did not explore this problem in depth, nor did we perform hyper-parameter finetuning, we cannot draw a conclusion yet. Despite only using the smiles features, the ChemBertA model outperforms all others, asserting the superiority of pre-trained language models in molecules predictions.

## Next steps
As next steps, we plan to propose an architecture that is able to both leverage pre-trained models (ChemBertA) and make use of other features (continuous and categorical). For that, we need to embed categorical data, rescale continuous variables and make use of dense layers to conveniently process the data. The figure below describes the full proposed architecture:

![archi](https://github.com/EmYassir/pIC50_predictor/assets/20977650/20bbe5b5-76cd-4104-bcbe-9635afcb73e3)

We also plan to put some effort in finetuning hyper-parameters, as the performance gains could be non-negligible. 

## References
. [ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction](https://arxiv.org/abs/2010.09885)
. [T007 · Ligand-based screening: machine learning](https://projects.volkamerlab.org/teachopencadd/talktorials/T007_compound_activity_machine_learning.html)
. [simpletransformers implementation](https://github.com/ThilinaRajapakse/simpletransformers)
