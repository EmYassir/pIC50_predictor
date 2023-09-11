import sys
import random
import logging
import torch
import json
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

from sklearn.metrics import auc, accuracy_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from .models import RF, CatBoostCls, ChemBerta


## Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


MODELS = {"RandomForest", "CatBoostClassifer", "ChemBerta"}
SEED = 99


def set_seed(seed):
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(cfg_path):
    with open(cfg_path, "r") as fp:
        return json.load(fp)


def load_dataset(ds_path):
    df = pd.read_csv(ds_path)
    df = df.drop(['Unnamed: 0', 'ro5_fulfilled', 'molecule_chembl_id', 'units'], axis=1)
    df['labels'] = np.where(df['pIC50'] > 8.0, 1.0, 0.0)
    return df


def model_performance(ml_model, dev_df, verbose=True, output_dir=None):
    """
    Helper function to calculate model performance

    Parameters
    ----------
    ml_model: sklearn model object
        The machine learning model to train.
    dev_x: list
        Molecular fingerprints for test set.
    dev_y: list
        Associated activity labels for test set.
    verbose: bool
        Print performance measure (default = True)

    Returns
    -------
    tuple:
        Accuracy, sensitivity, specificity on dev set.
    """
    # Prediction class on test set
    dev_pred = ml_model.predict(dev_df, output_dir)
    dev_y = dev_df.labels.values.tolist()


    # Performance of model on test set
    accuracy = accuracy_score(dev_y, dev_pred)
    sens = recall_score(dev_y, dev_pred)
    spec = recall_score(dev_y, dev_pred, pos_label=0)
    f1 = f1_score(dev_y, dev_pred)

    if verbose:
        # Print performance results
        logger.info(f"Accuracy: {accuracy:.2f}")
        logger.info(f"Sensitivity: {sens:.2f}")
        logger.info(f"Specificity: {spec:.2f}")
        logger.info(f"F1 score: {f1:.2f}")

    return {"Accuracy": accuracy, 
            "Sensitivity": sens, 
            "Specificity": spec, 
            "F1 score": f1}


def smiles_to_fp(smiles, method="maccs", n_bits=2048):
    # convert smiles to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)

    if method == "maccs":
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    if method == "morgan2":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        return np.array(fpg.GetFingerprint(mol))
    if method == "morgan3":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
        return np.array(fpg.GetFingerprint(mol))
    else:
        # NBVAL_CHECK_OUTPUT
        logger.info(f"Warning: Wrong method specified: {method}. Default will be used instead.")
        return np.array(MACCSkeys.GenMACCSKeys(mol))

def tokenize_smiles(smile_token, tokenizer):
    return tokenizer.encode_plus(smile_token, return_tensors='pt', add_special_tokens=True)


def parse_models(cfg_dict):
    models = []
    for model_cfg in cfg_dict:
        mtype = model_cfg["type"]
        if mtype == "RandomForest":
            models.append(RF(model_cfg))
        elif mtype == "CatBoostClassifier":
            models.append(CatBoostCls(model_cfg))
        elif mtype == "ChemBerta":
            models.append(ChemBerta(model_cfg))
    return models



def process_dataset(df, tokenize=False, tokenizer=None):
    
    df["smiles_fp"] = df["smiles"].apply(smiles_to_fp, args=("maccs", 2048))
    if tokenize:
        if not tokenizer:
            raise ValueError("Variable 'tokenizer' should be set when parameter 'tokenize' is True.")
        df["smiles_tok"] = df["smiles"].apply(tokenize_smiles, args=(tokenizer,))
    return df

def split_dataset(df, lbl_name="labels", train_size=0.8, seed=42):
    df_train, df_eval = train_test_split(df, shuffle=True, stratify=df[lbl_name], test_size=(1.0 - train_size), random_state=seed)
    return(df_train, df_eval)






def help():
    logger.info(f"######################## HELP ########################")
    logger.info(f"## script options")
    logger.info(f"--dataset_path: path to the dataset's csv file")
    logger.info(f"--configs_path: The path to the file containing models' configurations. Models supported: {MODELS}")
    logger.info(f"--output_dir: The dump directory.")
    logger.info(f"--train: Runs models training.")
    logger.info(f"--valid: Runs models validation.")
    logger.info(f"--save: Saves the weights after training.")
    logger.info(f"--seed: The seed used for experiments.")
    logger.info(f"--help: Displays the help menu.")
    logger.info(f"######################################################")
