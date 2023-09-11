import os
import sys
import time
import shutil
import logging
import pickle as pkl
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from simpletransformers.classification import ClassificationModel

## Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


class Model:
    def __init__(self):
        pass

    def fit(self, train_df, eval_df = None, output_dir = None):
        raise NotImplemented()

    def predict(self, dev_df, output_dir=None):
        raise NotImplemented()
    
    def save(self, path):
        raise NotImplemented()

    def load_weights(self, path):
        raise NotImplemented()


class RF(Model):
    def __init__(self, model_cfg):
        super().__init__()
        self.type = model_cfg["type"]
        self.name = model_cfg["name"]
        self.config = model_cfg
        self.model = RandomForestClassifier(**self.config['model_params'])

    def fit(self, train_df, eval_df = None, output_dir = None):
        # extract train_x, train_y
        if self.config['model_feats']:
            train_x = train_df[self.config['model_feats']].values.tolist()
        else:
            train_x = train_df.smiles_fp.tolist()
        train_y = train_df.labels.tolist()
        self.model.fit(train_x, train_y, **self.config['train_params'])

    def predict(self, dev_df, output_dir=None):
        if self.config['model_feats']:
            dev_x = dev_df[self.config['model_feats']].values.tolist()
        else:
            dev_x = dev_df.smiles_fp.tolist()
        return self.model.predict(dev_x)
    
    def save(self, path):
        with open(path, 'wb') as fp:
            pkl.dump(self.model, fp)

    def load_weights(self, path):
        with open(path, 'rb') as fp:
            self.model = pkl.load(fp)


class CatBoostCls(Model):
    def __init__(self, model_cfg):
        super().__init__()
        self.type = model_cfg["type"]
        self.name = model_cfg["name"]
        self.config = model_cfg
        self.model = CatBoostClassifier(**self.config['model_params'])

    def fit(self, train_df, eval_df = None, output_dir = None):
        if self.config['model_feats']:
            train_x = train_df[self.config['model_feats']].values.tolist()
        else:
            train_x = train_df.smiles_fp.tolist()
        train_y = train_df.labels.tolist()
        if eval_df is not None:
            if self.config['model_feats']:
                dev_x = eval_df[self.config['model_feats']].values.tolist()
            else:
                dev_x = eval_df.smiles_fp.tolist()
            dev_y = eval_df.labels.tolist()
            self.config['train_params']['eval_set'] = (dev_x, dev_y)
        self.model.fit(train_x, train_y, **self.config['train_params'])

    def predict(self, dev_df, output_dir=None):
        if self.config['model_feats']:
            dev_x = dev_df[self.config['model_feats']].values.tolist()
        else:
            dev_x = dev_df.smiles_fp.tolist()
        return self.model.predict(dev_x)
    
    def save(self, path):
        self.model.save_model(path)

    def load_weights(self, path):
        self.model.load_model(path)


class ChemBerta(Model):
    def __init__(self, model_cfg):
        super().__init__()
        self.type = model_cfg["type"]
        self.name = model_cfg["name"]
        self.config = model_cfg
        self.model = ClassificationModel(**self.config['model_params'])

    def fit(self, train_df, dev_df, output_dir = None):
        if output_dir:
            if os.path.isdir(output_dir) and self.config['overwrite_output_dir']:
                shutil.rmtree(output_dir)
            self.config['train_params']['output_dir'] = output_dir
        self.config['train_params']['acc'] = accuracy_score
        self.model.train_model(train_df.rename(columns={"smiles": "text"}), eval_df=dev_df.rename(columns={"smiles": "text"}), **self.config['train_params'])

    def predict(self, dev_df, output_dir=None):
        out_dir = "./output_eval"
        if output_dir:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            out_dir = os.path.join(output_dir, 'eval_' + timestr)
        _, outputs, _ = self.model.evaluate(dev_df.rename(columns={"smiles": "text"}), out_dir)
        return np.argmax(outputs, axis=1)
    
    def save(self, path):
        self.model.save_model(output_dir=path)

    def load_weights(self, path):
        self.model = ClassificationModel(model_type=self.config["model_params"]["model_type"], model_name=path)