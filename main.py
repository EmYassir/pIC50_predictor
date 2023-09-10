import os
import sys
sys.path.insert(1, os.getcwd())

import time
import argparse
import logging
from src.utils import SEED
from src.utils import set_seed, load_config, model_performance, prepare_dataset, split_dataset, help, parse_models

## Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)

def main():
    

    ## Parse arguments
    parser = argparse.ArgumentParser(description="Pipeline training for Epidermal Growth Factor Receptor prediction models.")
    parser.add_argument("--dataset_path", type=str, default="", help="The path to the dataset's csv file.")
    parser.add_argument("--configs_path", type=str, required=True, help="The path to the file containing models' configurations.")
    parser.add_argument("--output_dir", type=str, default="./output", help="The dump directory.")
    parser.add_argument('--train',  action='store_true', help="Runs training.")
    parser.add_argument('--valid', action='store_true', help="Runs validation.")
    parser.add_argument('--save', action='store_true', help="Saves the weights after training.")
    parser.add_argument('--seed', type=int, default=SEED, help="The seed.")
    parser.add_argument('--h', action='store_true', help="Displays the help menu.")
    args = parser.parse_args()


    if args.h:
        help()
        exit(0)
    if not args.train and not args.valid:
        logger.info(f"Neither '--train' nor '--valid' options were provided. At least one these options should be selected.")
        exit(0)
    ## Setting up cuda device
    #device = torch.cuda.device(f"cuda:{args.device}") 
    #torch.cuda.set_device(device)
    cfgs, n_saved = [], 0
    cfgs = load_config(args.configs_path)
    for cfg in cfgs:
        if cfg['weights']:
            n_saved += 1
    if not args.train:
        if cfgs and n_saved != len(cfgs):
            logger.info(f"When only '--valid' option is provided, all models being tested should have saved weights. Please provide a saved weights' path in the configuration file")
            exit(1)
    
    # Parse models
    models = parse_models(cfgs)
    if len(models) != len(cfgs):
        raise ValueError("Error while parsing the models configuration")
    
    # Seed
    set_seed(args.seed)
   
    
    ## Loading/preparing the dataset
    df = prepare_dataset(args.dataset_path, featurize=True)
    df_train, df_eval = split_dataset(df)

    ## Making output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Training
    if args.train:
        for m in models:
            logger.info(f"#### Training model {m.config['name']} ({m.config['type']})")
            if m.config['weights']:
                logger.info(f"## Loading weights from {m.config['weights']}")
                m.load_weights(m.config['weights']) 
            output_dir = os.path.join(args.output_dir, m.config['name'] + '_train')
            m.fit(df_train, df_eval, output_dir)
            if args.save:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join(args.output_dir, m.config['name'] + '_' + timestr)
                logger.info(f"#### Saving the model to {filename}...")
                m.save(filename)

    # Validation
    if args.valid:
        results = {}
        for i,m in enumerate(models):
            logger.info(f"#### Evaluating model {m.config['type']} ({m.config['name']})")
            if not args.train:
                # Means that models were not trained, so we need to load them
                logger.info(f"#### Loading weights from {m.config['weights']}")
                m.load_weights(m.config['weights'])          
            key = m.config['name'] if m.config['name'] not in results else m.config['name'] + str(i)
            results[key] = model_performance(m, df_eval, True, args.output_dir)
    
    logger.info(f'###########################################')
    logger.info(f'##############    SUMMARY    ##############')
    logger.info(f'###########################################')
    for name, res in results.items():
        logger.info(f'####### Model {name}:')
        logger.info(f"## => Accuracy: {res['Accuracy']:.2f}")
        logger.info(f"## => Sensitivity: {res['Sensitivity']:.2f}")
        logger.info(f"## => Specificity: {res['Specificity']:.2f}")
        logger.info(f"## => F1 score: {res['F1 score']:.2f}")

    logger.info(f'#### Done.')


if __name__ == "__main__":
    main()