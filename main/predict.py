import copy
from utils import get_args_parser
from inferences import Predict

def main(train_cfg, prep_cfg):
  options = prep_cfg.get('options')
  output_csv = prep_cfg.get('output_data')
  Predict(
    options.get('y_scaler'),
    output_csv.get('train_target_original_csv'),
    options.get('index_col'),
    options.get('target_col'),
    **train_cfg
  ).run()

if __name__ == "__main__":
  train_args = get_args_parser().parse_args()
  config = {}
  exec(open(train_args.config).read())
  train_cfg = copy.deepcopy(config)
  
  preprocess_args = get_args_parser(config_type='preprocess').parse_args()
  exec(open(preprocess_args.config).read())
  
  main(train_cfg, config)