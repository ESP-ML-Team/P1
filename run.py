import numpy as np
import altair as alt # visualization
from helpers import load_csv_data
from implementations import logistic_regression
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path

# can be run with: (uv run) python run.py max_iters=100 gamma=0.1
# can be run with: (uv run) python run.py -cn exp1
# save results in HydraConfig.get().run.dir or change the working directory and use to_absolute_path for everything else
@hydra.main(version_base=None, config_path="./configs", config_name="defaults")
def main(config):
    # x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/dataset")
    x_train = np.load("data/subsampled_dataset/x_train.npy")
    x_test = np.load("data/subsampled_dataset/x_test.npy")
    y_train = np.load("data/subsampled_dataset/y_train.npy")
    
    # w_initial = np.random.rand(x_train.shape[1])
    w_initial = np.zeros(x_train.shape[1])
    w, loss = logistic_regression(y_train, x_train, w_initial, config.hyperparameters.max_iters, config.hyperparameters.gamma)

if __name__ == "__main__":
    main()