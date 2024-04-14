from train_utils import setup_and_train
from config import temp_transform, TRAIN_DATA_PATH, TEST_DATA_PATH
import mlflow


def main():
    mlflow.set_experiment('FER-2013 dataset')

    setup_and_train(TRAIN_DATA_PATH, TEST_DATA_PATH, temp_transform)


if __name__ == "__main__":
    main()
