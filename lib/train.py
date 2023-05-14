import json
import os
import pickle
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
import mlflow

mlflow.set_tracking_uri('http://158.160.11.51:90/')
mlflow.set_experiment('valeevar_hw')

RANDOM_SEED = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

METRICS = {
    'recall': partial(recall_score, average='macro'),
    'precision': partial(precision_score, average='macro'),
    'accuracy': accuracy_score,
}


MODELS = {
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'SGDClassifier': SGDClassifier(),
    'LogisticRegression': LogisticRegression(),
}


def save_dict(data: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_dict(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)


def train_model(x, y, model):
    # model = DecisionTreeClassifier()
    model.fit(x, y)
    return model


def train():
    with open('../params.yaml', 'r') as f:
        params_data = yaml.safe_load(f)

    config = params_data['train']

    iris = datasets.load_iris()
    task_dir = '../data/train'

    x = iris['data'].tolist()
    y = iris['target'].tolist()

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=config['test_size'])

    for model_name in params_data['train']['model']:

        with mlflow.start_run() as run:
            model1 = MODELS[model_name]

            model = train_model(train_x, train_y, model1)

            preds = model.predict(x)

            metrics = {}
            for metric_name in params_data['eval']['metrics']:
                metrics[metric_name] = METRICS[metric_name](y, preds)

            save_data = {
                'train_x': train_x,
                'test_x': test_x,
                'train_y': train_y,
                'test_y': test_y,
            }

            if not os.path.exists(task_dir):
                os.mkdir(task_dir)

            save_dict(save_data, os.path.join(task_dir, 'data.json'))
            save_dict(metrics, os.path.join(task_dir, 'metrics.json'))

            sns.heatmap(pd.DataFrame(train_x).corr())


            params = {}

            for i in params_data.values():
                params.update(i)

            params['model'] = model_name

            params['run_type'] = 'train'

            print(f'train params - {params}')
            print(f'train metrics - {metrics}')

            pic_name = '_'.join([str(i) for i in params.values()])
            plt.savefig(f'../data/train/heatmap_{pic_name}.png')

            with open('../data/train/model.pkl', 'wb') as f:
                pickle.dump(model, f)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)


if __name__ == '__main__':
    train()
