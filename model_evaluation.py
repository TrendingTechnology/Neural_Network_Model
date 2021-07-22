# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from model import train_models
from metrics import Metrics
from Data import InputData
import model_train
import numpy as np


"""

Created on 25.06.2021

@author: Nikita

    The module is designed to evaluate a neural network model using various metrics.

"""


def create_y_predict(model, x_val, main_path, weights_file, file_y_predict):

    """

    The function is designed to create "y_predict" using the weights of the neural network, as well as to write
    "y_predict" to a separate file to reuse them.

    """

    model.load_weights(fr'{main_path}\{weights_file}.h5')

    with open(fr'{main_path}\{file_y_predict}.txt', 'w') as document:

        for k in range(len(x_val)):

            x_val = np.expand_dims(x_val[k], axis=0)
            y_predict = model.predict(x_val)
            document.write(f'{float(y_predict)},')

        document.close()


def get_values(model, y_val, main_path, weights_file, file_y_predict, step, threshold_tpr, threshold_fpr):

    """

    The function is designed to get the values of "tpr, fpr, roc_auc", as well as to find the threshold value
    for the binary classification of the text.

    """

    model.load_weights(fr'{main_path}\{weights_file}.h5')

    with open(fr'{main_path}\{file_y_predict}.txt', 'r') as document:
        y_predict = [float(item) for item in document.read().split(',') if len(item) >= 1]

    metric = Metrics(y_predict=y_predict, y_true=y_val)
    tpr, fpr, thresholds = metric.curve_values(step=step)
    roc_auc = metric.auc_roc(tpr, fpr)

    threshold = metric.get_threshold(tpr=tpr,
                                     fpr=fpr,
                                     threshold_values=thresholds,
                                     threshold_tpr=threshold_tpr,
                                     threshold_fpr=threshold_fpr)

    return tpr, fpr, roc_auc, threshold


def show_auc_roc(tpr, fpr, roc_auc, threshold):

    """

    The function is intended for plotting "AUC-ROC". The drawing will be saved in the working directory
    under the name "AUC-ROC.png".

    """

    plt.subplots(figsize=(10, 6))
    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr,
             tpr,
             color='#001DBC')

    plt.plot([0, 1],
             [0, 1],
             linestyle='--',
             color='#452732')

    plt.xticks(np.arange(0, 1, 0.05),
               rotation=90,
               color='#452732')

    plt.yticks(np.arange(0, 1, 0.05),
               color='#452732')

    plt.text(0.83,
             0.25,
             f'AUC = {roc_auc:.3f}\nThreshold = {threshold:.3f}',
             family="serif",
             color='#452732',
             fontsize=9)

    plt.xlabel('False Positive Rate', color='#452732')
    plt.ylabel('True Positive Rate', color='#452732')
    plt.savefig('AUC-ROC.png')
    plt.show()


def evaluation():

    """

    The function is designed to connect the necessary modules and start evaluating the neural network model
    using the AUC metric.

    """

    text_articles_mchs, articles, text = model_train.load_data(path)

    df = model_train.create_dataframe(text_articles_mchs, articles, text, path)

    data = InputData(text=df,
                     columns_name_text=columns_name_text,
                     columns_name_labels=columns_name_labels,
                     max_words=max_words,
                     max_len=max_len)

    tokenizer = data.tokenizer()

    x_train, y_train, x_test, y_test, x_val, y_val = data.data_separation(tokenizer)

    model = train_models(x_train=x_train,
                         y_train=y_train,
                         x_test=x_test,
                         y_test=y_test,
                         max_words=max_words,
                         max_len=max_len)

    model_lstm = model.model_lstm(show_structure=False)

    tpr, fpr, roc_auc, threshold = get_values(model=model_lstm,
                                              y_val=y_val,
                                              main_path=path,
                                              weights_file=weights_file,
                                              file_y_predict=file_y_predict,
                                              step=step,
                                              threshold_tpr=threshold_tpr,
                                              threshold_fpr=threshold_fpr)

    show_auc_roc(tpr, fpr, roc_auc, threshold)


if __name__ == "__main__":

    path = r'C:\PythonProjects\Jobs\LSTM_model'
    weights_file = 'weights'
    file_y_predict = 'y_predict'

    columns_name_text = 'text'
    columns_name_labels = 'Index'

    threshold_tpr = 0.97
    threshold_fpr = 0.1
    max_words = 10000
    max_len = 250
    step = 0.0002

    evaluation()
