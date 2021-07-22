# -*- coding: utf-8 -*-

from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import train_models
from Data import InputData
import model_train


"""

Created on 15.06.2021

@author: Nikita

    The module is designed to predict the topic of the entered text. To make a prediction, it is enough to run the 
    module as the main program.The longer the text, the better the neural network classifies.

"""


def text_prediction(model, file_weight, tokenizer, max_len):

    model.load_weights(fr'{file_weight}.h5')

    input_text = input("Enter the text: ")

    text_sequences = tokenizer.texts_to_sequences([input_text])
    text_sequences = pad_sequences(text_sequences, maxlen=max_len)
    result = model.predict(text_sequences)

    if result > 0.809:
        print('Neural Network class: Тематика МЧС')
    else:
        print('Neural Network class: Всесторонняя тематика')


def prediction(const_prediction: bool = False):

    """

    If const_prediction=False, then the function triggers a single topic prediction for the text or
    Const_prediction=True, then there will be infinite texts predictions.

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

    model_lstm = model.model_lstm(show_structure=True)

    # Prediction #
    text_prediction(model=model_lstm,
                    file_weight=save_file,
                    tokenizer=tokenizer,
                    max_len=max_len)

    while const_prediction:

        text_prediction(model=model_lstm,
                        file_weight=save_file,
                        tokenizer=tokenizer,
                        max_len=max_len)


if __name__ == "__main__":

    path = r'C:\PythonProjects\Jobs\LSTM_model'
    save_file = 'weights'

    columns_name_text = 'text'
    columns_name_labels = 'Index'

    max_words = 10000
    max_len = 250

    prediction(const_prediction=True)
