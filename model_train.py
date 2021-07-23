# -*- coding: utf-8 -*-

from Text_processing import Processing
from model import train_models
from Data import InputData
from Parser import Parser
import pandas as pd

"""

Created on 15.06.2021

@author: Nikita

    The module is designed for training a neural network. The neural network was trained using the accuracy metric 
    and the entropy function binary_cross. The accuracy of the model is 0.96. To train a neural network, it is enough 
    to run the module as the main program. To display neural network error graphs, you need "show_model" = True.

"""


def load_data(path):

    text_articles_mchs = Parser.pars_local(fr"{path}\data")
    articles = pd.read_json(fr"{path}\MCHS_2300.json", orient='records')
    text = pd.read_json(fr"{path}\topic_full.json", orient='records')

    return text_articles_mchs, articles, text


def text_processing(base_path: str, index: int, text_list: list, language='russian'):

    texts = [' '.join(item) for item in Processing.get_text(text=text_list,
                                                            base_path=base_path,
                                                            language=language)]

    df_text = pd.DataFrame(texts, columns=['text'])
    df_text['Index'] = index

    return df_text


def create_dataframe(text_articles_mchs, articles, text, path):

    """

    The module is designed to create a single dataframe. The modules are loaded with 3 datasets, two of which relate
    to the structure of the Ministry of Emergency Situations and the third dataset is made up of comprehensive topics.
    The dataset data is passed to the "text_processing" for text processing and assignment "index".

    Parameters text_processing
    ----------

    base_path : Location of the module package.

    index : This is the class that the text belongs to.

            1 - The subject of the text that relates to the structure of the Ministry of Emergency Situations.
            0 - The subject of the text, which refers to comprehensive topics.

    text_list : This is a list with texts.

    """

    text_mchs = text_processing(base_path=path,
                                index=1,
                                text_list=text_articles_mchs)

    text_articles = text_processing(base_path=path,
                                    index=1,
                                    text_list=articles.text)

    text_text = text_processing(base_path=path,
                                index=0,
                                text_list=text.text)

    df = pd.concat([text_mchs, text_articles, text_text], axis=0, ignore_index=True)

    return df


def tokenization(df):

    """

    The module is designed to form a bag of tokenizer words, as well as to divide the dataset
    into a training, test, and validation sample.

    Parameters InputData
    ----------

    text : Dataframe containing text for vectorization.

    max_words : Maximum number of words.

    max_len : Maximum news length.

    columns_name_labels : Сolumn name of the independent variable.

    columns_name_text : Сolumn name of the dependent variable.

    """

    data = InputData(text=df,
                     columns_name_text=columns_name_text,
                     columns_name_labels=columns_name_labels,
                     max_words=max_words,
                     max_len=max_len)

    bag_of_words = data.tokenizer()
    x_train, y_train, x_test, y_test, x_val, y_val = data.data_separation(bag_of_words)

    return x_train, y_train, x_test, y_test, x_val, y_val


def train(show_model: bool = False):

    text_articles_mchs, articles, text = load_data(path)
    df = create_dataframe(text_articles_mchs, articles, text, path)
    x_train, y_train, x_test, y_test, x_val, y_val = tokenization(df)

    model = train_models(x_train=x_train,
                         y_train=y_train,
                         x_test=x_test,
                         y_test=y_test,
                         max_words=max_words,
                         max_len=max_len)

    model_lstm = model.model_lstm(show_structure=True)

    train_model = model.model_lstm_train(model=model_lstm,
                                         epochs=epochs,
                                         batch_size=batch_size,
                                         save_file=save_file,
                                         show_model=show_model)


if __name__ == "__main__":

    path = r'C:\PythonProjects\Jobs\LSTM_model'
    save_file = 'weights_new'

    columns_name_text = 'text'
    columns_name_labels = 'Index'

    max_words = 10000
    max_len = 250
    epochs = 6
    batch_size = 5

    train(show_model=True)
