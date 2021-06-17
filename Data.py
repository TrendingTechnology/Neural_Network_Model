# -*- coding: utf-8 -*-

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


"""

Created on 08.06.2021

@author: Nikita

    The module is designed to prepare input data for a neural network. Split into 3 sets (training, validation,
    testing). The data is a set of 2 classes. The first class contains a list of texts related to the structure 
    of the Ministry of Emergency Situations, and the second class contains texts that are parsed from 
    different sites on different topics.

"""


class InputData:

    def __init__(self, text, max_words, max_len, columns_name_labels, columns_name_text):

        self.text = text
        self.max_words = max_words
        self.max_len = max_len
        self.columns_name_labels = columns_name_labels
        self.columns_name_text = columns_name_text

        """
        Parameters
        ----------
        text : Dataframe containing text for vectorization.
        
        max_words : Maximum number of words.
        
        max_len : Maximum news length.
        
        columns_name_labels : Сolumn name of the independent variable.
        
        columns_name_text : Сolumn name of the dependent variable.
        
        """

    def tokenizer(self):

        tokenizers = Tokenizer(num_words=self.max_words)
        tokenizers.fit_on_texts(self.text[f'{self.columns_name_text}'])

        return tokenizers

    def data_separation(self, tokenizers):

        sequences = tokenizers.texts_to_sequences(self.text[f'{self.columns_name_text}'])

        # Input data.
        x = pad_sequences(sequences, maxlen=self.max_len)
        y = self.text[f'{self.columns_name_labels}']

        # Divide the data into training, validation, and test data sets.
        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            train_size=0.7,
                                                            random_state=50,
                                                            stratify=y)

        x_test, x_val, y_test, y_val = train_test_split(x_test,
                                                        y_test,
                                                        train_size=0.5,
                                                        random_state=42,
                                                        stratify=y_test)

        return x_train, y_train, x_test, y_test, x_val, y_val
