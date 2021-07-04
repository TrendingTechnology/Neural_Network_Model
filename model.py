# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

"""

Created on 09.06.2021

@author: Nikita

    This module presents the architecture of neural network models.

"""


class train_models:

    def __init__(self, x_train, y_train, x_test, y_test, max_words, max_len):
        self.max_words = max_words
        self.max_len = max_len
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    """
    
    Parameters
    ----------
    max_words : The maximum number of unique words in the word bag for vectorization.
    
    max_len : The maximum number of words in the text.

    """

    def model_lstm(self, show_structure: bool = False):

        """

        A recurrent neural network of the Long short-term memory type, capable of learning long-term dependencies.
        The Sequence to one recurrent neural network configuration was used. This configuration is applicable for
        solving problems of text or video classification, when a sequence of words (or images) is fed to the input
        of the network, and as an output we get a single probability vector for the classes.

        """

        model = Sequential()
        model.add(Embedding(self.max_words, 12, input_length=self.max_len))
        model.add(LSTM(6))
        model.add(Dropout(0.6))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        if show_structure:
            model.summary()

        return model

    def model_lstm_train(self, model, epochs, batch_size, save_file, show_model: bool = True):

        """

        Parameters
        ----------
        model : Neural network model architecture.

        epochs : Number of training epochs.

        batch_size : The number of samples per gradient update.

        save_file : The location of the file where the weights of the neural network will be saved.

        show_model : A function for displaying a graph of neural network training errors as a percentage.

        """

        model_lstm_save_file = f'{save_file}.h5'
        checkpoint_callback_lstm = ModelCheckpoint(model_lstm_save_file,
                                                   monitor='val_accuracy',
                                                   save_best_only=True,
                                                   verbose=1)

        history_lstm = model.fit(self.x_train,
                                 self.y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(self.x_test, self.y_test),
                                 callbacks=[checkpoint_callback_lstm])

        if show_model:
            plt.plot(history_lstm.history['loss'],
                     label='The number of errors in the percentage on the training set')
            plt.plot(history_lstm.history['val_loss'],
                     label='The number of errors in the percentage on the test set')
            plt.xlabel('The Age of Learning')
            plt.ylabel('Percentage of correct predictions')
            plt.legend()
            plt.show()

        return history_lstm
