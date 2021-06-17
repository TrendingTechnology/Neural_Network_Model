# -*- coding: utf-8 -*-

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords
import pymorphy2


"""

Created on 03.06.2021

@author: Nikita

    This is code designed for processing rus and eng text. The code has two methods, the first is a static method and is 
    used to add additional Russian stopwords to the stopwords bag. The second method processes the text, removes 
    stopwords, punctuation in sentences, and reduces all words to lowercase and to a single (initial) form. 
    This module outputs the result in the form of word sequences.

"""


class Processing:

    # Preprocessing text
    @staticmethod
    def text(base_path, text: list, del_stopwords: bool = True, language=None, add_rus_stopwords: bool = True):

        """
        Parameters
        ----------

        base_path : Location of the file with additional stopwords.

        text : Dataframe containing text.

        del_stopwords : Function for deleting stop words. This feature is enabled by default. If you don't need to
                        delete stop words, specify del_stopwords = False in the function parameters.

        language : Text language.

        add_rus_stopwords : Function for adding additional stopwords.

        """

        token_sequence, new_text = [], []

        # Converts text to a sequence of words (or tokens).
        for line in text:
            tokens = text_to_word_sequence(line, filters='!"#$%&amp;()*+,-./:;&lt;=>?@[\\]^_`{|}~\t\n\ufeff', split=' ')
            token_sequence.append(tokens)

        if language == 'russian':

            morph = pymorphy2.MorphAnalyzer()
            for words in token_sequence:
                words = [w for w in words if w.isalpha()]

                if del_stopwords:
                    russian_stopwords = stopwords.words('russian')

                    if add_rus_stopwords:
                        with open(base_path + r'\Stopwords.txt', encoding='utf-8') as f:
                            stop_words = f.read()
                            stop_words = stop_words.split(" ")
                            for word in stop_words:
                                russian_stopwords.append(word)

                    words = [w for w in words if w not in russian_stopwords]

                words = [w for w in words if
                         w == 'чс' or len(list(w)) > 3]
                words = [morph.parse(w)[0].normal_form for w in words]  # bringing the words to a single(normal) form
                new_text.append(words)

        if language == 'english':

            for words in token_sequence:
                words = [w for w in words if w.isalpha()]

                if del_stopwords:
                    english_stopwords = stopwords.words('english')
                    words = [w for w in words if w not in english_stopwords]

                words = [w for w in words if len(list(w)) > 3]
                new_text.append(words)

        return new_text
