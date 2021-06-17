## Text Classification

The *purpose* of this repository is to create a neural network model for binary classification of scientific articles on the topic of "Disaster Medicine in Emergencies" in NLP with deep learning.

---
A sample of 532 articles was used as input, where 266 scientific articles were manually marked by an expert and had a label of 1, and the remaining 266 articles were taken from an open source Habr.com and had a label of 0. The data was divided into 3 data sets: training, validation, test, and mixed. The purpose of the model was to recognize articles that relate to the topic "Disaster medicine in emergency situations". The model was evaluated using the "*accuracy* " metric, and the model's accuracy was 98%.


## Usage
1. The model is located in `RNN_model_rus.py.`
2. Run python `RNN_model_rus.py` to predict the topic of a scientific article, if you need to train the model, you need to call the function `show_model()`.

## Components of the model

The block contains a brief excerpt of the project files, a more detailed description is located inside each module.


`RNN_model_rus.py` - The module is designed to predict the topic of a scientific article (whether it relates to the topic of "Disaster Medicine in emergency situations" or not).

`Input_data.py` - The module is designed to prepare input data for a neural network.

`Parser.py` - The module is designed for local parsing of html files of scientific articles from the data\2015 folder.

`Reindex.py` - The module is designed to change the index of topics of scientific articles and group them.

`Text_processing.py` - This is a code designed for processing text in Russian and English.

`"weight. h5"` - Coefficients of the trained neural network.

`"texts.json"` - Contains articles with Habr.com.

`"2015.docx"` - Contains tags of scientific articles marked up by an expert.
## Version

Python 3.8

Tensorflow 2.4.1
