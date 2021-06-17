## Text Classification

The *purpose* of this repository is to create a neural network model of NLP with deep learning for binary classification of texts related to the Ministry of Emergency Situations.

---

## Data

A sample of 4,300 texts was used as input, of which 2,800 texts were labeled 1:

   1) 2300 texts were obtained by parsing sites such as [rg.ru](https://rg.ru), [iz.ru](https://iz.ru) and others;
   2) 500 scientific articles were marked by an expert manually (scientific articles are intended for further development of the model, in particular, the classification of texts on 3 topics: Comprehensive topics, the topic of the Ministry of Emergency Situations, the topic "Disaster medicine in emergency situations", at the moment, a dataset is being formed on the topic "Disaster Medicine in Emergency situations" and a comprehensive topic is being finalized).

The remaining 1,500 texts were obtained by parsing a scientific journal on comprehensive topics and were labeled 0. The data was divided into 3 data sets: training, validation, test, and mixed. Data on scientific articles on the topic "Disaster Medicine in Emergency situations" can be found in [Scientific articles](https://github.com/Non1ce/Data_LSTM#readme).

## LSTM model

The purpose of the model was to recognize text related to the structure of the Ministry of Emergency Situations. A recurrent neural network with long-term short-term memory (LSTM) was used as a model.

<a href="url"><img src="https://github.com/Non1ce/Image/blob/image/LSTM/LSTM.png" align="middle" height="250" width="250" ></a>


Neural Network Architecture:

   1. Embedding

   2. LSTM

   3. Dropout

   4. Dense + Sigmoid




The neural network was trained using the "*accuracy*" metric and the entropy function binary_cross. The accuracy of the model is 95%. In the future, it is planned to introduce other accuracy metrics (f1_score, f beta_score, etc.), as well as additional training of the neural network on additional topics. 



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
