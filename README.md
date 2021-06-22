## Text Classification

   The *purpose* of this repository is to create a neural network model of NLP with deep learning for binary classification of texts related to the Ministry of Emergency Situations.
   
---

## Data

   A sample of 4,300 texts was used as input, of which 2,800 texts were labeled 1:

   1) 2300 texts were obtained by parsing sites such as [rg.ru](https://rg.ru), [iz.ru](https://iz.ru) and others;
   2) 500 scientific articles were marked by an expert manually (scientific articles are intended for further development of the model, in particular, the classification of texts on 3 topics: Comprehensive topics, the topic of the Ministry of Emergency Situations, the topic "Disaster medicine in emergency situations", at the moment, a dataset is being formed on the topic "Disaster Medicine in Emergency situations" and a comprehensive topic is being finalized).

   The remaining 1,500 texts were obtained by parsing a scientific journal on comprehensive topics and were labeled 0. The data was divided into 3 data sets: training, validation, test, and mixed. Data on scientific articles on the topic "Disaster Medicine in Emergency situations" can be found in [Scientific articles](https://github.com/Non1ce/Data_LSTM#readme).

## LSTM model

   Long Short-Term Memory~(LSTM) was introduced by [S. Hochreiter and J. Schmidhuber](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory) and developed by many research scientists.

   To deal with these problems Long Short-Term Memory (LSTM) is a special type of RNN that preserves long term dependency in a more effective way compared to the basic RNNs. This is particularly useful to overcome vanishing gradient problem. Although LSTM has a chain-like structure similar to RNN, LSTM uses multiple gates to carefully regulate the amount of information that will be allowed into each node state. Figure shows the basic cell of a LSTM model.
   
<p align="center">
  <img width="407" height="298" src="https://github.com/Non1ce/Image/blob/image/LSTM/LSTM.png">
</p>

   A recurrent neural network with long-term short-term memory (LSTM) was used as a model. The purpose of the model was to recognize text related to the structure of the Ministry of Emergency Situations.
   
```python
def model_lstm(self, show_structure: bool = False):

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
```
## Neural Network Architecture:
   
   

<p align="center">1. Embedding :arrow_right: 2.LSTM :arrow_right: 3. Dropout :arrow_right: 4. Dense + Sigmoid</p>

<html>
 <body>
  <p class="thumb" align="center">
   <img src="https://github.com/Non1ce/Image/blob/image/LSTM/Model%20architecture.PNG" width="539" height="331" align="center"/>
  </p>
 </body>
</html>


   The neural network was trained using the "*accuracy*" metric and the entropy function binary_cross. The accuracy of the model is *96%*. In the future, it is planned to introduce other accuracy metrics (f1_score, f beta_score, etc.), as well as additional training of the neural network on additional topics. 

 
 
<html>
 <body>
  <p class="thumb" align="center">
   <img src="https://github.com/Non1ce/Image/blob/image/LSTM/img.PNG" alt="Фотография 1" width="486" height="369">
   <img src="https://github.com/Non1ce/Image/blob/image/LSTM/result2.PNG" alt="Фотография 2" width="881" height="253">
  </p>
 </body>
</html>


## Usage
1. The model is located in [`model.py.`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/model.py)
2. Run the module [`model_predict.py`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/model_predict.py) to predict the topic of a scientific article, if you need to train the model, you need to run a module [`model_train.py`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/model_train.py).

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
