## Text Classification

   The *purpose* of this repository is to create a neural network model of NLP with deep learning for binary classification of texts related to the Ministry of Emergency Situations.
   
---

## Components of the model

The block contains the structure of the project, as well as a brief excerpt from the files, a more detailed description is located inside each module.


<html>
 <body>
  <p class="thumb" align="center">
   <img src="https://github.com/Non1ce/Image/blob/no_nice/LSTM/Description_LSTM.svg" align="center"/>
  </p>
 </body>
</html>

[`model_predict.py`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/model_predict.py) - The module is designed to predict the topic of the text, whether the text belongs to the structure of the Ministry of Emergency Situations or not.

[`model_train.py`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/model_train.py) - The module is designed to connect all the modules of the package and start training the neural network. Contains 5 functions that access certain modules. The output is the coefficients (weights) of the neural network.

[`model_evaluation.py`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/model_evaluation.py) - The module is designed to evaluate a neural network model using various metrics.

[`model.py`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/model.py) - The module contains the architecture of the model and a function for its training.

[`metrics.py`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/metrics.py) - The module contains Metrics for evaluating the effectiveness of classification of neural network models.

[`Data.py`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/Data.py) - The module is designed to prepare input data for a neural network (split into training, test and validation dataset).

[`Parser.py`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/Parser.py) - The module is designed for parsing html files of scientific articles from the data folder, as well as for parsing certain sites.

[`Text_processing.py`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/Text_processing.py) - This is a module designed for processing text in Russian and English (removing extra characters, reducing to lowercase, removing stopwords, removing punctuation, stemming).

[`weights.h5`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/weights.h5) - Coefficients of the trained neural network.

[`MCHS_2300.json`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/MCHS_2300.json) - Texts that relate to the structure of the Ministry of Emergency Situations (news about emergencies, terms of the Ministry of Emergency Situations).

[`topic_full.json`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/topic_full.json) - Contains texts related to a comprehensive topic. The text data was obtained using parsing sites.

## Data

   A sample of 4,300 texts was used as input, of which 2,800 texts were labeled 1:

   1) 2300 texts were obtained by parsing sites such as [rg.ru](https://rg.ru), [iz.ru](https://iz.ru) and others;
   2) 500 scientific articles were marked by an expert manually (scientific articles are intended for further development of the model, in particular, the classification of texts on 3 topics: Comprehensive topics, the topic of the Ministry of Emergency Situations, the topic "Disaster medicine in emergency situations", at the moment, a dataset is being formed on the topic "Disaster Medicine in Emergency situations" and a comprehensive topic is being finalized).

   The remaining 1,500 texts were obtained by parsing a scientific journal on comprehensive topics and were labeled 0. The data was divided into 3 data sets: training, validation and test. Data on scientific articles on the topic "Disaster Medicine in Emergency situations" can be found in [Scientific articles](https://github.com/Non1ce/Data_LSTM#readme).

## LSTM model

   Long Short-Term Memory~(LSTM) was introduced by [S. Hochreiter and J. Schmidhuber](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory) and developed by many research scientists.

   To deal with these problems Long Short-Term Memory (LSTM) is a special type of RNN that preserves long term dependency in a more effective way compared to the basic RNNs. This is particularly useful to overcome vanishing gradient problem. Although LSTM has a chain-like structure similar to RNN, LSTM uses multiple gates to carefully regulate the amount of information that will be allowed into each node state. Figure shows the basic cell of a LSTM model.
   
<p align="center">
  <img width="407" height="298" src="https://github.com/Non1ce/Image/blob/no_nice/LSTM/LSTM.png">
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
   
   

<p align="center">1. Embedding :arrow_right: 2. LSTM :arrow_right: 3. Dropout :arrow_right: 4. Dense + Sigmoid</p>

<html>
 <body>
  <p class="thumb" align="center">
   <img src="https://github.com/Non1ce/Image/blob/no_nice/LSTM/Model%20architecture.PNG" width="539" height="331" align="center"/>
  </p>
 </body>
</html>

## Evaluation of the model:


   The neural network was trained using the "accuracy" metric and the binary_cross entropy function. The accuracy of the model is 98.7%. The model was evaluated using the AUC metric. The AUC-ROC was constructed for the threshold values of the binary classification from 0 to 1 with a step of 0.0002. According to the following formula, the optimal threshold value was selected:
   
   <p align="center"> optimal = |TPR - (1-FPR)|, optimal -> min </p>


TPR = The number of true positives among all class labels that were defined as "positive".


FPR = The number of truly negative labels among all the class labels that were defined as "negative".


At each step, optimal was calculated and written to the dictionary, where the key was optimal, and the value was the threshold. Next, the smallest optimal was selected, which corresponded to the optimal threshold value.


<html>
 <body>
  <p class="thumb" align="center">
   <img src="https://github.com/Non1ce/Image/blob/no_nice/LSTM/AUC-ROC-LSTM.png" width="638" height="383">
  </p>
 </body>
</html>


## Usage
1. The model is located in [`model.py`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/model.py).
2. Run the module [`model_predict.py`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/model_predict.py) to predict the topic of a scientific article, if you need to train the model, you need to run a module [`model_train.py`](https://github.com/Non1ce/Neural_Network_Model/blob/no_nice/model_train.py).


## Version

Python 3.8

Tensorflow 2.4.1
