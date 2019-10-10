import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
text=text.lower()
charac = sorted(list(set(text)))
char_to_n = {char:n for n, char in enumerate(charac)}
X = []
Y = []
length = len(text)
seq_length = 25000
for i in range(0, length-seq_length, 1):
     seq = text[i:i + seq_length]
     label =text[i + seq_length]
     X.append([char_to_n[char] for char in seq])
     Y.append(char_to_n[label])
X_mod = np.reshape(X, (len(X), seq_length, 1))
X_mod = X_mod / float(len(charac))
Y_mod = np_utils.to_categorical(Y)
model = Sequential()
model.add(LSTM(500, input_shape=(X_mod.shape[1], X_mod.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(500))
model.add(Dropout(0.2))
model.add(Dense(Y_mod.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X_mod, Y_mod, epochs=50, batch_size=100)
model.save_weights('text_generator_400_0.2_400_0.2_400_0.2_100.h5')
model.load_weights('text_generator_400_0.2_400_0.2_400_0.2_100.h5')
str_mapped = X[99]
fullstr = [n_to_char[value] for value in str_mapped]
# generating charac
for i in range(400):
    x = np.reshape(str_mapped,(1,len(str_mapped), 1))
    x = x / float(len(charac))
    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_char[value] for value in str_mapped]
    fullstr.append(n_to_char[pred_index])
    str_mapped.append(pred_index)
    str_mapped = str_mapped[1:len(str_mapped)]
    #combining text
txt=""
for char in fullstr:
    txt = txt+char
txt