import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
import re
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Dropout
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


train_in = open('train.tsv')
test_in = open('test.tsv')
traindata = pd.read_csv(train_in, delimiter='\t')
testdata = pd.read_csv(test_in, delimiter='\t')

traindata['Phrase'] = traindata['Phrase'].apply(lambda x: x.lower())
traindata['Phrase'] = traindata['Phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

testdata['Phrase'] = testdata['Phrase'].apply(lambda x: x.lower())
testdata['Phrase'] = testdata['Phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

train_sentences = traindata['Phrase'].values
train_labels = traindata['Sentiment'].values

train_tokenizer = Tokenizer()
train_tokenizer.fit_on_texts(train_sentences)
train_sentences = train_tokenizer.texts_to_matrix(train_sentences)
train_sentences = pad_sequences(train_sentences, maxlen=300)
vocab_size = len(train_tokenizer.word_index) + 1

le = preprocessing.LabelEncoder()
train_labels = le.fit_transform(train_labels)
train_labels = to_categorical(train_labels)

test_sentences = testdata["Phrase"]
test_tokenizer = Tokenizer()
test_tokenizer.fit_on_texts(test_sentences)
test_sentences = test_tokenizer.texts_to_matrix(test_sentences)
test_sentences = pad_sequences(test_sentences, maxlen=300)

X_train, X_test, y_train, y_test = train_test_split(train_sentences, train_labels, test_size=0.25, random_state=1000)


def createModel():
    model = Sequential()
    model.add(Embedding(vocab_size, 300, input_length=train_sentences.shape[1]))
    model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


gridModel = KerasClassifier(build_fn=createModel, verbose=0)
batch_size = [128, 256, 500]
epochs = [1, 2, 3]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=gridModel, param_grid=param_grid)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#model = createModel()
#model.fit(X_train, y_train, epochs = 1, batch_size=500, verbose = 1)
#score,acc = model.evaluate(X_test,y_test,verbose=1,batch_size=500)
#print(score)
#print(acc)
#print(model.metrics_names)