from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy

def MLP(X, y):
    num_classes = 3
    epochs = 150
    numpy.random.seed(7)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test  = np_utils.to_categorical(y_test, num_classes)
    # load pima price dataset
    # split into input (X) and output (Y) variables
    # create model
    model = Sequential()
    model.add(Dense(50, input_shape=(X.shape[1],), activation='tanh'))
#    model.add(Dense(8, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Model Summary
    model.summary()
    # Fit the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=10)
    # evaluate the model
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
     
X = data_open_norm['JPM']
y = y['JPM']

MLP(X, y)