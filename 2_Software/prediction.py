from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy
import time

def MLP(x_open, x_high, x_low, x_close, y_all):
    s_time = time.clock()
    num_classes = 3
    epochs = 3
    bs = 10
    numpy.random.seed(7)
    
    X_open_train, X_open_test, y_open_train, y_open_test     = train_test_split(x_open, y_all.iloc[:,0], test_size=0.33)
    X_high_train, X_high_test, y_high_train, y_high_test     = train_test_split(x_high, y_all.iloc[:,1], test_size=0.33)
    X_low_train, X_low_test, y_low_train, y_low_test         = train_test_split(x_low, y_all.iloc[:,2], test_size=0.33)
    X_close_train, X_close_test, y_close_train, y_close_test = train_test_split(x_close, y_all.iloc[:,3], test_size=0.33)
    
    y_open_train  = np_utils.to_categorical(y_open_train, num_classes)
    y_open_test   = np_utils.to_categorical(y_open_test, num_classes)
    y_high_train  = np_utils.to_categorical(y_high_train, num_classes)
    y_high_test   = np_utils.to_categorical(y_high_test, num_classes)
    y_low_train   = np_utils.to_categorical(y_low_train, num_classes)
    y_low_test    = np_utils.to_categorical(y_low_test, num_classes)
    y_close_train = np_utils.to_categorical(y_close_train, num_classes)
    y_close_test  = np_utils.to_categorical(y_close_test, num_classes)
    # split into input (X) and output (Y) variables
    # create model
    model_open  = Sequential()
    model_high  = Sequential()
    model_low   = Sequential()
    model_close = Sequential()
    
    model_open.add( Dense(50, input_shape=(x_open.shape[1],), activation='tanh'))
    model_high.add( Dense(50, input_shape=(x_high.shape[1],), activation='tanh'))
    model_low.add(  Dense(50, input_shape=(x_low.shape[1],), activation='tanh'))
    model_close.add(Dense(50, input_shape=(x_close.shape[1],), activation='tanh'))
    
    model_open.add( Dense(num_classes, activation='softmax'))
    model_high.add( Dense(num_classes, activation='softmax'))
    model_low.add(  Dense(num_classes, activation='softmax'))
    model_close.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model_open.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_high.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_low.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_close.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Model Summary
    model_open.summary()
    model_high.summary()
    model_low.summary()
    model_close.summary()
    # Fit the model
    model_open.fit(  X_open_train,  y_open_train, epochs=epochs, batch_size=bs)
    model_high.fit(  X_high_train,  y_high_train, epochs=epochs, batch_size=bs)
    model_low.fit(    X_low_train,   y_low_train, epochs=epochs, batch_size=bs)
    model_close.fit(X_close_train, y_close_train, epochs=epochs, batch_size=bs)
    # evaluate the model
    scores_open  = model_open.evaluate(  X_open_test,  y_open_test)
    print("\n%s: %.2f%%" % (model_open.metrics_names[1], scores_open[1]*100))
    
    scores_high  = model_high.evaluate(  X_high_test,  y_high_test)
    print("\n%s: %.2f%%" % (model_high.metrics_names[1], scores_high[1]*100))
    
    scores_low   = model_low.evaluate(    X_low_test,   y_low_test)
    print("\n%s: %.2f%%" % (model_low.metrics_names[1], scores_low[1]*100))
    
    scores_close = model_close.evaluate(X_close_test, y_close_test)
    print("\n%s: %.2f%%" % (model_close.metrics_names[1], scores_close[1]*100))
    
    e_time = time.clock()
    print('Total Time: ', s_time-e_time)

MLP(x_open, x_high, x_low, x_close, y_all)
