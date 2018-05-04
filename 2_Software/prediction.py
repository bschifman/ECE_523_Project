from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.feature_selection import RFE
import numpy
import time

def MLP(x_open, x_high, x_low, x_close, y_all):
    s_time = time.clock()
    num_features = np.array((5,10,15))
    num_classes = 3
    epochs = 1
    bs = 10
    numpy.random.seed(7)
    
    x_open_train,  x_open_test,  y_open_train,  y_open_test  = train_test_split(x_open, y_all.iloc[:,0], test_size=0.33)
    x_high_train,  x_high_test,  y_high_train,  y_high_test  = train_test_split(x_high, y_all.iloc[:,1], test_size=0.33)
    x_low_train,   x_low_test,   y_low_train,   y_low_test   = train_test_split(x_low, y_all.iloc[:,2], test_size=0.33)
    x_close_train, x_close_test, y_close_train, y_close_test = train_test_split(x_close, y_all.iloc[:,3], test_size=0.33)
    
#    y_open_train  = np_utils.to_categorical(y_open_train, num_classes)
#    y_open_test   = np_utils.to_categorical(y_open_test, num_classes)
#    y_high_train  = np_utils.to_categorical(y_high_train, num_classes)
#    y_high_test   = np_utils.to_categorical(y_high_test, num_classes)
#    y_low_train   = np_utils.to_categorical(y_low_train, num_classes)
#    y_low_test    = np_utils.to_categorical(y_low_test, num_classes)
#    y_close_train = np_utils.to_categorical(y_close_train, num_classes)
#    y_close_test  = np_utils.to_categorical(y_close_test, num_classes)
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
#    model_open.summary()
#    model_high.summary()
#    model_low.summary()
#    model_close.summary()
    
    k_model_open  = KerasClassifier(build_fn=model_open, epochs=epochs, batch_size=bs, verbose=0)
    selector_open  = k_model_open.fit(x_open_train,   y_open_train)
#    k_model_high  = KerasClassifier(build_fn=model_high, epochs=epochs, batch_size=bs)
#    k_model_low   = KerasClassifier(build_fn=model_low, epochs=epochs, batch_size=bs)
#    k_model_close = KerasClassifier(build_fn=model_close, epochs=epochs, batch_size=bs)
    # RFE
#    num_features = np.zeros(1)    #Remove this line once this is functional
#    for i in range(len(num_features)):    
##        selector = RFE(k_model_open, num_features[i], step=1)
#        selector_open  = RFE(k_model_open, step=1)
#        selector_high  = RFE(k_model_open, step=1)
#        selector_low   = RFE(k_model_open, step=1)
#        selector_close = RFE(k_model_open, step=1)
#        
#        selector_open  = selector_open.fit(x_open_train,   y_open_train)
#        selector_high  = selector_high.fit(x_high_train,   y_high_train)
#        selector_low   = selector_low.fit(x_low_train,     y_low_train)
#        selector_close = selector_close.fit(x_close_train, y_close_train)
#         
#        print('OPEN: ',  selector_open.support_)
#        print('HIGH: ',  selector_high.support_)
#        print('LOW: ',   selector_low.support_)
#        print('CLOSE: ', selector_close.support_)
    # evaluate the model
#    scores_open  = model_open.evaluate(  X_open_test,  y_open_test)
#    print("\n%s: %.2f%%" % (model_open.metrics_names[1], scores_open[1]*100))
#    
#    scores_high  = model_high.evaluate(  X_high_test,  y_high_test)
#    print("\n%s: %.2f%%" % (model_high.metrics_names[1], scores_high[1]*100))
#    
#    scores_low   = model_low.evaluate(    X_low_test,   y_low_test)
#    print("\n%s: %.2f%%" % (model_low.metrics_names[1], scores_low[1]*100))
#    
#    scores_close = model_close.evaluate(X_close_test, y_close_test)
#    print("\n%s: %.2f%%" % (model_close.metrics_names[1], scores_close[1]*100))
    
    e_time = time.clock()
    print('Total Time: ', e_time-s_time)

#ATR, MOM, RSI, OBV
keep_features = ['ATR', 'MOM', 'RSI', 'OBV']
drop_features = list(set(list(x_close)).difference(keep_features))
x_open.drop(drop_features, axis=1, inplace=True)
x_high.drop(drop_features, axis=1, inplace=True)
x_low.drop(drop_features, axis=1, inplace=True)
x_close.drop(drop_features, axis=1, inplace=True)

MLP(x_open, x_high, x_low, x_close, y_all)
