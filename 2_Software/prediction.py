# =============================================================================
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.feature_selection import RFE
import numpy
import time
# =============================================================================
# Functions:
# MLP(x, y)
# =============================================================================
def MLP(x, y):
    s_time = time.clock()
    #ATR, MOM, RSI, OBV
    keep_features = ['ATR', 'MOM', 'RSI', 'OBV']
    drop_features = list(set(list(x)).difference(keep_features))
    x.drop(drop_features, axis=1, inplace=True)    
    
    num_features = np.array((5,10,15))
    num_classes = 3
    epochs = 1
    bs = 10
    numpy.random.seed(7)
    
    x_train,  x_test,  y_open_train,  y_open_test  = train_test_split(x, y, test_size=0.33)
        
#    y_open_train  = np_utils.to_categorical(y_open_train, num_classes)
#    y_open_test   = np_utils.to_categorical(y_open_test, num_classes)

    # create model
    model  = Sequential()    
    model.add( Dense(50, input_shape=(x_train.shape[1],), activation='tanh'))    
    model.add( Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Model Summary
#    indicators.summary()
#    model_high.summary()
#    model_low.summary()
#    model_close.summary()
    
    k_model  = KerasClassifier(build_fn=model, epochs=epochs, batch_size=bs, verbose=0)
    selector = k_model.fit(x_train, y_open_train)

    # RFE
#    num_features = np.zeros(1)    #Remove this line once this is functional
#    for i in range(len(num_features)):    
#        selector = RFE(k_model, num_features[i], step=1)
#        selector  = RFE(k_model, step=1)#        
#        selector  = selector.fit(x_train,   y_open_train)
#        print('OPEN: ',  selector_open.support_)
    # evaluate the model
#    scores = model.evaluate(x_test,  y_open_test)
#    print("\n%s: %.2f%%" % (indicators.metrics_names[1], scores_open[1]*100))
    
    e_time = time.clock()
    print('Total Time: ', e_time-s_time)
# =============================================================================
