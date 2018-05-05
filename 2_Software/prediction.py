# =============================================================================
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils, plot_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier as ABC
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.feature_selection import RFE
import numpy as np
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
    x = x.as_matrix()
    y = y.as_matrix()
    num_features = np.array((5,10,15))
    num_classes = 3
    epochs = 1
    bs = 10
    np.random.seed(7)
    
    x_train,  x_test,  y_train,  y_test  = train_test_split(x, y, test_size=0.33)
        
#    y_train  = np_utils.to_categorical(y_train, num_classes)
#    y_test   = np_utils.to_categorical(y_test, num_classes)

    # create model
    model  = Sequential()    
    model.add( Dense(50, input_shape=(x_train.shape[1],), activation='tanh'))    
    model.add( Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    plot_model(model, to_file='../3_Deliverables/Final Paper/data/keras_model.png')
    
    # Port Keras Framework into SK-Learn
    k_model  = KerasClassifier(build_fn=model, epochs=epochs, batch_size=bs, verbose=0)
    temp = k_model
    selector = RFE(temp, step=1)
#    out = selector.fit(x_train, y_train[:,0])

    # evaluate the model
#    scores = model.evaluate(x_test,  y_test)
#    print("\n%s: %.2f%%" % (indicators.metrics_names[1], scores[1]*100))
    
    e_time = time.clock()
    print('Total Time: ', e_time-s_time)
# =============================================================================
def adaBoost(x, y):
    temp = 1
# =============================================================================
