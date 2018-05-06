# =============================================================================
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils, plot_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
import time
# =============================================================================
# Functions:
# MLP(x, y)
# randomForest(x, y)
# RFE_SVM(x, y)
# RFE_AdaBoost(x, y)
# =============================================================================
def MLP(x, y):
    s_time = time.clock()
    #ATR, MOM, RSI, OBV
#    keep_features = ['ATR', 'MOM', 'RSI', 'OBV']
#    drop_features = list(set(list(x)).difference(keep_features))
#    x.drop(drop_features, axis=1, inplace=True)    
 #   x = x.as_matrix()
 #   y = y.as_matrix()
    num_features = np.array((5,10,15))
    num_classes = 3
    epochs = 1
    bs = 10
    np.random.seed(7)
    
    x_train,  x_test,  y_train,  y_test  = train_test_split(x, y, test_size=0.33)
        
 #   y_train  = np_utils.to_categorical(y_train, num_classes)
 #   y_test   = np_utils.to_categorical(y_test, num_classes)
 
    # create model
    model  = Sequential()    
    model.add( Dense(50, input_shape=(x_train.shape[1],), activation='tanh'))    
    model.add( Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Port Keras Framework into SK-Learn
    k_model  = KerasClassifier(build_fn=model, epochs=epochs, batch_size=bs, verbose=0)
    selector = RFE(k_model, n_features_to_select=5, step=5)
    selector.fit(x_train, y_train)
 
    # evaluate the model
 # ============================================================================
 #    scores = model.evaluate(x_test,  y_test)
 #    print("\n%s: %.2f%%" % (indicators.metrics_names[1], scores[1]*100))
 # ============================================================================
    
    e_time = time.clock()
    print('\n Total Time: ', e_time-s_time)
# =============================================================================
def randomForest(x, y):
    np.random.seed(7)
    max_depth = 2
  
    x_train,  x_test,  y_train,  y_test  = train_test_split(x, y, test_size=0.33)
    
#    RFC = RandomForestClassifier()
    RFC = RandomForestClassifier(max_depth=max_depth)
    RFC.fit(x_train, y_train)
    
    #Return the feature importances (the higher, the more important the feature).
    f_importances = (pd.DataFrame(RFC.feature_importances_)).transpose()
    f_names = list(x_train.columns.values)
    f_importances = pd.DataFrame([f_importances], columns=[f_names])
    f_importances.to_csv('../3_Deliverables/Final Paper/data/RFC_importances.csv', index=False)
# =============================================================================
def RFE_SVM(x, y):
    np.random.seed(7)
    
    x_train,  x_test,  y_train,  y_test  = train_test_split(x, y, test_size=0.33)
    
    clf = SVC(kernel='linear',probability=True, max_iter=200)
    selector = RFE(estimator=clf, n_features_to_select=5, step=5)
    
    s_time = time.clock()
    
    selector.fit(x_train, y_train.iloc[:,0])
    
    selectedFeatures = selector.ranking_
    acc = selector.score(x_test, y_test.iloc[:,0])
    
    e_time = time.clock()
    print('\n Total Time: ', e_time-s_time)
    print(' Accuracy:', acc)
    return selectedFeatures, acc
# =============================================================================
def RFE_AdaBoost(x, y):
    np.random.seed(7)
    
    x_train,  x_test,  y_train,  y_test  = train_test_split(x, y, test_size=0.33)
    
    clf = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0)
    selector = RFE(estimator=clf, n_features_to_select=5, step=5)
    
    s_time = time.clock()
    
    selector.fit(x_train, y_train.iloc[:,0])
    
    selectedFeatures = selector.ranking_
    acc = selector.score(x_test, y_test.iloc[:,0])
    
    e_time = time.clock()
    print('\n Total Time: ', e_time-s_time)
    print(' Accuracy:', acc)
    return selectedFeatures, acc