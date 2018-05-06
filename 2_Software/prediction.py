# =============================================================================
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils, plot_model
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
# =============================================================================
# Functions:
# MLP(x, y)
# randomForest(x, y)
# =============================================================================
def MLP(x, y):
    s_time = time.clock()
    num_classes = 3
    epochs = 20
    bs = 10
    np.random.seed(7)
    
    x_train,  x_test,  y_train,  y_test  = train_test_split(x, y, test_size=0.33)
    
    y_train  = np_utils.to_categorical(y_train, num_classes)
    y_test   = np_utils.to_categorical(y_test, num_classes)
    
    # create model
    model  = Sequential()
    model.add( Dense(50, input_shape=(x_train.shape[1],), activation='tanh'))
    model.add( Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(x_train, y_train,  epochs=epochs, batch_size=bs, verbose=0)
    
    acc = model.evaluate(x_test, y_test)[1]

    # evaluate the model
 # ============================================================================
 #    scores = model.evaluate(x_test,  y_test)
 #    print("\n%s: %.2f%%" % (indicators.metrics_names[1], scores[1]*100))
 # ============================================================================
    
    e_time = time.clock()
    print('\n Total Time: ', e_time-s_time)
    return model, acc
# =============================================================================
def randomForest(x, y, switch, t):
    np.random.seed(7)
    numFolds = 3
    k_fold = KFold(n_splits=numFolds)
    score = 0
    RFC = RandomForestClassifier()

    
    x_arr = x.as_matrix()
    y_arr = y.as_matrix()
    
    for train_index, test_index in k_fold.split(x_arr):
        x_train, x_test = x_arr[train_index], x_arr[test_index]
        y_train, y_test = y_arr[train_index], y_arr[test_index]
        RFC.fit(x_train, y_train[:,0])
        score += RFC.score(x_test, y_test[:,0])
    score /= numFolds
    print(numFolds, 'Fold RFC Score: ', np.round(score,3))
    
    #Return the feature importances (the higher, the more important the feature).
    if(switch):
        f_names = list(x.columns.values)
        f_importances = pd.DataFrame(np.round(RFC.feature_importances_, 2)).transpose()
        f_importances.columns = f_names
        plot = f_importances.plot.bar(figsize=(22.0, 14.0))
        plot.set_xlabel('Features', fontsize=24)
        plot.set_ylabel('Feature Importance (%)', fontsize=24)
        plot.tick_params(labelsize=24)
        fig  = plot.get_figure()
        fig.savefig('../3_Deliverables/Final Paper/data/RFC_importancesT'+str(t)+'.png')
        f_importances.to_csv('../3_Deliverables/Final Paper/data/RFC_importancesT'+str(t)+'.csv', index=False)
    else:
        return(score)
# =============================================================================
def pca(x, y, t):
    pca_out = np.zeros((x.shape[1]-1,3))
    pca_columns = ['Principal Components', 'Score', 'Time']
    i = 1
    while(i < x.shape[1]):
        pca = PCA(n_components=i)
        s = time.clock()
        x_new = pd.DataFrame(pca.fit_transform(x))    
        score = randomForest(x_new, y, switch=0, t=t)
        pca_out[i-1,0] = i
        pca_out[i-1,1] = round(score,2)
        pca_out[i-1,2] = round(time.clock()-s,2)
        i += 1
        
    fig, ax1 = plt.subplots()    
    principalComponents = pca_out[:,0]
    Times = pca_out[:,2]
    scores = pca_out[:,1]
    
    ax1.plot(principalComponents, scores, 'b')
    ax1.set_xlabel('Principal Components')
    ax1.set_ylabel('Accuracy %', color='b')
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()    

    ax2.plot(principalComponents, Times, 'r')
    ax2.set_ylabel('Time (s)', color='r')
    ax2.tick_params('y', colors='r')    
    fig.tight_layout()
    
    pca_out = pd.DataFrame(pca_out)
    pca_out.columns = pca_columns
    pca_out.to_csv('../3_Deliverables/Final Paper/data/PCAT'+str(t)+'.csv', index=False)
    fig.savefig('../3_Deliverables/Final Paper/data/PCAT'+str(t)+'.png')
# =============================================================================