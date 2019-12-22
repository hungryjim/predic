# get all the imports
from datetime import datetime
import pandas as pd
from pandas import concat,DataFrame
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, BatchNormalization
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import PolynomialFeatures
from prepare_data_for_modal import prepare_data, series_to_supervised


def train_model(stock_num):
    train_x = None
    train_y = None
    test_x = None
    test_y = None
    pre_stock_data = None
    #combine the data
    for i in np.arange(8):
        stock_number = i + 1
        pre_stock_data, stock, news = prepare_data(str(stock_number))
        # Polymerise Stock data¶
        # since we are using only one data coloumn form stock
        # # to make polynomial feature set
        poly = PolynomialFeatures(degree=2)
        stock = poly.fit_transform(stock)
        # Split the data into train and test data¶
        if stock_number == 7:
            train_x1 = np.hstack([stock[:400], news[:400]])
            train_y1 = pre_stock_data.iloc[:406, 9:].values
        else:
            train_x1 = np.hstack([stock[:400], news[:400]])
            train_y1 = pre_stock_data.iloc[:400, 9:].values
        if train_x is None:
            train_x = train_x1
            train_y = train_y1
        else:
            train_x = np.vstack((train_x, train_x1))
            train_y = np.vstack((train_y, train_y1))
        #这个是训练第几只股票
        if stock_num == stock_number:
            test_x = np.hstack([stock[400:], news[400:]])
            test_y = pre_stock_data.iloc[400:, 9:].values

    train_y = series_to_supervised(train_y, 6, 1)

    # reshape the data acording to the lstm
    train_x=train_x.reshape(-1,1,7)
    test_x=test_x.reshape(-1,1,7)



    #Made on the basis of assumption made on data analysis
    # make model
    model=Sequential()
    # layer 1
    model.add(LSTM(128,input_shape=(1,train_x.shape[2:][0]),activation='relu', return_sequences=True ))
    model.add(Dropout(0.2))#全连接相关，尝试丢掉一些输出
    model.add(BatchNormalization())
    # layer 2
    model.add(LSTM(128, return_sequences=True ))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    # layer 3
    model.add(LSTM(128, activation='relu',return_sequences=False ))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    # layer 4
    model.add(Dense(100,activation='relu' ))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    # layer 5
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    # layer 6
    model.add(Dense(50,activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    # final output in 1
    model.add(Dense(7))

    # make optimiser
    opt=keras.optimizers.RMSprop(lr=0.001,rho=0.9, decay=0.0)
    # compile the model
    model.compile(optimizer=opt, loss='mean_squared_error',metrics=['accuracy'])

    # to log the data for tensorboard
    time=datetime.now()
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./log/stock_01_baseline'+str(time), write_graph=True)

    # for the model checkpoints
    filepath="./log/2_baseline_weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint,tbCallBack]
    model.summary()
    model.fit(
        train_x
        , train_y
        , epochs=40
        , batch_size=40
        , verbose=1
        , validation_split=0.1
        , callbacks=callbacks_list
    )

    predict=model.predict(test_x)

    #Plot the data that is predicted by modelc
    #matplotlib inline
    fig=plt.figure(figsize=(15, 8), dpi= 80, facecolor='w', edgecolor='k')
    # index = pd.date_range(start = pre_stock_data['Date'][0], end = pre_stock_data['Date'][406], freq = "D")
    predict_Date = ['2015/12/21', '2015/12/22', '2015/12/23', '2015/12/24', '2015/12/28', '2015/12/29', '2015/12/30']
    dataArr = np.append(pre_stock_data['Date'], predict_Date)
    plt.plot(dataArr[400:496] ,np.squeeze(test_y,axis=1), label='test_y')
    plt.plot(dataArr[407:], np.round(np.squeeze(predict[:, :1], axis=1), 1), label='predict_test')
    plt.legend()
    _ = plt.ylim()

train_model(1)