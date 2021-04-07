# You can write code above the if-main block.
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import argparse
from matplotlib import pyplot as plt
import numpy as np


if __name__ == "__main__":
    # You should not modify this part.

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training_data.csv", help="input training data file name")
    parser.add_argument("--testing", default="testing_data.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    #training_data = load_data(args.training)
    training_data= pd.read_csv(args.training,squeeze=True) 
    #trader = Trader()
    #trader.train(training_data)

    testing_data = pd.read_csv(args.testing,squeeze=True) 
    
    
    p=6
    d=2
    q=1
   
    #open
    train=[x for x in training_data['open'].values]
    test=testing_data['open'].values  
    predictions=list()
   
    for t in range(0,len(test),20):
        model = ARIMA(train, order=(p,d,q))     
        model_fit = model.fit()        
        output = model_fit.forecast(20)
        
        
        if (t+20)>len(test):
            yhat=output[0:len(test)-t]
        else:
            yhat = output[0:20]
        for i in range(len(yhat)):
            predictions.append(yhat[i])
        obs = test[t:t+20]
        for i in range(len(obs)):
            train.append(obs[i])
        #for i in range(len(yhat)):
         #   print('predicted=%f, expected=%f' % (yhat[i], obs[i]))
 
    
    error = mean_squared_error(test, predictions)
    print('open Test MSE: %.3f' % error)
    
    
    
    model = ARIMA(train, order=(p,d,q)) 
    model_fit = model.fit()
    openoutput = model_fit.forecast(20)
    
    	
    
    #high
    train=[x for x in training_data['high'].values]
    test=testing_data['high'].values     
    predictions=list()

    for t in range(0,len(test),20):
        model = ARIMA(train, order=(p,d,q))    
        model_fit = model.fit()        
        output = model_fit.forecast(20)
        if (t+20)>len(test):
            yhat=output[0:len(test)-t]
        else:
            yhat = output[0:20]
        for i in range(len(yhat)):
            predictions.append(yhat[i])
        obs = test[t:t+20]
        for i in range(len(obs)):
            train.append(obs[i])
        #print('%d predicted=%f, expected=%f' % (t,yhat[0], obs[0]))
 
    error = mean_squared_error(test, predictions)
    print('high Test MSE: %.3f' % error)
      
    model = ARIMA(train, order=(p,d,q)) 
    model_fit = model.fit()
    highoutput = model_fit.forecast(20)        
    
    #low
    train=[x for x in training_data['low'].values]
    test=testing_data['low'].values     
    predictions=list()

    for t in range(0,len(test),20):
        model = ARIMA(train, order=(p,d,q))    
        model_fit = model.fit()        
        output = model_fit.forecast(20)
        if (t+20)>len(test):
            yhat=output[0:len(test)-t]
        else:
            yhat = output[0:20]
        for i in range(len(yhat)):
            predictions.append(yhat[i])
        obs = test[t:t+20]
        for i in range(len(obs)):
            train.append(obs[i])
        #print('%d predicted=%f, expected=%f' % (t,yhat[0], obs[0]))
 
    error = mean_squared_error(test, predictions)
    print('low Test MSE: %.3f' % error)
      
    model = ARIMA(train, order=(p,d,q)) 
    model_fit = model.fit()
    lowoutput = model_fit.forecast(20)
    
    #close
    train=[x for x in training_data['close'].values]
    test=testing_data['close'].values     
    predictions=list()

    for t in range(0,len(test),20):
        model = ARIMA(train, order=(p,d,q))    
        model_fit = model.fit()        
        output = model_fit.forecast(20)
        if (t+20)>len(test):
            yhat=output[0:len(test)-t]
        else:
            yhat = output[0:20]
        for i in range(len(yhat)):
            predictions.append(yhat[i])
        obs = test[t:t+20]
        for i in range(len(obs)):
            train.append(obs[i])
        #print('%d predicted=%f, expected=%f' % (t,yhat[0], obs[0]))
 
    error = mean_squared_error(test, predictions)
    print('close Test MSE: %.3f' % error)
      
    model = ARIMA(train, order=(p,d,q)) 
    model_fit = model.fit()
    closeoutput = model_fit.forecast(20)
    
    def change(a,b):
        t=a
        a=b
        b=t
        return a,b
    
    for i in range(20):
        if openoutput[i]>highoutput[i]:
            openoutput[i],highoutput[i]=change(openoutput[i],highoutput[i])
        if closeoutput[i]>highoutput[i]:
            closeoutput[i],highoutput[i]=change(closeoutput[i],highoutput[i])
        if openoutput[i]<lowoutput[i]:
            openoutput[i],lowhoutput[i]=change(openoutput[i],lowoutput[i])
        if closeoutput[i]<lowoutput[i]:
            closeoutput[i],lowoutput[i]=change(closeoutput[i],lowoutput[i])
    
    frame={
        'open':openoutput,
        'high':highoutput,
        'low':lowoutput,
        'close':closeoutput
    }
    
    output=pd.DataFrame(frame)
    output.to_csv(args.output)
    #plt.plot(predictions,'r',label='predict')
    #plt.plot(test,'g',label='real')   
    #plt.legend()
    #plt.show()
    
    
    #with open(args.output, "w") as output_file:
        #for row in testing_data:
            # We will perform your action as the open price in the next day.
            #action = trader.predict_action(row)
            #output_file.write(action)

            # this is your option, you can leave it empty.
            #trader.re_training()
