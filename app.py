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
    
    
  
   
    #open
    trainhigh=[x for x in training_data['high'].values]
    testhigh=testing_data['high'].values  
    trainlow=[x for x in training_data['low'].values]
    testlow=testing_data['low'].values  
   
    action=np.zeros(len(testhigh)-1) 
    slot=0
    
    predictions=list()
    predictions2=list()
    old=trainhigh[len(trainhigh)-1]
    old2=trainhigh[len(trainlow)-1]
    for t in range(len(testhigh)-1):
        model = ARIMA(trainhigh, order=(3,1,1))     
        model_fit = model.fit()
        output = model_fit.forecast(2)
        yhat = output[0]
        predictions.append(yhat)
        if yhat>old and yhat>output[1] and slot!=-1 :
            action[t]=-1
            slot=slot-1
        obs = testhigh[t]
        old=obs
        trainhigh.append(obs)
       #print('high predicted=%f, expected=%f' % (yhat, obs))
        
        model = ARIMA(trainlow, order=(5,1,1))     
        model_fit = model.fit()
        output = model_fit.forecast(2)
        yhat = output[0]
        predictions2.append(yhat)
        if yhat<old2 and yhat<output[1] and slot!=1:
            action[t]=1
            slot=slot+1
        obs = testlow[t]
        old2=obs
        trainlow.append(obs)
        #print('low predicted=%f, expected=%f\n' % (yhat, obs))
 
    error = mean_squared_error(testhigh[:len(testhigh)-1], predictions)
    print('Test MSE: %.3f' % error)
    error = mean_squared_error(testlow[:len(testlow)-1], predictions2)
    print('Test MSE: %.3f' % error)


    
    frame={
        
        'action':action
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
