from flask import Flask, request
from flask import render_template
import pandas as pd
import numpy as np
import json
#from sklearn.externals import joblib
import joblib
from datetime import datetime, timedelta
app = Flask(__name__)

#main page
@app.route('/')
def home():
	return render_template('home.html')

#capture the input and render the data for next html
@app.route('/analytics',methods=['POST','GET'])
def analytics():
    if request.method=='POST':
        #reading the template to parse through the model
        my_df = pd.read_csv("Resources/user_input_df.csv")
        result=request.form
        print("---------------------")
        print(result)
        print("---------------------")
        
        #get the date entered and converting to date format
        dt = result['dt']
        dt = datetime.strptime(dt, "%Y-%m-%d")
        
        #creating -/+3 days from the date entered by the user
        dt_more1 = dt + timedelta(days=1)
        dt_more2 = dt_more1 + timedelta(days=1)
        dt_more3 = dt_more2 + timedelta(days=1)

        dt_less1 = dt - timedelta(days=1)
        dt_less2 = dt_less1 - timedelta(days=1)
        dt_less3 = dt_less2 - timedelta(days=1)
        
        day = dt.day
    
        #save month from the date 
        month = dt.month
        
        #save day of the week from the date
        day_of_week = dt.weekday()
        
        #get the airline from user input
        airline = result['airline']
        
        #get the origin airport from user input
        origin = result['origin']
        
        #get the destination airport from user input
        destination = result['dest']
        print("-----------")
        print(destination)
        #get the departure hour from user input
        dep_hour = result['dep_hour']
        
        #create df and store the user input to the df
        columns = ['origin','destination','airline','day','month','dep_hour',]
        user_df = pd.DataFrame(columns=columns)
        print("before")
        print(user_df)
        
        user_df = user_df.append({'origin': origin, 'destination': destination, 'airline':airline,
                                 'day':str(day),'month':str(month), 'dep_hour':dep_hour}, ignore_index=True )
        
        print("after")
        print(user_df)
        
        user_df.to_json(orient='records', path_or_buf = 'static/data/main_page_input.json')
        
        #create airline df to store airlines and prediction delay value for each airline
        columns = ['airline_code','prob_delay']
        airline_delay_df = pd.DataFrame(columns=columns)
        
        #airline list for prediction
        airline_list = ['AS', 'AA', 'US', 'DL', 'NK', 'UA', 'HA', 'B6', 'OO', 'EV', 'MQ','F9', 'WN', 'VX']
        
        #loop to check delay prediction for each airline
        for i in airline_list:
            #print(i)
            #reading the user input template to parse through the model
            my_df =  pd.read_csv("Resources/user_input_df.csv")
            
            #dt = result['dt']
            #dt = datetime.strptime(dt, "%Y-%m-%d")
            #month = dt.month
            my_df['MONTH_'+str(month)] = 1

            #day_of_week = dt.weekday()
            my_df['DAY_OF_WEEK_'+str(day_of_week)] = 1

            #airline = result['airline']
            #updating the model format with each airline
            my_df['AIRLINE_'+str(i)] = 1

            #origin = result['origin']
            my_df['ORIGIN_'+str(origin)] = 1

            #destination = result['dest']
            my_df['DEST_'+str(destination)] = 1

            #dep_hour = result['dep_hour']
            my_df['DEP_HOUR_'+str(dep_hour)] = 1

            #print(my_df)
            
            #loading the model
            logmodel = joblib.load('Model/Airline_Delay_Predictition_model.pkl')
            
            #check prediction for the airline for each airline
            my_df['DELAY_YN'] = logmodel.predict_proba(my_df.drop(['DELAY_YN','ARRIVAL_DELAY'],axis=1))[:,1]
            my_df['DELAY_YN'] = my_df['DELAY_YN'].apply(lambda x:(x)*100,2)  
            probability_delay = (int(my_df['DELAY_YN'].values[0]*100))/100
            print("Probability of flight delay : " + str(probability_delay) + "%")
            
            #append the airline and delay prediction to the df
            airline_delay_df = airline_delay_df.append({'airline_code': i, 'prob_delay': probability_delay}, ignore_index=True )
        
        #save the airline prediction df 
#        airline_delay_df.to_json('Resources/airline_delay_prediction.json')
        airline_delay_df.to_json(orient='records',path_or_buf = 'static/data/airline_delay_prediction.json')
        
        print(airline_delay_df)

        #confirm the departure delay for 3+ and 3- days from the day entered by the user input
        #creating a df to store the days and prediction delay for each day
        columns = ['dep_days','prob_delay']
        dep_day_delay_df = pd.DataFrame(columns=columns)
        
        #array of total days
        days_input = [dt_less3,dt_less2,dt_less1,dt,dt_more1,dt_more2,dt_more3]
        
        #loop to check the delay prediction for each day
        for i in days_input:
            #print(i)
            
            #reading the template to parse through the model
            my_df =  pd.read_csv("Resources/user_input_df.csv")

#            dt = result['dt']
            #check the day of the week and month for each date
            dt = i

#            dt = datetime.strptime(dt, "%Y-%m-%d")
            month = dt.month
            #print(month)
            #update the user input df for the prediction model
            my_df['MONTH_'+str(month)] = 1

            day_of_week = dt.weekday()
            #adding 1 to the day of week to match the prediction model format
            day_of_week = day_of_week + 1
            #print(day_of_week)
            
            my_df['DAY_OF_WEEK_'+str(day_of_week)] = 1

            #airline = result['airline']
            my_df['AIRLINE_'+str(airline)] = 1

            #origin = result['origin']
            my_df['ORIGIN_'+str(origin)] = 1

            #destination = result['dest']
            my_df['DEST_'+str(destination)] = 1

            #dep_hour = result['dep_hour']
            my_df['DEP_HOUR_'+str(dep_hour)] = 1
            
            #converting the date to day to save for each day prediction
            dep_day = i.day
            #print(my_df)
            
            #loading the model
            logmodel = joblib.load('Model/Airline_Delay_Predictition_model.pkl')
            
            #predicting the delay probability
            my_df['DELAY_YN'] = logmodel.predict_proba(my_df.drop(['DELAY_YN','ARRIVAL_DELAY'],axis=1))[:,1]
            my_df['DELAY_YN'] = my_df['DELAY_YN'].apply(lambda x:(x)*100,2)  
            probability_delay = (int(my_df['DELAY_YN'].values[0]*100))/100
            
            #print("Probability of flight delay : " + str(probability_delay) + "%")
            #store the value to the df
            dep_day_delay_df = dep_day_delay_df.append({'dep_days': dep_day, 'prob_delay': probability_delay}, ignore_index=True)
        
        #save the per day departure prediction delay 
        dep_day_delay_df.to_json(orient='records',path_or_buf = 'static/data/day_delay_prediction.json')
        print(dep_day_delay_df)
    
    #calling the next result html
#    return render_template('score.html',airline=airline, origin=origin, dest=destination, 
#                           prob=probability_delay,airline_delay_df=airline_delay_df,dep_day_delay_df=dep_day_delay_df) 

    return render_template('analytics.html') 
          
if __name__ == '__main__':
	app.debug = True
	app.run()