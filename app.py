#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle #To read model.pkl file

app = Flask(__name__) # Create Flask app
model = pickle.load(open('model.pkl', 'rb')) #Loading the pickle


#Home page
@app.route('/') #By default your route page which is just '/' will render the template index.html
# index.html has 3 fields 
#<input type="text" name="experience" placeholder="Experience" required="required" />
#<input type="text" name="test_score" placeholder="Test Score" required="required" />
#<input type="text" name="interview_score" placeholder="Interview Score" required="required" />
def home():
    return render_template("index.html") #Home page where you have the fields that you want from that fields you would be able to put your input when you click predict button you get the output


#This is just like Web API: when I say '/predict' it will go and hit the function predict 
@app.route('/predict',methods=['POST']) #I have created '/predict' which is basically a post method where in I will be providing features to my model.pkl file so that model takes those inputs and give us some output that is what I will be doing in this particular function.
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()] #It will take inputs from all the forms.Here I have 3 text fields if you use this request library you will be able to take all values from text field and store it in your feature called int_features
    final_features = [np.array(int_features)] #converting this into an array
    prediction = model.predict(final_features) 

    output = round(prediction[0], 2) #Finding out the output

    return render_template("index.html", prediction_text='Employee Salary should be $ {}'.format(output)) #This prediction text will get replaced over  {{ prediction_text }} in html code inside the braces




if __name__ == "__main__": #Main function to run this complete flask
    app.run(debug=True)

