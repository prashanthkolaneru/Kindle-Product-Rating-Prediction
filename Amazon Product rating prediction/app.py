#step -1 # Importing flask module in the project is mandatory 
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from nltk import stem
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vt = TfidfVectorizer(vocabulary=None,tokenizer=None)




stemmer = stem.SnowballStemmer('english')
stopwords = set(stopwords.words('english'))

def alternative_review_messages(msg):
    # converting messages to lowercase
    msg = msg.lower()
    # removing stopwords
    msg = [word for word in msg.split() if word not in stopwords]
    # using a stemmer
    msg = " ".join([stemmer.stem(word) for word in msg])
    return msg

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

#Step -2 Flask constructor takes the name of  
# current module (__name__) as argument.app = Flask(__name__)

app = Flask(__name__)

#Step -3 Load Trained  Model
model = pickle.load(open('Rating.pkl', 'rb'))
vec  = pickle.load(open('Rating_t.pkl', 'rb'))

# Step -4 The route() function of the Flask class is a decorator,  
# which tells the application which URL should call  
# the associated function


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    msg=request.form['msg']       
    #Data['Review'][0]=msg
    
    
    Data={'Text':[msg]}
   
    
    df1 = pd.DataFrame(Data)
    df1["Text"] = df1['Text'].apply(alternative_review_messages)
    df1["Text"] = df1["Text"].str.replace(r'\d+','')
    df1["Text"] = df1["Text"].str.replace(r'\s+',' ')
    df1["Text"] = df1["Text"].str.replace(r'\[[0-9]*\]',' ')
    df1["Text"] = df1['Text'].apply(remove_punctuations)

    ve = vec.transform(df1["Text"])
   
    
   
    output = model.predict(ve)
    

    
    return render_template('index.html', prediction_text=' Product Rating: {}'.format(output))


# main driver function
 # run() method of Flask class runs the application  
    # on the local development server.
if __name__ == "__main__":
    app.run(debug=True)

