#Import Dependentias
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
import re
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

#Importing the SMS dataset
msg= pd.read_csv('SMSSpamCollection', sep='\t',names=["label", "message"])
msg.drop_duplicates(inplace=True)
msg.reset_index(drop=True,inplace=True)

#Cleaning the data and Preprocessing
corpus = []
ps = PorterStemmer()
for i in range(0, len(msg)):
    message = re.sub('[^a-zA-Z]', ' ', msg['message'][i])  #Cleaning special character from the message
    message = message.lower()    #Converting the entire message into lower case
    words = message.split()    #spliting each and every sentences where iam getting list of words.
    words = [word for word in words if word not in set(stopwords.words('english'))] #Removing the stop words
    words = [ps.stem(word) for word in words] #Stemming the words
    message = ' '.join(words) #Joining the stemmed words
    corpus.append(message) #Building a corpus of messages

#Save corpus for use in deployment
file_name = "corpus.pkl"
pickle.dump(corpus, open(file_name, 'wb'))

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

#Extracting dependent variable from the dataset
y=pd.get_dummies(msg['label'])
y=y.iloc[:,1].values

#Train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
classifier= MultinomialNB()
classifier.fit(X_train, y_train)

#Save Model
file_name = "Spam_sms_prediction.pkl"
pickle.dump(classifier, open(file_name, 'wb'))