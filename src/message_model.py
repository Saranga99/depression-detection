import os
import re
from nltk.stem import WordNetLemmatizer
import pickle
from pathlib import Path

class MessageModel():
    class_name = os.path.basename(__file__)

    def __init__(self) -> None:
        self.wo = WordNetLemmatizer()
        filename="prediction.sav"
        self.mnb=pickle.load(open(filename,'rb'))
        filename="vectorizer.sav"
        self.vectorizer=pickle.load(open(filename,'rb'))


    def preprocess(self,data):
    #preprocess
        a = re.sub('[^a-zA-Z]',' ',data)
        a = a.lower()
        a = a.split()
        a = [self.wo.lemmatize(word) for word in a ]
        a = ' '.join(a)  
        return a


    def predict(self,message):
        a = self.preprocess(message)
        example_counts = self.vectorizer.transform([a])
        prediction = self.mnb.predict(example_counts)
        return prediction
