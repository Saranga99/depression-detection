import os
import re
from nltk.stem import WordNetLemmatizer
import pickle

class MessageModel():
    class_name = os.path.basename(__file__)

    def __init__(self) -> None:
        self.wo = WordNetLemmatizer()
        self.mnb=pickle.load(open("models/message_model/prediction.sav",'rb'))
        self.vectorizer=pickle.load(open("models/message_model/vectorizer.sav",'rb'))


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
        if prediction[0]==0:
            return "positive"
        elif prediction[0]==1:
            return "depressive"
        
