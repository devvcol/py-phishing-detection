import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

emails = pd.read_csv('your_emails.csv?')

phish_map = {'Safe Email': 0, 'Phishing Email': 1}

emails['Email Type'] = emails['Email Type'].map(phish_map) # converts the eMail type into a number for ML processing.

#### THIS CSV CLEANING IS ONLY TESTED ON THE SPECIFIC PHISHING_EMAIL.CSV FILE ####
#### CHANGE IT DEPENDING ON YOUR DATA                                         ####

def clean_text(text: str): # cleans up csv text for ML processing
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)   # no html
        text = re.sub(r'[^\w\s]', '', text) # no punctuation
        text = re.sub(r'\n|\r', ' ', text)  # no new lines/\n
    else:
        text = ''                           # pandas floats removed
    return text

emails['cleaned_text'] = emails['Email Text'].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(emails['cleaned_text'])

y = emails['Email Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ml_model = LogisticRegression()
ml_model.fit(X_train, y_train)

y_predict = ml_model.predict(X_test)

joblib.dump(ml_model, 'phishing-detection-model.pk')
joblib.dump(vectorizer, 'phishing-detection-vectorizer.pk')

