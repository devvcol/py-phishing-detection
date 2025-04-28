from flask import Flask, request, render_template
import joblib
import re

ml_model = joblib.load('phishing-detection-model.pk')
vectorizer = joblib.load('phishing-detection-vectorizer.pk')

app = Flask(__name__)

def process_text(email: str):
    if isinstance(email, str):
        email = email.lower()
        email = re.sub(r'<.*?>', '', email)
        email = re.sub(r'[^\w\s]', '', email)
        email = re.sub(r'\n|\r', ' ', email)
    else:
        email = ''
    return email


@app.route('/', methods=['GET', 'POST'])
def main():
    prediction = None
    probability = None

    if request.method == "POST":
        email_text = request.form['email_text']
        processed_email = process_text(email_text)
        vector_text = vectorizer.transform([processed_email])
        prediction = ml_model.predict(vector_text)[0]
        probability = (ml_model.predict_proba(vector_text)[0][1]) * 100

        result = 'Phishing' if prediction == 1 else 'Safe'
        return render_template('main.html', result=result, probability=probability)
    return render_template('main.html', result=prediction, probability = probability)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5500)