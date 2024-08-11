from flask import Flask, request, render_template, redirect, url_for, flash
import os
import re
import pickle
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader

app = Flask(__name__)
app.secret_key = 'your_secret_key'  

model = pickle.load(open('clf.pkl', 'rb'))
tfid = pickle.load(open('tfidf.pkl', 'rb'))

category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf'}

def cleanResume(txt):
    cleanText = re.sub(r'http\S+', ' ', txt)  # Remove URLs
    cleanText = re.sub(r'RT|cc', ' ', cleanText)  # Remove RT and cc
    cleanText = re.sub(r'#\S+', ' ', cleanText)  # Remove hashtags
    cleanText = re.sub(r'@\S+', ' ', cleanText)  # Remove mentions
    cleanText = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', cleanText)  # Remove punctuation
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)  # Remove non-ASCII characters
    cleanText = re.sub(r'\s+', ' ', cleanText)  # Remove extra whitespace
    return cleanText

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('static', filename)
            file.save(file_path)

            if filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(file)
            elif filename.endswith('.txt'):
                resume_text = file.read().decode('utf-8')
            else:
                flash('Unsupported file type')
                return redirect(request.url)
            
            cleaned_resume = cleanResume(resume_text)

            input_feature = tfid.transform([cleaned_resume])
            prediction_id = model.predict(input_feature)[0]
            category_name = category_mapping.get(prediction_id, 'Unknown')
            
            return render_template('home.html', category_name=category_name)

        else:
            flash('Allowed file types are txt, pdf')
            return redirect(request.url)
    
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
