#Import main library
#from pyexpat import model
import pandas as pd
import numpy as np
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords

#Import Flask modules
from flask import Flask, request, render_template, redirect, url_for

#Import pickle to save our regression model
import pickle

import os
from os.path import join, dirname, realpath

#Initialize Flask and set the template folder to "template"
app = Flask(__name__, template_folder='template')
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#Open our model
model = pickle.load(open('model.pkl', 'rb'))
tfidfv = pickle.load(open('wordvector.mkl', 'rb'))
#create our "home" route using the "index.html" page


@app.route('/')
def home():
    return render_template('index2.html')

#Set a post method to yield predictions on page


@app.route('/', methods=['POST'])
def predict():
    #model = pickle.load(open('model.pkl','rb'))
    #tfidf = pickle.load(open('wordvector.mkl','rb'))
    #obtain all form values and place them in an array, convert into integers
    features = tfidfv.transform(request.form.values())
    #Combine them all into a final numpy array
    #final_features = [np.array(features)]
    #predict the price given the values inputted by user
    prediction = model.predict(features)

    #Round the output to 2 decimal places
    #output = round(prediction[0], 2)

    #If the output is negative, the values entered are unreasonable to the context of the application
    #If the output is greater than 0, return prediction
    """
    if output < 0:
        return render_template('index.html', prediction_text = "Predicted Price is negative, values entered not reasonable")
    elif output >= 0:
        """
    return render_template('index2.html', prediction_text='The review is: ${}'.format(prediction))

#Set the model


@app.route('/csv', methods=['POST'])
def feed():
    print(request.files)
    stopword_es = stopwords.words('spanish')
    #print(stopword_es)
    uploaded_file = request.files['filename']
    stopw = request.form.get('swselector')
    if uploaded_file.filename != '':
        try:
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            #print("Path: ",file_path)
            # set the file path
            uploaded_file.save(file_path)
            #print("UploadedFile:", uploaded_file.filename)
            #redirect(url_for('index'))
            #print("About to read")
            dfreview = pd.read_csv(file_path)
            #print("File Read")
            #data_top = dfreview.head()
            # display
            print("len: ", len(dfreview.columns))
            if len(dfreview.columns) != 2:
                raise Exception('Not the right number of columns')

            review = dfreview.columns[0]
            sentiment = dfreview.columns[1]
            print("Columnas guardadas")
            #for col in dfreview.columns:
            #    print(col)
            print("Review: ", review)
            print("Sentiment: ", sentiment)
            #sentpos, sentneg = "", ""

            #print(dfreview.value_counts(sentiment))
            #sentlist = dfreview[sentiment].tolist()
            #print(sentlist)
            #print("")
            sentvals = dfreview[sentiment].unique()
            print("Sentvals seteados")
            print(sentvals)
            if len(sentvals) < 2:
                raise Exception('Not the right number of sentiments')

            sentpos = sentvals[0]
            sentneg = sentvals[1]
            #slices o particiones para crear un set desbalanceado
            dfpositivos = dfreview[dfreview[sentiment] == sentpos][:5000]
            dfnegativos = dfreview[dfreview[sentiment] == sentneg][:5000]
            #print("Los negas: ",dfnegativos)
            dfreviewdes = pd.concat([dfpositivos, dfnegativos])
            for i in range(2, len(sentvals)):
                dfNS = dfreview[dfreview[sentiment] == sentvals[i]][:5000]
                #print("Los ",sentvals[i],": ",dfNS)
                dfreviewdes = pd.concat([dfreviewdes, dfNS])
            #Se hace un undersampling para balancear el dataset partido
            #print(dfreviewdes)
            rus = RandomUnderSampler()
            dfreviewbal, dfreviewbal[sentiment] = rus.fit_resample(
                dfreviewdes[[review]], dfreviewdes[sentiment])
            #print(dfreviewbal)
            #se divide en un conjunto de entrenamiento y otro de prueba
            train, test = train_test_split(
                dfreviewbal, test_size=0.33, random_state=42)

            #Se llenan los conjuntos de entrenamiento y prueba en valor(review) y output(sentiment)
            trainX, trainY = train[review], train[sentiment]
            testX, testY = test[review], test[sentiment]
            """    
            #ejemplo de manejo de texto a representacion numerica
            text = ["I love writing code in Python. I love Python code",
                    "I hate writing code in Java. I hate Java code"]
            #Ejemplo para crear una matriz de conteo de frecuencias
            df = pd.DataFrame({review: ['review1', 'review2'], 'text':text})
            cv = CountVectorizer(stop_words='english')
            cv_matrix = cv.fit_transform(df['text'])
            df_dtm = pd.DataFrame(cv_matrix.toarray(), index=df[review].values, columns=cv.get_feature_names_out())
            #ejemplo para crear una matriz con valores tfidf
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df['text'])
            df_dtm = pd.DataFrame(tfidf_matrix.toarray(), index=df[review].values, columns=tfidf.get_feature_names_out())
            """

            #creamos y llenamos los conjuntos de valores que vamos a usar para los modelos y creamos nuestra bolsa de palabras en base al conjunto de entrenamiento
            if stopw == "no":
                tfidf = TfidfVectorizer()
            elif stopw == "spanish":
                tfidf = TfidfVectorizer(stop_words=stopword_es)
            else:
                tfidf = TfidfVectorizer(stop_words='english')
            trainXVector = tfidf.fit_transform(trainX)
            testXVector = tfidf.transform(testX)

            #SVM
            svc = SVC(kernel='linear')
            svc.fit(trainXVector, trainY)

            #save model
            pickle.dump(svc, open('model.pkl', 'wb'))
            pickle.dump(tfidf, open('wordvector.mkl', 'wb'))
            global model
            global tfidfv
            model = svc
            tfidfv = tfidf

            return render_template('index2.html', feed_result='The model was feed with: {} with score of {}%'.format(uploaded_file.filename, round(model.score(testXVector, testY)*100, 2)))
        except Exception:
            print(Exception)
            return render_template('index2.html', feed_result='No valid file provided')
    else:
        return render_template('index2.html', feed_result='No valid file provided')


#Run app
if __name__ == "__main__":
    app.run(debug=True)
