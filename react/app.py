from flask import Flask, render_template, url_for, request, redirect
import document_similarity

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template("main.html")

#@app.route('/dashboard/')
#def dashboard():
#    return render_template("dashboard.html") 

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file_upload = request.files['file']
        if file_upload.filename != '':
            file_upload.save('upload.txt')
            prediction = str(document_similarity.open_predict_classification('upload.txt'))
            similarity = str(document_similarity.process_email_similarity('upload.txt'))
            #return redirect(url_for('main.html'))
            #output = 'Prediction: {}Similarity: {}'.format(prediction, similarity)
            # print(type(similarity))
            # print(len(similarity))
            similarity = similarity.split('\n')
            return render_template('response.html',prediction=prediction, similarity=similarity, len=len(similarity))
            return redirect(url_for('main.html'))

if __name__ == "__main__":
    app.run()