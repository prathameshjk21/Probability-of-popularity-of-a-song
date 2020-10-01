from flask import Flask,jsonify, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('music_popularity.pkl', 'rb'))


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


#standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
     
        acousticness = float(request.form['acousticness'])
        danceability=float(request.form['danceability'])
        energy=float(request.form['energy'])
        instrumentalness =float(request.form['instrumentalness'])
        key = int(request.form['key'])
        liveness=float(request.form['liveness'])
        loudness=float(request.form['loudness'])
        speechiness = int(request.form['speechiness'])
        tempo = float(request.form['tempo'])
        valence = float(request.form['valence'])
        
        prediction=model.predict([[acousticness,danceability,energy,instrumentalness,key,liveness,loudness,speechiness,tempo,valence]])
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="please enter proper values")
        else:
            return render_template('index.html',prediction_text="probability that your song going to be famous is {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

