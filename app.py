import numpy as np
from flask import app,Flask,render_template,jsonify,json
import pickle

app = Flask(__name__)
model = pickle.load(open(r"C:\Users\hp\PycharmProjects\pythonProject1\model_LR.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods = ["POST"])
def predict_api():
    data = request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = np.array(list(data.values())).reshape(1,-1)
    prediction = model.predict(new_data)

    output = prediction[0]
    return jsonify(output[0])

#     return render_template('index.html', prediction_text='Dead[0]/Alive[1] = {}'.format(output))

if __name__ == "__main__":
    app.run(host="127.0.0.3",debug=True)