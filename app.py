from flask import Flask,request,jsonify
import pandas as pd
from helper import make_predict

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/predict",method=["POST"])
def predict():
    data=request.get_json()
    if "comment" not in data:
        return jsonify({"error": "No comment provided"}), 400
    comments=data["comment"]
    if not isinstance(comments, list):
        return jsonify({"error": "comments must be a list"}), 400
    
    predictions=make_predict(comments)
    return jsonify({
        "predictions": predictions.tolist()
    })

if __name__ == "__main__":
    df=pd.read_csv("test.csv")
    print(df.shape)
    app.run(debug=True)