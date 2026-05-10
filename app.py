from flask import Flask,request,jsonify
import pandas as pd
from helper import make_predict
from flask_cors import CORS
import json
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": ["http://localhost:3000","https://next-blog-dun.vercel.app"]}},
      supports_credentials=True,  
     )


@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/predict",methods=["POST"])
def predict():
    data=request.get_json()
    if "comments" not in data:
        return jsonify({"error": "No comment provided"}), 400
    comments=data["comments"]
    parsedComments=json.loads(comments)
    if not isinstance(parsedComments, list):
        return jsonify({"error": "comments must be a list"}), 400
    
    predictions=make_predict(parsedComments)
    return jsonify({
        "predictions": predictions
    })

if __name__ == "__main__":
    df=pd.read_csv("test.csv")
    print(df.shape)
    app.run(debug=True)