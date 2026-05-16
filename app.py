from flask import Flask,request,jsonify

from helper import make_predict,recommend_blogs,similar_posts,trending_in_network
from flask_cors import CORS
import json
app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": [
        "http://localhost:3000",
        "https://next-blog-dun.vercel.app"
    ]}},
    supports_credentials=True
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

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id")
    N = int(request.args.get("n", 10))
    res = recommend_blogs(user_id, N)

    return jsonify({"data":res})

@app.route("/similar", methods=["GET"])
def similar():
    post_id = request.args.get("post_id")
    N = int(request.args.get("n", 10))
    res = similar_posts(post_id, N)

    return jsonify({"data":res})

@app.route("/trending-in-network", methods=["GET"])
def trending():
    user_id = request.args.get("user_id")
    N = int(request.args.get("n", 10))
    res = trending_in_network(user_id, N)

    return jsonify({"data":res})

if __name__ == "__main__":
    app.run(debug=True)