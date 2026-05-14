from flask import Flask, render_template, request, jsonify
import json
import pickle
import random

# Load trained model and vectorizer
model=pickle.load(open("chatbot_model.pkl","rb"))
vectorizer=pickle.load(open("vectorizer.pkl","rb"))

# Load intents
with open("intents.json") as f:
    intents=json.load(f)

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input=request.json["message"]
    X=vectorizer.transform([user_input])
    tag=model.predict(X)[0]

    for intent in intents["intents"]:
        if intent["tag"]==tag:
            return jsonify({"response":random.choice(intent["responses"])})

    return jsonify({"response":"Sorry, I didn't understand that."})

if __name__=="__main__":
    app.run(debug=True)
