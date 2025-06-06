from flask import Flask, render_template, request, jsonify
import chatbot 

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_msg = request.json.get("message")
    bot_reply = chatbot.get_response(user_msg)  
    return jsonify({"response": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
