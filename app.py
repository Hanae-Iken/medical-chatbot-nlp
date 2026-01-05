from flask import Flask, render_template, request
from chatbot import chatbot_response

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        user_input = request.form.get("message", "")
        response = chatbot_response(user_input or "")
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
