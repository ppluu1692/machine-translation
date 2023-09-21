
from translate import E2V, V2E

from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route("/en-to-vi")
def translate_E2V():
    msg = request.args.get("message")
    return f"{E2V(msg)}"

@app.route("/vi-to-en")
def translate_V2E():
    msg = request.args.get("message")
    return f"{V2E(msg)}"

if __name__ == "__main__":
    app.run(debug=True)

