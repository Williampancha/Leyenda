from __future__ import annotations

import json

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

try:
    from .utils import predict_image, predict_image_steps
except ImportError:
    from utils import predict_image, predict_image_steps


app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify(
            {
                "is_image": False,
                "prediction": None,
                "error": "Aucun fichier reçu (champ attendu: image).",
            }
        ), 400

    content = file.read()
    try:
        result = predict_image(content)
    except (FileNotFoundError, ValueError, OSError) as error:
        return jsonify(
            {
                "is_image": True,
                "prediction": None,
                "error": str(error),
            }
        ), 500

    if not result["is_image"]:
        status = 400
    elif not result["is_photo"]:
        status = 422
    else:
        status = 200

    return jsonify(result), status


@app.post("/api/predict-stream")
def predict_stream():
    file = request.files.get("image")
    if file is None:
        return jsonify(
            {
                "is_image": False,
                "prediction": None,
                "error": "Aucun fichier reçu (champ attendu: image).",
            }
        ), 400

    content = file.read()

    def events():
        try:
            for event in predict_image_steps(content):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except (FileNotFoundError, ValueError, OSError) as error:
            yield json.dumps(
                {
                    "stage": "error",
                    "message": str(error),
                    "result": {
                        "is_image": True,
                        "prediction": None,
                        "error": str(error),
                    },
                },
                ensure_ascii=False,
            ) + "\n"

    return Response(
        stream_with_context(events()),
        mimetype="application/x-ndjson",
    )


if __name__ == "__main__":
    app.run(debug=True)
