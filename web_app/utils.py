from __future__ import annotations

import gc
import io
import os
import collections
from collections.abc import Iterator
from pathlib import Path
from typing import Any

os.environ["KERAS_BACKEND"]="torch"

import keras
import numpy as np
import torch
import tensorflow as tf
from PIL import Image
from keras import layers, ops
from keras.applications.inception_v3 import InceptionV3, preprocess_input


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"

CLASSIFIER_WEIGHTS = MODELS_DIR / "model_full.weights.h5"
DENOISER_MODEL = MODELS_DIR / "unet_denoiser_livrable2.keras"
LSTM_ENCODER_WEIGHTS = MODELS_DIR / "encoder_LSTM_epoch_019.weights.h5"
LSTM_DECODER_WEIGHTS = MODELS_DIR / "decoder_LSTM_epoch_019.weights.h5"

CLASS_NAMES = ["Painting", "Photo", "Schematics", "Sketch", "Text"]
PHOTO_CLASS_INDEX = CLASS_NAMES.index("Photo")
THRESHOLD = 0.2301
ATTENTION_FEATURES_SHAPE = 64
EMBEDDING_DIM = 256
UNITS = 512
VOCAB_SIZE = 5001

_classifier: keras.Model | None = None
_denoiser: keras.Model | None = None
_inception_extractor: keras.Model | None = None
_lstm_encoder: CNNEncoder | None = None
_lstm_decoder: AttentionDecoder | None = None
_tokenizer: Any | None = None


def _require_paths(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Fichiers nécessaires introuvables : " + ", ".join(missing))


def _image_from_bytes(content: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            return image.convert("RGB").copy()
    except (OSError, ValueError) as error:
        raise ValueError("Le fichier envoyé n'est pas une image valide.") from error


def is_valid_image(content: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


# Reprise de load_image_as_array du workflow de prétraitement du notebook.
def load_image_as_array(
    image: str | Path | bytes,
    target_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    if isinstance(image, bytes):
        source = _image_from_bytes(image)
    else:
        source = Image.open(image).convert("RGB")

    with source:
        return np.asarray(source.resize(target_size), dtype=np.float32)


def load_classifier_model() -> keras.Model:
    global _classifier
    if _classifier is None:
        _require_paths(CLASSIFIER_WEIGHTS)
        _classifier = keras.models.load_model(
            CLASSIFIER_WEIGHTS,
            compile=False,
        )
    return _classifier


def classify_image(content: bytes) -> dict[str, Any]:
    classifier = load_classifier_model()
    image_array = np.expand_dims(load_image_as_array(content), axis=0)
    probabilities = classifier.predict(image_array, verbose=0)
    proba_photo = float(probabilities[0][PHOTO_CLASS_INDEX])
    predicted_index = int(np.argmax(probabilities[0]))

    return {
        "is_image": True,
        "is_photo": proba_photo >= THRESHOLD,
        "label": CLASS_NAMES[predicted_index],
        "photo_probability": proba_photo,
    }


# Reprise de load_denoiser_model et load_image_for_denoiser du notebook.
def load_denoiser_model() -> keras.Model:
    global _denoiser
    if _denoiser is None:
        _require_paths(DENOISER_MODEL)
        _denoiser = keras.models.load_model(DENOISER_MODEL, compile=False)
    return _denoiser


def load_image_for_denoiser(
    image: str | Path | bytes,
    target_size: tuple[int, int] = (299, 299),
) -> np.ndarray:
    return load_image_as_array(image, target_size) / 255.0


def denoise_image(content: bytes) -> np.ndarray:
    denoiser = load_denoiser_model()
    image = load_image_for_denoiser(content)
    height, width = image.shape[:2]
    pad_height = (-height) % 4
    pad_width = (-width) % 4
    padded = np.pad(
        image,
        ((0, pad_height), (0, pad_width), (0, 0)),
        mode="edge",
    )
    batch = np.expand_dims(padded, axis=0)
    denoised = keras.ops.convert_to_numpy(
        denoiser(batch, training=False)
    )[0]
    denoised = denoised[:height, :width, :]
    gc.collect()
    return np.clip(denoised, 0.0, 1.0).astype(np.float32)


def build_inception_feature_extractor() -> keras.Model:
    global _inception_extractor
    if _inception_extractor is None:
        image_model = InceptionV3(weights="imagenet", include_top=False)
        new_input = keras.Input(shape=(299, 299, 3))
        _inception_extractor = keras.Model(
            new_input,
            image_model(new_input),
        )
    return _inception_extractor


def load_image_for_inception(
    image: str | Path | np.ndarray,
    target_size: tuple[int, int] = (299, 299),
) -> np.ndarray:
    if isinstance(image, np.ndarray):
        image_array = image
        if image_array.max() <= 1.0:
            image_array = image_array * 255.0
        image_array = np.asarray(image_array, dtype="float32")
        image_array = np.asarray(
            Image.fromarray(image_array.astype("uint8")).resize(target_size),
            dtype="float32",
        )
    else:
        image_array = load_image_as_array(image, target_size)
    return preprocess_input(image_array)


def extract_features(image: np.ndarray) -> Any:
    extractor = build_inception_feature_extractor()
    batch = np.expand_dims(
        load_image_for_inception(image),
        axis=0,
    )
    features = extractor(batch, training=False)
    return ops.reshape(features, (features.shape[0], -1, features.shape[-1]))

class CNNEncoder(keras.Model):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.fc = layers.Dense(embedding_dim)

    def call(self, features, training: bool = False):
        return ops.relu(self.fc(features))

class BahdanauAttention(layers.Layer):
    def __init__(self, units: int):
        super().__init__()
        self.W_features = layers.Dense(units)
        self.W_hidden = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features, hidden):
        hidden_expanded = ops.expand_dims(hidden, axis=1)
        scores = self.V(
            ops.tanh(
                self.W_features(features)
                + self.W_hidden(hidden_expanded)
            )
        )
        weights = ops.softmax(scores, axis=1)
        context = ops.sum(weights * features, axis=1)
        return context, weights


class AttentionDecoder(keras.Model):
    def __init__(self, embedding_dim: int, units: int, vocab_size: int):
        super().__init__()
        self.rnn_type = "lstm"
        self.units = units
        
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.attention = BahdanauAttention(units)
        self.rnn = layers.LSTM(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc1 = layers.Dense(units)
        self.dropout = layers.Dropout(0.0)
        self.fc2 = layers.Dense(vocab_size)

    def call(self, token, features, state, training: bool = False):
        hidden = state[0]
        context, attention_weights = self.attention(features, hidden)
        embedded = self.embedding(token)
        recurrent_input = ops.concatenate(
            [ops.expand_dims(context, axis=1), embedded],
            axis=-1,
        )
        output, next_hidden, next_cell = self.rnn(
            recurrent_input,
            initial_state=state,
            training=training,
        )
        output = self.fc1(output)
        output = self.dropout(output, training=training)
        logits = self.fc2(output)
        return (
            ops.squeeze(logits, axis=1),
            (next_hidden, next_cell),
            attention_weights,
        )

    def reset_state(self, batch_size: int):
        hidden = ops.zeros((batch_size, self.units))
        return hidden, ops.zeros((batch_size, self.units))


def _load_tokenizer() -> Any:
    global _tokenizer
    with open(MODELS_DIR / 'tokenizer.json', 'r') as f:
        _tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
    return _tokenizer


def _load_lstm_models() -> tuple[CNNEncoder, AttentionDecoder]:
    global _lstm_encoder, _lstm_decoder
    if _lstm_encoder is not None and _lstm_decoder is not None:
        return _lstm_encoder, _lstm_decoder

    _require_paths(LSTM_ENCODER_WEIGHTS, LSTM_DECODER_WEIGHTS)
    encoder = CNNEncoder(EMBEDDING_DIM)
    decoder = AttentionDecoder(EMBEDDING_DIM, UNITS, VOCAB_SIZE)
    dummy_features = torch.zeros((1, ATTENTION_FEATURES_SHAPE, 2048),)
    encoded_features = encoder(dummy_features, training=False)
    dummy_token = torch.zeros((1, 1),dtype=torch.int64,)
    decoder(dummy_token,encoded_features,decoder.reset_state(1),training=False,)
    encoder.load_weights(LSTM_ENCODER_WEIGHTS)
    decoder.load_weights(LSTM_DECODER_WEIGHTS)
    _lstm_encoder, _lstm_decoder = encoder, decoder
    return encoder, decoder


def infer_caption(image: np.ndarray) -> str:
    encoder, decoder = _load_lstm_models()
    tokenizer = _load_tokenizer()
    features = encoder(extract_features(image), training=False)
    state = decoder.reset_state(batch_size=1)
    start_token = tokenizer.word_index.get("<start>")
    if start_token is None:
        raise ValueError("Le tokenizer ne contient pas le token <start>.")

    decoder_input = torch.tensor([[start_token]],dtype=torch.int64,)
    result = []
    for _ in range(52): #max_length = 52
        predictions, state, _ = decoder(decoder_input,features,state,training=False,)
        predicted_id = int(torch.argmax(predictions[0]).item())
        word = tokenizer.index_word.get(predicted_id, "<unk>")
        if word == "<end>":
            break
        if word not in {"<start>", "<unk>"}:
            result.append(word)
        decoder_input = torch.tensor([[predicted_id]],dtype=torch.int64,)
    return " ".join(result)


def predict_image(content: bytes) -> dict[str, Any]:
    result = None
    for event in predict_image_steps(content):
        if "result" in event:
            result = event["result"]

    if result is None:
        raise RuntimeError("La pipeline n'a produit aucun résultat.")
    return result


def predict_image_steps(content: bytes) -> Iterator[dict[str, Any]]:
    if not is_valid_image(content):
        result = {
            "is_image": False,
            "is_photo": False,
            "prediction": None,
            "error": "Le fichier envoyé n'est pas une image valide.",
        }
        yield {"stage": "error", "message": result["error"], "result": result}
        return

    yield {
        "stage": "validation",
        "message": "Image reçue et vérifiée.",
    }
    classification = classify_image(content)
    if not classification["is_photo"]:
        result = {
            **classification,
            "prediction": None,
            "error": "L'image n'est pas classée comme une photographie.",
        }
        yield {
            "stage": "classification",
            "message": (
                f"Image classée comme « {classification['label']} » : "
                "ce n'est pas une photographie."
            ),
            "result": result,
        }
        return

    yield {
        "stage": "classification",
        "message": (
            f"Image classée comme photographie "
            f"(probabilité : {classification['photo_probability']:.1%})."
        ),
        "result": classification,
    }

    denoised = denoise_image(content)
    yield {
        "stage": "denoising",
        "message": "Image débruitée avec succès.",
    }

    prediction = infer_caption(denoised)
    result = {
        **classification,
        "prediction": prediction,
        "error": None,
    }
    yield {
        "stage": "caption",
        "message": "Légende générée.",
        "result": result,
    }
