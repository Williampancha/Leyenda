# Legenda — interface web d'upload image

## 1) Installation

```bash
pip install -r web_app/requirements.txt
```

## 2) Lancer l'application

```powershell
python web_app\app.py
```

Puis ouvre `http://127.0.0.1:5000`.

## 3) Où brancher ton modèle

La pipeline complète se trouve dans [utils.py](./utils.py). Elle utilise Keras
avec le backend Torch, mais l'inférence est explicitement exécutée sur CPU et
ne force pas l'utilisation du GPU :

```python
classification = classify_image(content)
denoised = denoise_image(content)
features = extract_features(denoised)
prediction = infer_caption(denoised)
```

L'API `/api/predict` reçoit le fichier uploadé dans le champ `image`.

Les fichiers attendus dans [models/](../models/) sont :

- `model_full.weights.h5` : classifieur Photo / Non-photo ;
- `unet_denoiser_livrable2.keras` : modèle de débruitage ;
- `encoder_LSTM_epoch_019.weights.h5` ;
- `decoder_LSTM_epoch_019.weights.h5` ;
Le tokenizer n'est pas stocké séparément. Il est reconstruit au démarrage avec
`create_captions_and_path_vectors` puis `annotation_preprocessing`, comme dans le
notebook, à partir de `datasets/Livrable 3/annotations/captions_train2014.json`
et des fichiers `.jpg.npy` présents dans `denoised_images`.

L'API `/api/predict` renvoie :

```json
{
  "is_image": true,
  "is_photo": true,
  "label": "Photo",
  "photo_probability": 0.97,
  "prediction": "une légende..."
}
```

Pour une image valide mais non photographique, elle renvoie un statut HTTP `422` :

```json
{
  "is_image": true,
  "is_photo": false,
  "prediction": null,
  "error": "L'image n'est pas classée comme une photographie."
}
```
