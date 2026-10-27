# Legenda — Interface Web d'Annotation Automatique d'Images

🌐 **Application déployée en ligne :** [**https://leyenda.darylbdr.tech**](https://leyenda.darylbdr.tech) *(hébergée sur Azure Container Apps & Cloudflare)*

Cette application web **Flask** expose le pipeline complet de Deep Learning développé au cours des 3 livrables du projet **Leyenda / TouNum** :
1. **Validation & Classification Photo / Non-photo** (CNN du Livrable 1, seuil de Youden $\theta = 0,2301$).
2. **Débruitage photographique** (U-Net $\text{MSE} + \text{SSIM}$ du Livrable 2).
3. **Extraction de features & Image Captioning** (`InceptionV3` + `CNNEncoder` + `BahdanauAttention` + `AttentionDecoder` LSTM du Livrable 3).

---

## 1. Installation et lancement en local

Depuis la racine du projet :

```bash
pip install -r web_app/requirements.txt
python web_app/app.py
```

Ouvrir ensuite **`http://127.0.0.1:5000`** dans un navigateur.

---

## 2. Lancement avec Docker (Production)

Le fichier [`Dockerfile`](../Dockerfile) situé à la racine du dépôt installe PyTorch en version CPU, pré-télécharge les poids ImageNet d'`InceptionV3` et démarre l'application avec **Gunicorn** sur le port `7860` :

```bash
docker build -t leyenda .
docker run -p 7860:7860 leyenda
```

L'interface est alors accessible sur **`http://localhost:7860`**.

---

## 3. Architecture du pipeline et inférence CPU (`utils.py`)

Toute la logique d'inférence se trouve dans [`utils.py`](./utils.py). Elle utilise **Keras 3** avec le backend **PyTorch**, en forçant explicitement l'exécution sur **CPU** afin de garantir la portabilité sur les environnements Cloud sans GPU :

```python
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")
CPU_DEVICE = torch.device("cpu")
torch.set_default_device(CPU_DEVICE)
```

Les quatre étapes exécutées pour chaque image envoyée sont :

```python
classification = classify_image(content)  # Redimensionnement 128x128 -> CNN 5 classes (seuil Photo >= 0.2301)
denoised = denoise_image(content)         # Redimensionnement 299x299 + padding multiple de 4 -> U-Net
features = extract_features(denoised)     # InceptionV3 (include_top=False) -> tenseur (1, 64, 2048)
prediction = infer_caption(denoised)      # Décodage glouton (max_length = 52) avec Attention de Bahdanau + LSTM
```

### Fichiers requis dans [`models/`](../models/)
- `model_full.weights.h5` : poids du classifieur Photo / Non-photo ;
- `unet_denoiser_livrable2.keras` : modèle complet de débruitage U-Net ;
- `encoder_LSTM_epoch_019.weights.h5` : poids de l'encodeur CNN ;
- `decoder_LSTM_epoch_019.weights.h5` : poids du décodeur LSTM à attention ;
- `tokenizer.json` : dictionnaire de vocabulaire (5 001 tokens dont `<pad>`, `<start>`, `<end>`, `<unk>`).

---

## 4. Endpoints de l'API (`app.py`)

### `POST /api/predict-stream` *(utilisé par l'interface web)*
Reçoit le fichier dans le champ `multipart/form-data` nommé `image` et renvoie un flux **`application/x-ndjson`** permettant à l'interface [`templates/index.html`](./templates/index.html) de mettre à jour la progression en temps réel :
1. `{"stage": "validation", "message": "Image reçue et vérifiée."}`
2. `{"stage": "classification", "message": "...", "result": {...}}` *(s'arrête ici si l'image n'est pas une photographie)*
3. `{"stage": "denoising", "message": "Image débruitée avec succès."}`
4. `{"stage": "caption", "message": "Légende générée.", "result": {...}}`

### `POST /api/predict` *(réponse JSON unique)*
Reçoit le fichier dans le champ `image` et renvoie directement le résultat final :

- **Statut `200 OK` (Photographie reconnue et légendée) :**
  ```json
  {
    "is_image": true,
    "is_photo": true,
    "label": "Photo",
    "photo_probability": 0.97,
    "prediction": "a man riding a wave on top of a surfboard",
    "error": null
  }
  ```
- **Statut `422 Unprocessable Entity` (Image valide mais classée comme non-photographie) :**
  ```json
  {
    "is_image": true,
    "is_photo": false,
    "label": "Sketch",
    "photo_probability": 0.04,
    "prediction": null,
    "error": "L'image n'est pas classée comme une photographie."
  }
  ```
- **Statut `400 Bad Request` (Fichier manquant ou format d'image invalide).**
