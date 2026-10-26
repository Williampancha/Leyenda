FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KERAS_BACKEND=torch \
    CUDA_VISIBLE_DEVICES=""

# Installer PyTorch en version CPU uniquement pour alléger l'image Docker
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Installer les dépendances de l'application web
COPY web_app/requirements.txt ./web_app/requirements.txt
RUN pip install --no-cache-dir -r web_app/requirements.txt

# Copier les modèles et le code de l'application
COPY models/ ./models/
COPY web_app/ ./web_app/

# Pré-télécharger les poids d'InceptionV3 lors du build pour éviter le téléchargement au premier appel
RUN python -c "from keras.applications.inception_v3 import InceptionV3; InceptionV3(weights='imagenet', include_top=False)"

EXPOSE 7860

# 1 worker unique pour ne pas dupliquer les modèles en RAM, 4 threads et timeout de 180s
CMD ["gunicorn", "--chdir", "web_app", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "180"]
