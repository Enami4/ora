# Installation de Poppler pour Windows

Poppler est nécessaire pour convertir les PDFs en images pour l'OCR.

## Option 1 : Installation manuelle (Recommandé)

1. Téléchargez Poppler pour Windows depuis :
   https://github.com/oschwartz10612/poppler-windows/releases/

2. Choisissez la version Release-XX.XX.X-X.zip (ex: Release-24.08.0-0.zip)

3. Extrayez le fichier ZIP dans un dossier, par exemple :
   `C:\Program Files\poppler`

4. Ajoutez le dossier `bin` au PATH Windows :
   - Le chemin sera : `C:\Program Files\poppler\Library\bin`
   - Ouvrez les Paramètres Système → Variables d'environnement
   - Ajoutez ce chemin à la variable PATH

## Option 2 : Utilisation avec chemin spécifique

Si vous ne voulez pas modifier le PATH, vous pouvez spécifier le chemin dans le code :

```python
from pdf2image import convert_from_path

poppler_path = r"C:\Program Files\poppler\Library\bin"
images = convert_from_path(pdf_path, poppler_path=poppler_path)
```

## Vérification

Après installation, testez avec :
```cmd
pdftoppm -h
```

Si la commande fonctionne, Poppler est correctement installé.