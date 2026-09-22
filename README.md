# 🍅 Tomato Leaf Disease Classifier

An end-to-end **Computer Vision / Image Classification** project: a model
that detects diseases in tomato leaves from an image, plus a live demo app
to try it out.

---

## What the project does

Upload a photo of a tomato leaf -> the model returns one of 6 predictions:

| Class | Description |
|---|---|
| `Tomato___healthy` | Healthy leaf |
| `Tomato___Late_blight` | Late blight |
| `Tomato___Early_blight` | Early blight |
| `Tomato___Leaf_Mold` | Leaf mold |
| `Tomato___Septoria_leaf_spot` | Septoria leaf spot |
| `Tomato___Bacterial_spot` | Bacterial spot |

## Results

After training, the model achieved **82.9%** accuracy on a held-out test
set never seen during training - 240 images, 6 balanced classes (40 per
class). For comparison, random guessing among 6 classes would give about
16.7% - so the model is clearly learning real patterns, not guessing.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Healthy | 0.80 | 1.00 | 0.89 |
| Late Blight | 0.97 | 0.75 | 0.85 |
| Leaf Mold | 0.87 | 0.97 | 0.92 |
| Bacterial Spot | 0.87 | 0.85 | 0.86 |
| Septoria Leaf Spot | 0.73 | 0.93 | 0.81 |
| Early Blight | 0.79 | 0.47 | 0.59 |

The weakest class is Early Blight (relatively low recall) - the model
confuses it with visually similar classes. Full details in
`models/metrics.json`, and a visual confusion matrix in
`models/confusion_matrix.png`.

![Confusion Matrix](models/confusion_matrix.png)
![Training History](models/training_history.png)

## Architecture and approach

- **Model**: MobileNetV2 (pretrained on ImageNet) + Transfer Learning
- **Dataset**: a subset (2,400 images, 6 classes) of the
  [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) -
  real images, not synthetic
- **Two-stage training**:
  1. Train only the top classification layers (base frozen) - 8 epochs
  2. Fine-tuning: unfreeze the top layers of the base and train further - 4 epochs
- **Data Augmentation**: random flips, rotation, zoom and contrast during
  training, to reduce overfitting
- **Demo app**: Streamlit - upload an image or take a photo directly with
  the camera

The project structure clearly separates stages (data prep / training /
evaluation / app), as expected in a professional ML project, rather than
everything in one big notebook.

## Project structure

```
leaf-disease-classifier/
├── data/
│   └── prepare_data.py     # Downloads and splits the dataset
├── src/
│   ├── model_utils.py      # Shared constants (paths, image sizes, etc.)
│   ├── train.py             # Model training (Transfer Learning + Fine-tuning)
│   ├── evaluate.py          # Evaluation on the test set + confusion matrix
│   └── predict.py           # Prediction on a single image from the CLI
├── app/
│   └── streamlit_app.py     # The live demo app
├── models/                  # Trained model + plots + metrics (created after training)
├── tests/
│   └── test_model.py        # Basic sanity tests
├── requirements.txt
└── README.md
```

## How to run

### 1. Installation

```bash
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the data

```bash
python data/prepare_data.py
```

Downloads ~2,400 images (about 160MB) and splits them into train/val/test.
A one-time operation that takes about a minute.

### 3. Train the model

```bash
python src/train.py
```

Takes about 5-10 minutes on a regular CPU without a GPU. If you have a GPU
available, TensorFlow will use it automatically and it will be significantly
faster. When done, the model is saved to `models/leaf_disease_model.keras`.

### 4. Evaluate the model

```bash
python src/evaluate.py
```

Generates the confusion matrix, a full classification report, and a
`metrics.json` file.

### 5. Run the demo app

```bash
streamlit run app/streamlit_app.py
```

Opens in your browser at `http://localhost:8501`. Upload a tomato leaf
image (you can search "tomato leaf disease" on Google Images for testing)
and see the prediction in real time.

### Tests

```bash
python tests/test_model.py
```

## Deployment - to get a live link to share

Two recommended free options:

### Option A: Streamlit Community Cloud (simplest)

1. Push the project to GitHub (public repo)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Select the repo, and set the app path to `app/streamlit_app.py`
4. Within a minute you'll get a public link like `https://your-app.streamlit.app`

> ⚠️ Important: make sure `models/leaf_disease_model.keras` **is** pushed
> to GitHub (it's small, about 9MB) - remove it from `.gitignore` if needed,
> otherwise the deployed app won't have a model to load.

### Option B: Hugging Face Spaces

1. Create a new Streamlit Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Upload the project files (or connect it to the GitHub repo)
3. You'll get a permanent link like `https://huggingface.co/spaces/your-username/your-app`

## Possible extensions

- Add more classes/plants from the full dataset (it has 38 classes, 14 plants)
- Add Grad-CAM to visually show "where the model is looking" in the image
- Deploy a separate FastAPI API alongside the Streamlit app
- Experiment with other architectures (EfficientNet, ResNet) and compare performance

## Dataset credit

Images come from the [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
(Hughes & Salathé, 2015), an open and public dataset widely used for plant
disease detection research.
