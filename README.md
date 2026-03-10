# 🎭 Image Emotion Recognition Challenge

<div align="center">

**Classify emotions in facial images — submit your predictions and climb the leaderboard!**

[![Leaderboard](https://img.shields.io/badge/🏆_Leaderboard-Live-brightgreen)](https://your-streamlit-app.streamlit.app)
[![Submissions](https://img.shields.io/badge/Submissions-Open-blue)](../../issues/new?template=submission.yml)

</div>

---

## 📋 Overview

This competition challenges participants to classify the **emotion** of facial images into 7 categories:

| Label | Emotion |
|-------|---------|
| 0 | Angry 😠 |
| 1 | Disgust 🤢 |
| 2 | Fear 😨 |
| 3 | Happy 😄 |
| 4 | Neutral 😐 |
| 5 | Sad 😢 |
| 6 | Surprise 😲 |

**Metric**: Accuracy on a hidden test set.

---

## 📁 Data

Download the dataset here: **[link to your dataset]**

```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── img_0001.jpg
    ├── img_0002.jpg
    └── ...
```

The `test/` folder contains images **without labels**. Your model must predict the emotion for each image.

A `sample_submission.csv` is provided as a starting template.

---

## 🚀 How to Participate

### Step 1 — Train your model

Use the `train/` folder to build and train your model. You are free to use any framework (PyTorch, TensorFlow, sklearn...).

A baseline is available in `docs/baseline_model.py`.

### Step 2 — Generate predictions on the test set

```python
import pandas as pd
import os

predictions = []

for img_file in sorted(os.listdir("data/test/")):
    image_id = img_file.replace(".jpg", "")
    label = your_model.predict(f"data/test/{img_file}")  # 0 to 6
    predictions.append({"image_id": image_id, "label": label})

df = pd.DataFrame(predictions)
df.to_csv("my_submission.csv", index=False)
```

### Step 3 — Check your CSV format

Your file must look exactly like this:

```csv
image_id,label
img_0001,3
img_0002,0
img_0003,5
```

- `image_id`: exact filename without extension
- `label`: integer from 0 to 6

### Step 4 — Submit via GitHub Issue

1. Go to [Issues → New Issue](../../issues/new?template=submission.yml)
2. Choose **"Prediction Submission"** template
3. Fill in your details and **attach your CSV file**
4. Submit — results appear on the leaderboard within ~5 minutes!

---

## 📦 Submission Requirements

| Requirement | Detail |
|-------------|--------|
| File format | `.csv` |
| Columns | `image_id`, `label` |
| Labels | Integers in `{0, 1, 2, 3, 4, 5, 6}` |
| Rows | Must match the number of test images exactly |
| Submissions/day | Maximum 3 per participant |

---

## 📊 Evaluation

Models are ranked by **Accuracy** on the secret test labels.

```
Accuracy = Number of correct predictions / Total predictions
```

---

## 🏗️ Repository Structure

```
sentiment-competition/
├── .github/
│   ├── workflows/
│   │   └── evaluate.yml           # Auto-evaluation pipeline
│   └── ISSUE_TEMPLATE/
│       └── submission.yml         # Submission form
├── evaluator/
│   └── evaluate.py                # Evaluation script
├── leaderboard/
│   ├── app.py                     # Streamlit leaderboard
│   └── results.json               # Scores database
├── submission_handler/
│   └── process_submission.py      # Parses issue + runs evaluation
├── data/
│   └── sample_submission.csv      # Template CSV for participants
└── docs/
    └── baseline_model.py          # Starter baseline
```

---

## 🏆 Prize

| Rank | Prize |
|------|-------|
| 🥇 1st | 🍫 Chocolate |
