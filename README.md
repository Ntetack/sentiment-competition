# 🎭 Image Sentiment Analysis Challenge

<div align="center">

![Banner](docs/banner.png)

**Compete to build the best image sentiment analysis model!**

[![Leaderboard](https://img.shields.io/badge/🏆_Leaderboard-Live-brightgreen)](https://your-streamlit-app.streamlit.app)
[![Submissions](https://img.shields.io/badge/Submissions-Open-blue)](https://github.com/your-username/sentiment-competition/issues/new?template=submission.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

## 📋 Overview

This competition challenges participants to classify the **emotional sentiment** of images into categories:

| Label | Description |
|-------|-------------|
| 0 | Very Negative 😠 |
| 1 | Negative 😞 |
| 2 | Neutral 😐 |
| 3 | Positive 😊 |
| 4 | Very Positive 😄 |

**Metric**: Weighted F1-Score on a hidden test set.

---

## 🚀 How to Participate

### Step 1 — Develop your model

Train a PyTorch model that accepts an image tensor `(B, 3, 224, 224)` and returns logits `(B, 5)`.

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # your architecture here

    def forward(self, x):
        # x: (B, 3, 224, 224)
        # returns: (B, 5) logits
        ...
```

### Step 2 — Export your model

Save your trained model:

```python
torch.save(model.state_dict(), "my_model.pth")
```

### Step 3 — Submit via GitHub Issue

1. Go to [Issues → New Issue](../../issues/new?template=submission.yml)
2. Choose **"Model Submission"** template
3. Fill in your details and upload your `.pth` file
4. Submit — our CI/CD pipeline will automatically evaluate your model!

### Step 4 — Check the Leaderboard

Results appear on the [live leaderboard](https://your-streamlit-app.streamlit.app) within ~10 minutes.

---

## 📦 Model Requirements

| Requirement | Detail |
|-------------|--------|
| Framework | PyTorch >= 1.12 |
| Input | `torch.Tensor` of shape `(B, 3, 224, 224)`, values in `[0, 1]` |
| Output | `torch.Tensor` of shape `(B, 5)` — raw logits |
| File size | ≤ 500MB |
| Inference time | ≤ 30s per batch of 32 |

---

## 🏗️ Repository Structure

```
sentiment-competition/
├── .github/
│   ├── workflows/
│   │   └── evaluate.yml        # Auto-evaluation pipeline
│   └── ISSUE_TEMPLATE/
│       └── submission.yml      # Submission form template
├── evaluator/
│   ├── evaluate.py             # Evaluation script
│   └── requirements.txt
├── leaderboard/
│   ├── app.py                  # Streamlit leaderboard
│   ├── results.json            # Scores database
│   └── requirements.txt
├── submission_handler/
│   └── process_submission.py   # Issue → eval pipeline
├── data/
│   └── README.md               # Info about dataset format
└── docs/
    └── baseline_model.py       # Starter baseline
```

---

## 📊 Evaluation Details

Models are evaluated on:
- **Primary**: Weighted F1-Score
- **Secondary**: Accuracy
- **Tertiary**: Inference Speed

The test set is **private** and never shared publicly.

---

## 🏆 Prizes

| Rank | Prize |
|------|-------|
| 🥇 1st | Certificate + Feature in paper |
| 🥈 2nd | Certificate |
| 🥉 3rd | Certificate |

---

## ❓ FAQ

**Q: Can I use pre-trained models?**  
A: Yes! Transfer learning is allowed and encouraged.

**Q: How many submissions per day?**  
A: Maximum 3 submissions per participant per day.

**Q: Can teams participate?**  
A: Yes, teams of up to 3 people. Use one GitHub account per team.

---

## 📬 Contact

Open a [Discussion](../../discussions) for any questions!
