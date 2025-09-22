# Twitter-Sentiment-Analysis-with-Transformers


This project fine-tunes a **Transformers** model to perform sentiment analysis on tweets.
We use **DistilBERT** as the base model for text classification into positive and negative categories.

---

## 📊 Dataset

The dataset comes from [Twitter Sentiment Analysis (Kaggle/GitHub)](https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv).
It contains over **31,000 tweets**, split into:

* \~25,000 samples for training
* \~6,000 samples for evaluation

Main columns:

* `tweet`: text of the tweet
* `label`: sentiment (0 = negative, 1 = positive)

---

## 🧠 Model

We use the pre-trained model **[distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)** with a classification head added to match the number of sentiment classes.

---

## ⚙️ Training

* **Main libraries**:

  * `transformers`
  * `datasets`
  * `torch`
  * `scikit-learn`
  * `pandas`

* **Key hyperparameters**:

  * `learning_rate = 2e-5`
  * `batch_size = 8`
  * `epochs = 2`
  * `weight_decay = 0.01`

---

## 📈 Results

After 2 epochs of training on CPU:

* **Training Loss**: 0.057
* **Validation Loss**: 0.114
* **Accuracy**: \~96%

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/twitter-sentiment-transformers.git
cd twitter-sentiment-transformers
pip install -r requirements.txt
```

Recommended `requirements.txt`:

```
transformers
datasets
torch
pandas
scikit-learn
evaluate
```

---

## ▶️ Usage

Run the training script:

```bash
python train.py
```

This trains the model and saves checkpoints in the `test_trainer/` folder.

---

## 🔮 Next steps

* Run training on **GPU (Google Colab)** to reduce training time.
* Extend to multi-class sentiment analysis (neutral, sarcasm, etc.).
* Deploy as an API for real-time predictions.

---

## 👨‍💻 Author

Project developed by **Leandro Ariel Divano**


---
