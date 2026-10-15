📧 Spam Mail Detection using Machine Learning

A machine learning project that classifies SMS / email messages as spam or ham (not spam), with an interactive Streamlit web app that explains why a message was flagged.

🔗 Live demo: https://spam-mail-detector-ml.streamlit.app/

📌 Project Overview

Spam messages waste time and often carry scams and phishing links. This project builds a text classifier that learns the patterns of spam from labelled messages and flags new messages automatically.

Highlights

End-to-end workflow: data exploration → text cleaning → model training → evaluation → deployment.
Compared four models and evaluated them with precision, recall and F1 (not just accuracy).
Removed duplicate messages before splitting to avoid data leakage.
Interactive app with spam score, word highlighting, adjustable threshold and batch CSV prediction
📂 Dataset
File: spam mail.csv.
Size: 5,572 messages, 2 columns: Category (ham / spam) and Message.
Class balance: about 87% ham and 13% spam (imbalanced).
Duplicates: 415 duplicate messages removed, leaving 5,157 unique messages.
🛠️ Approach
Exploratory analysis: class balance, message length, duplicates. Spam messages are noticeably longer than ham.
Text cleaning: lowercase; replace URLs, emails, money amounts and phone numbers with tokens; remove punctuation.
Feature extraction: TF-IDF with unigrams and bigrams, English stop words removed.
Models compared: Multinomial Naive Bayes, Logistic Regression, Linear SVM, Random Forest.
Evaluation: stratified 80/20 train-test split plus 5-fold cross-validation, using precision, recall and F1 on the spam class.
Explainability: top words that push a message towards spam or ham.
Deployment: Streamlit app using the saved model (spam_classifier.joblib).
📊 Results

Test-set performance (spam = positive class):

Model	Accuracy	Precision	Recall	F1
Linear SVM	98.74%	96.75%	92.97%	0.948
Naive Bayes	98.74%	98.32%	91.41%	0.947
Logistic Regression	98.64%	95.97%	92.97%	0.944
Random Forest	98.55%	100.00%	88.28%	0.938

Best model (Linear SVM) on the test set: 119 of 128 spam messages caught, 4 genuine messages wrongly flagged.

5-fold cross-validation F1 scores are all between 0.945 and 0.952, so the results are stable.

Because only about 13% of messages are spam, a model that always predicts "ham" would already score about 87% accuracy. That is why precision, recall and F1 are used here.

🖥️ Streamlit App Features
Type or paste a message and get a SPAM / HAM result with a spam-score percentage
Example buttons for quick testing
Word highlighting showing which words pushed the prediction towards spam or ham
Threshold slider to make the filter stricter or more lenient
CSV upload to classify many messages at once and download the results
History log and session statistics

Add screenshots here, for example:

![Spam Mail Detector app]<img width="1364" height="634" alt="spam" src="https://github.com/user-attachments/assets/5e73af91-46d2-4837-b9c3-77c92f9becbd" />


🧰 Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib · Streamlit

📁 Project Structure
spam-mail-detection/
├── Spam_Mail_Detection.ipynb       # data analysis, training and evaluation
├── spam_app.py                     # Streamlit web app
├── spam_classifier.joblib          # trained model (TF-IDF + Linear SVM pipeline)
├── spam mail.csv                   # dataset
├── requirements.txt                # dependencies
└── README.md
🚀 How to Run Locally
bash
# 1. Clone the repository
git clone https://github.com/Dhanusri-analyst/spam-mail-detection.git
cd spam-mail-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the app
python3 -m streamlit run spam_app.py

To retrain the model, open Spam_Mail_Detection.ipynb in Jupyter and run all cells. The last cell saves a fresh spam_classifier.joblib.

☁️ Deploy for Free
Push this repository to GitHub.
Sign in at share.streamlit.io with your GitHub account.
Click Create app, select this repo, and set the main file to spam_app.py.
💡 Key Learnings
Handling imbalanced data and choosing the right metrics
Preventing data leakage by removing duplicates before splitting
Building a reproducible text pipeline with Pipeline so the vectorizer is fitted on training data only
Making a model explainable with linear model weights
Packaging and deploying a machine learning model as a web app
🔮 Future Improvements
Hyperparameter tuning with GridSearchCV
Try word embeddings or a transformer model such as DistilBERT
Add extra features such as message length, digit count and uppercase ratio
Train on a larger and more recent email dataset
👩‍💻 Author

Dhanusri 
B.Tech Information Technology · aspiring Data Analyst / Data Scientist

GitHub: Dhanusri-analyst
LinkedIn: www.linkedin.com/in/dhanusri-prabhakaran-19106b329

⭐ If you found this project useful, consider giving it a star!
