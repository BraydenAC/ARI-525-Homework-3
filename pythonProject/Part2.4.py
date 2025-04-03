import random

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_parquet("hf://datasets/ucirvine/sms_spam/plain_text/train-00000-of-00001.parquet")

df_X = df["sms"]
df_y = df["label"]

vectorizer = CountVectorizer()
vectorizer.fit(df_X)
BoW_X = vectorizer.transform(df_X)

train_X, dev_test_X, train_y, dev_test_y = train_test_split(BoW_X, df_y, test_size=0.2, random_state=42)
dev_X, test_X, dev_y, test_y = train_test_split(dev_test_X, dev_test_y, test_size=0.5, random_state=42)
print("Dataset loaded...")

#BoW Baseline
BoW_LogReg = LogisticRegression(penalty='l2')
BoW_LogReg.fit(train_X, train_y)
LogReg_preds = BoW_LogReg.predict(test_X)
print("Logistic Regression Baseline")
print(classification_report(test_y, LogReg_preds))


#majority/target-class Baseline
majority_preds = [0, 0]
for entry in train_y:
    if entry == 0:
        majority_preds[0] += 1
    elif entry == 1:
        majority_preds[1] += 1
    else: print("Problem with majority prediction")
final_pred = 1
if majority_preds[0] >= majority_preds[1]:
    final_pred = 0
majority_final_pred_list = []
for entry in test_X:
    majority_final_pred_list.append(final_pred)
print("Majority Baseline")
print(classification_report(test_y, majority_final_pred_list))


#Random Baseline
random_y_preds = []
for entry in test_X:
    random_y_preds.append(random.randrange(2))
print("Random Baseline")
print(classification_report(test_y, random_y_preds))