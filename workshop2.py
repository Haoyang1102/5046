import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def load_data(file_path):
    messages = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            messages.extend(data["messages"])
            labels.extend(data["sender_labels"])
    return messages, np.array(labels)

train_file = "mod-train.jsonl"
val_file = "mod-validation.jsonl"
test_file = "mod-test.jsonl"

X_train, y_train = load_data(train_file)
X_val, y_val = load_data(val_file)
X_test, y_test = load_data(test_file)



vectorizer = TfidfVectorizer()
#实例化了一个 TfidfVectorizer 对象
X_train_tfidf = vectorizer.fit_transform(X_train)#1.建立词汇表；2.把每个文本转换为对应的 TF-IDF 特征向量
X_val_tfidf = vectorizer.transform(X_val)#这两行代码使用之前在训练数据上已经拟合好的 vectorizer（即已经建立好的词汇表），对验证集文本数据 X_val 和测试集文本数据 X_test 进行转换，将它们也转换为 TF-IDF 特征向量，分别存储在 X_val_tfidf 和 X_test_tfidf 中
X_test_tfidf = vectorizer.transform(X_test)

print(X_test_tfidf)

model = LogisticRegression(penalty = 'l2', solver='saga', max_iter=1000)


model.fit(X_train_tfidf, y_train)


y_val_pred = model.predict(X_val_tfidf)
print("Validation Set Performance:")
print(classification_report(y_val, y_val_pred))

y_test_pred = model.predict(X_test_tfidf)
print("Test Set Performance:")
print(classification_report(y_test, y_test_pred))

