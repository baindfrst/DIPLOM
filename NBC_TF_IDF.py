from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv("datasets_csv\\cyberleninka.scv")
print(df['keys'])
len(df['text'])
documents = df['text'].to_list()[:300]
keywords = df['keys'].apply(eval).tolist()[:300]

D_alot = []
K_alot = []

for i, k_arr in enumerate(keywords):
    for word in k_arr:
        D_alot.append(documents[i])
        K_alot.append(word)

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(D_alot).toarray()

# Convert the list of keywords into a numerical format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(K_alot)

# Train a Naive Bayes classifier on the data
classifier = GaussianNB()
classifier.fit(X, y)

n_t = df['text'].to_list()[1800:1820]
n_k = df['keys'].apply(eval).tolist()[1800:1820]

new_X = tfidf_vectorizer.transform(n_t).toarray()
proba = classifier.predict_proba(new_X)
top_3_indices = np.argsort(proba, axis=1)[:, -10:]
top_3_keywords = []
for row in top_3_indices:
    keywords = label_encoder.inverse_transform(row)
    top_3_keywords.append(keywords)

file = open("rezult_test\\cyberleninka\\NBC_TF_IDF.txt", 'w', encoding='utf-8')

for i in range(len(n_k)):
    file.write(n_t[i])
    file.write('\n')
    file.write("extracted:" + str(top_3_keywords[i]))
    file.write('\n')
    file.write("marked:" + str(n_k[i]))
    file.write('\n')
