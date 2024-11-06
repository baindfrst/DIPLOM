from gensim.models import Word2Vec
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

# Tokenize the texts
tokenized_documents = [doc.split() for doc in D_alot]

# Train a word2vec model on the tokenized documents
word2vec_model = Word2Vec(tokenized_documents, vector_size=100, window=5, min_count=1, workers=4)

# Convert the list of texts into a matrix of word2vec features
X = np.array([np.mean([word2vec_model.wv[word] for word in doc if word in word2vec_model.wv]
                      or [np.zeros(100)], axis=0) for doc in tokenized_documents])

# Convert the list of keywords into a numerical format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(K_alot)

# Train a Naive Bayes classifier on the data
print("start fit")
classifier = GaussianNB()
classifier.fit(X, y)
print("end fit")

n_t = df['text'].to_list()[1800:1820]
n_k = df['keys'].apply(eval).tolist()[1800:1820]

# Tokenize the new texts
tokenized_new_texts = [doc.split() for doc in n_t]

# Convert the list of new texts into a matrix of word2vec features
new_X = np.array([np.mean([word2vec_model.wv[word] for word in doc if word in word2vec_model.wv]
                          or [np.zeros(100)], axis=0) for doc in tokenized_new_texts])

proba = classifier.predict_proba(new_X)
top_3_indices = np.argsort(proba, axis=1)[:, -10:]
top_3_keywords = []
for row in top_3_indices:
    keywords = label_encoder.inverse_transform(row)
    top_3_keywords.append(keywords)

file = open("rezult_test\\cyberleninka\\NBC_W2V.txt", 'w', encoding='utf-8')

for i in range(len(n_k)):
    file.write(n_t[i])
    file.write('\n')
    file.write("extracted:" + str(top_3_keywords[i]))
    file.write('\n')
    file.write("marked:" + str(n_k[i]))
    file.write('\n')
