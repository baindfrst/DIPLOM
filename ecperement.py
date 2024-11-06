#3 фичи отвечающих о том, есть ли кандидат в названии докупента, его кратком содержании abstract и назавании разделов
from tqdm import tqdm
import re
import nltk
import itertools
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import numpy as np
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import string
from nltk.util import ngrams
import pandas as pd
import joblib
from ast import literal_eval
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing
def create_graph(text, n=1):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]

    # Remove stopwords and punctuation, and lemmatize words
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # words = [[lemmatizer.lemmatize(word.lower()) for word in sentence if word.isalpha() and word.lower() not in stop_words and word.lower() not in string.punctuation] for sentence in words]
    words = [[word.lower() for word in sentence if word.isalpha() and word.lower() not in stop_words and word.lower() not in string.punctuation] for sentence in words]

    # Create a graph
    G = nx.Graph()

    # Add edges and nodes
    for sentence in words:
        # Generate n-grams
        grams = ngrams(sentence, n)
        for gram in grams:
            # Join the words in the gram into a single string
            gram_str = ' '.join(gram)
            # Add the gram as a node to the graph
            G.add_node(gram_str)
            # Add edges between the gram and each of its constituent words
            for word in gram:
                if G.has_edge(gram_str, word):
                    G[gram_str][word]['weight'] += 1
                else:
                    G.add_edge(gram_str, word, weight=1)

    return G

def get_keywords(G, num_keywords=5, kw_exeprt = []):
    degrees = dict(G.degree())

    weighted_degrees = {node: degree * len(node.split()) for node, degree in degrees.items()}

    sorted_nodes = sorted(weighted_degrees, key=weighted_degrees.get, reverse=True)

    keywords = sorted_nodes[:num_keywords]
    keywords += kw_exeprt
    return keywords

def encode_ngram_with_pos_tags(ngram):
    pos_tag_to_number = {
    'NN': 1,  # noun, singular or mass
    'NNS': 1,  # noun, plural
    'JJ': 2,  # adjective or numeral, ordinal
    'VB': 3,  # verb, base form
    'VBD': 3,  # verb, past tense
    'VBG': 3,  # verb, gerund or present participle
    'VBN': 3,  # verb, past participle
    'VBP': 3,  # verb, non-3rd person singular present
    'VBZ': 3,  # verb, 3rd person singular present
    'PRP': 4,  # personal pronoun
    'PRP$': 4,  # possessive pronoun
    'IN': 5,  # preposition or subordinating conjunction
    'DT': 6,  # determiner
    'CC': 7,  # coordinating conjunction
    'CD': 8,  # cardinal number
    'MD': 9,  # modal
    'RP': 10,  # particle
    'TO': 11,  # "to"
    'EX': 12,  # existential there
    'FW': 13,  # foreign word
    'LS': 14,  # list item marker
    'PDT': 15,  # predeterminer
    'POS': 16,  # possessive ending
    'RBR': 17,  # adverb, comparative
    'RBS': 18,  # adverb, superlative
    'RB': 19,  # adverb
    'WRB': 20,  # wh-adverb
    'UH': 21,  # interjection
    'SYM': 22,  # symbol
    '.': 23,  # punctuation mark, sentence closer
    ',': 24,  # punctuation mark, comma
    ':': 25,  # punctuation mark, colon
    '(': 26,  # punctuation mark, left parenthesis
    ')': 27,  # punctuation mark, right parenthesis
    '``': 28,  # punctuation mark, left quote
    "''": 29,  # punctuation mark, right quote
    '$': 30,  # dollar
    '#': 31,  # pound
    }


    # Tokenize the n-gram into individual words
    tokenized_ngram = nltk.word_tokenize(ngram)

    # Perform POS tagging on the tokenized n-gram
    pos_tagged_ngram = nltk.pos_tag(tokenized_ngram)

    # Extract the words and POS tags for the n-gram
    words_and_pos_tags = [(word, pos_tag_to_number.get(tag, 0)) for word, tag in pos_tagged_ngram]

    # Convert the list of words and POS tags into a list of numbers
    encoded_ngram = [pos_tag for word, pos_tag in words_and_pos_tags]
    # Concatenate the numbers in the list into a single number
    encoded_ngram_num = int(''.join(map(str, encoded_ngram)))

    return encoded_ngram_num

def FOF(keyword, text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    keywords = nltk.word_tokenize(keyword)
    position = -1

    for i, word in enumerate(words):
        if(word == keywords[0]):
            flag_true = True
            for j, keyword in enumerate(keywords):
                if(i+j <= len(words) - 1):
                    if(words[i+j] != keyword):
                        flag_true = False
                else:
                    flag_true = False
            if(flag_true):
                return position / len(words)

    return -1

def extract_headlines(text):
    headlines = re.findall(r'^(?:[A-Z0-9]+: .*$|^\d+\.\d+\s[A-Z].*$)', text, flags=re.MULTILINE)
    return headlines

def is_in_headlines(kw, text):
    heads = extract_headlines(text)
    for head in heads:
        if kw in head.lower():
            return 1
    return 0
# Feature extraction

def feature_extraction(documents, keyword_extr):
    # Create a TfidfVectorizer object with ngram_range=(1, 3)
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))

    # Fit the vectorizer to the documents
    vectorizer.fit(documents)

    # Get the feature names (i.e., the n-grams)
    feature_names = vectorizer.get_feature_names_out()
    transformed_docs = vectorizer.transform(documents).toarray()

    # Initialize a dictionary to store the TF-IDF scores for each n-gram
    # Iterate over the n-grams in keyword_extr
    out_arr = []
    for i, doc_keyword_extr in tqdm(enumerate(keyword_extr), desc = "extraction fuetures doc: ", total=len(documents)):
        ngram_scores = {}
        for ngram in tqdm(doc_keyword_extr, desc="ngram processing", total = len(doc_keyword_extr)):
            # Check if the n-gram is a feature name
            if ngram in feature_names:
                # Get the index of the n-gram in the feature names list
                ngram_index = np.where(feature_names == ngram)[0][0]

                # Get the TF-IDF scores for the n-gram in all documents
                ngram_tfidf_scores = transformed_docs[:, ngram_index]

                # Compute the average TF-IDF score for the n-gram
                ngram_avg_tfidf_score = ngram_tfidf_scores.mean()

                # Add the average TF-IDF score for the n-gram to the dictionary
                ngram_scores[ngram] = [ngram_avg_tfidf_score, len(ngram.split()), encode_ngram_with_pos_tags(ngram), FOF(ngram, documents[i]), is_in_headlines(ngram, documents[i])]
            else:
                ngram_scores[ngram] = [0, encode_ngram_with_pos_tags(ngram), len(ngram.split()), FOF(ngram, documents[i]), is_in_headlines(ngram, documents[i])]
        out_arr.append(ngram_scores)

    return out_arr


def keyword_vector(keywords, keywords_candidats):

    # Initialize the binary vector
    vector = [0] * len(keywords_candidats)

    # Set the corresponding element in the vector to 1 if the keyword is present in the document
    for i, word in enumerate(keywords_candidats):
        if word in keywords:
            vector[i] = 1
        else:
            vector[i] = 0
    return vector

def RBF(X, gamma):
    
    # Free parameter gamma
    if gamma == None:
        gamma = 1.0/X.shape[1]
        
    # RBF kernel Equation
    K = np.exp(-gamma * np.sum((X - X[:,np.newaxis])**2, axis = -1))
    
    return K

# Main function
def main():
    df = pd.read_csv("datasets_csv\\Inspec.scv")

    # documents = df["text"][:100].to_list()
    # labels = df["keys"][:100].apply(literal_eval).to_list()

    # # Preprocess
    # keyword_extra = [get_keywords(kw_exeprt=labels[i], G = create_graph(document, 3), num_keywords=10) for i, document in tqdm(enumerate(documents), desc = "extractiom kw candidats", total=len(documents))]

    # # Feature
    # features = feature_extraction(documents, keyword_extra)
    # keyword_extra = [features_dict.keys() for features_dict in features]
    # labels = [keyword_vector(labels[i], keyword_extra[i]) for i in range(len(labels))]
    # features = [list(feature.values()) for feature in features]
    # features_concat = np.concatenate(features)
    # labels_concat = np.concatenate(labels)

    # clf = svm.SVC(probability=True)

    # # Train the model using the training sets
    # clf.fit(features_concat, labels_concat)

    # # Save model
    # joblib.dump(clf, "SVM_MODEL.joblib.pkl", compress=9)


    clf = joblib.load("SVM_MODEL.joblib.pkl")
    text_test = df["text"][101:104].to_list()
    # G = create_graph(text_test[0], 3)
    # nx.draw(G, with_labels = True)
    # plt.show()
    print(extract_headlines(text_test[0]))
    k_test = [get_keywords(G = create_graph(text_test_doc, 3), num_keywords=40) for text_test_doc in text_test]
    features_test = feature_extraction(documents=text_test, keyword_extr=k_test)
    print("_"*30)
    start = 101
    with open("rezult_test\\svc_experem.txt", 'w', encoding="utf-8") as out_file:
        for i in range(len(text_test)):
            predicted = clf.predict_proba(list(features_test[i].values()))
            probabilities_class_1 = predicted[:, 1]
            sorted_indices = np.argsort(probabilities_class_1)[::-1]
            original_indices = np.arange(len(list(features_test[i].keys())))[sorted_indices[:20]]
            out_file.write(text_test[i])


            out_file.write('\n' + '_' * 20 + '\n')
            out_file.write('predicted\n')
            for j in original_indices:
                out_file.write(list(features_test[i].keys())[int(j)])
                out_file.write(" | ")
            
            out_file.write('\n' + '_' * 20 + '\n')
            out_file.write('true\n')
            out_file.write(df['keys'][start + i])

if __name__ == "__main__":
    main()