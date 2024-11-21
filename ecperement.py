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
from sklearn.svm import SVC
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
import pymorphy3
import spacy
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation

# from pytextrank import BaseTextRankt
from rake_nltk import Rake
import yake as YAKE
from keybert import KeyBERT

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from collections import Counter
from stopwords import get_stopwords

# Получаем русский список стоп-слов
custom_stopwords = set(get_stopwords('ru'))
english_alphabet = 'abcdefghijklmnopqrstuvwxyz'

nlp = spacy.load("ru_core_news_sm")
def create_graph(text, window_size=5):
    # Токенизация текста на предложения и слова
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]

    # Убираем стоп-слова и пунктуацию, лемматизируем слова
    morph = pymorphy3.MorphAnalyzer()
    words = [[morph.parse(word.lower())[0].normal_form for word in sentence
              if word.isalpha() and word.lower() not in custom_stopwords] for sentence in words]

    # Создаем граф
    G = nx.Graph()

    # Добавляем узлы в граф
    for sentence in words:
        for word in sentence:
            G.add_node(word)

    # Добавляем ребра между словами, которые встречаются в окне определенного размера
    for sentence in words:
        for i in range(len(sentence)):
            for j in range(i + 1, min(i + window_size + 1, len(sentence))):
                word1 = sentence[i]
                word2 = sentence[j]
                if G.has_edge(word1, word2):
                    G[word1][word2]['weight'] += 1
                else:
                    G.add_edge(word1, word2, weight=1)

    return G

import spacy

def get_keywords(G, text, num_keywords=5, kw_expert=[], max_n=3):
    degrees = dict(G.degree())

    # Load the spacy model for Russian

    # Process the text with spacy
    doc = nlp(text)

    # Extract keywords using dependency parsing
    keywords = [token.text for token in doc if token.dep_ in ["nsubj", "dobj"]]

    # Generate all n-grams of the specified length in the text
    words = word_tokenize(text)
    grams = []
    for n in range(1, max_n + 1):
        grams.extend(ngrams(words, n))
    # Filter n-grams based on the condition: do not include n-grams consisting of stop-words
    grams = [' '.join(gram) for gram in grams if all(word.isalpha() and word.lower() not in custom_stopwords for word in gram) and len(gram) >= 1]

    # Count the frequency of n-grams in the text
    ngram_freq = Counter(grams)

    # Calculate the weighted degrees for n-grams
    weighted_degrees = {}
    for gram, freq in ngram_freq.items():
        gram_words = gram.split()
        weighted_degrees[gram] = freq * sum(degrees.get(word, 0) for word in gram_words)

    # Sort n-grams by weighted degrees
    sorted_keywords = sorted(weighted_degrees.items(), key=lambda x: x[1], reverse=True)

    # Get the keywords
    if (num_keywords != -1):
        keywords.extend([kw for kw, _ in sorted_keywords[:num_keywords]])
    else:
        keywords.extend([kw for kw, _ in sorted_keywords])
    keywords.extend(kw_expert)

    # Remove duplicates
    keywords = list(set(keywords))
    return keywords




def ngram_pos_features(ngram):
    # Создаем объект для анализа текста на русском языке
    morph = pymorphy3.MorphAnalyzer()

    # Токенизируем n-грамму
    tokens = nltk.word_tokenize(ngram)

    # Определяем части речи для каждого токена
    pos_tags = [morph.parse(token)[0].tag.POS for token in tokens]

    # Создаем словарь, где ключ - часть речи, а значение - количество ее вхождений
    pos_counts = Counter(pos_tags)

    # Создаем список категориальных фичей
    features = []
    # Перебираем все возможные части речи
    for pos in ['ADJF', 'ADJS', 'COMP', 'CONJ', 'INTJ', 'NOUN', 'NUMR', 'PRTF', 'PRTS', 'VERB', 'NPRO', 'PRED', 'PREP', 'PRCL', 'ADVB', 'INFN', 'GRND', 'LATN']:
        # Добавляем количество вхождений части речи в список фичей
        features.append(pos_counts.get(pos, 0))

    # Добавляем тег для английского языка
    features.append(sum([1 if werb[0].lower() in english_alphabet else 0 for werb in ngram.split()]))

    return features

def FOF(keyword, text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    keywords = nltk.word_tokenize(keyword)
    position = -1

    for i, word in enumerate(words):
        if(word == keywords[0]):
            flag_true = True
            position = i
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

def is_capitalized(text, words):
    # Split the text into words or phrases
    phrases = text.split('. ')
    word = ' '.join([word.capitalize() for word in words.split()])
    for phrase in phrases:
        if word in phrase:
            if not phrase.startswith(word.capitalize()):
                return 1
    return 0

def is_abbreviation(text, word):
    # Split the text into words
    words = text.split()

    # Check if the word is an abbreviation
    for w in words:
        if word.lower() in w.lower():
            if len(word) <= len(w) and w.isupper():
                return 1
    return 0

def is_named_entity(word):

    # Process the word with spacy
    doc = nlp(word)

    # Check if the word is a named entity
    if len(doc.ents) > 0:
        return 1
    else:
        return 0

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
                ngram_scores[ngram] = [ngram_avg_tfidf_score, len(ngram.split()), *ngram_pos_features(ngram), FOF(ngram, documents[i]), is_in_headlines(ngram, documents[i]), is_capitalized(documents[i], ngram), is_abbreviation(documents[i], ngram), is_named_entity(ngram)]
            else:
                ngram_scores[ngram] = [0, len(ngram.split()), *ngram_pos_features(ngram), FOF(ngram, documents[i]), is_in_headlines(ngram, documents[i]), is_capitalized(documents[i], ngram), is_abbreviation(documents[i], ngram), is_named_entity(ngram)]
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


def create_dataset_of_features(name_dataset):
    if(type(name_dataset) is list):
        documents = []
        labels = []
        for name in name_dataset:
            df = pd.read_csv("datasets_csv\\" + name)

            documents += df["text"].to_list()
            labels += df["keys"].apply(literal_eval).to_list()
    else:
        df = pd.read_csv("datasets_csv\\" + name_dataset)

        documents = df["text"].to_list()
        labels = df["keys"].apply(literal_eval).to_list()

    # Preprocess
    i = 0
    print("check docs")
    while i != len(documents):
        if(type(documents[i]) is float):
            del documents[i]
            del labels[i]
        else:
            i+=1

    keyword_extra = []
    for i, document in tqdm(enumerate(documents), desc="extractiom kw candidats", total=len(documents)):
        keywords = get_keywords(kw_expert=labels[i], text=document, G=create_graph(document, 3), num_keywords=3)
        keyword_extra.append(keywords)

    # Feature
    features = feature_extraction(documents, keyword_extra)
    keyword_extra = [features_dict.keys() for features_dict in features]
    print(len(keyword_extra), len(labels))
    labels = [keyword_vector(labels[i], keyword_extra[i]) for i in range(len(labels))]
    df_features = pd.DataFrame()
    # Проходим по каждому элементу списка
    i = 0
    for item in features:
        # Для каждого ключа в элементе
        for key, value in item.items():
            # Добавляем столбец в DataFrame
            l = list(value)
            l.append(key)
            df_features[i] = l
            i+=1


    df_features = df_features.T
    df_features = df_features.rename(columns={0: 'TF_IDF', 1: 'Len_NG', 2: 'ADJF', 3:'ADJS', 4:'COMP', 5:'CONJ', 6:'INTJ', 7:'NOUN', 8:'NUMR', 9:'PRTF', 10:'PRTS', 11:'VERB', 12:'NPRO', 13:'PRED', 14:'PREP', 15:'PRCL', 16:'ADVB', 17:'INFN', 18:'GRND', 19 :'LATN',20: 'ENG', 21: 'FOF', 22: 'is_in_headlines', 23: 'is_capitalized', 24: 'is_abbreviation', 25: 'is_named_entity', 26: 'word'})
    df_features['labels'] = np.concatenate(labels)
    print(df_features)
    df_features.to_csv("features\\" + "_".join([name.replace(".csv", "") for name in name_dataset])+ ".csv")

def learn_SVC(dataset_name):
    df_features = pd.read_csv("features\\" + dataset_name, index_col='Unnamed: 0')

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(df_features.drop(columns=['word', 'labels']), df_features['labels'], test_size=0.2, random_state=42)

    # Балансировка классов с использованием метода перевыборки класса меньшего размера
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    print("SVC")
    clf = SVC(kernel='rbf')
    clf.fit(X_train_res, y_train_res)

    # Предсказание на тестовой выборке
    y_pred_clf = clf.predict(X_test)

    # Оценка качества модели
    print("SVC Classifier:")
    print(classification_report(y_test, y_pred_clf))

    # Save model
    joblib.dump(clf, "SVM_MODEL.joblib.pkl", compress=9)
    
    return "SVM_MODEL.joblib.pkl"

def learn_NBC(dataset_name):
    df_features = pd.read_csv("features\\" + dataset_name, index_col='Unnamed: 0')

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(df_features.drop(columns=['word', 'labels']), df_features['labels'], test_size=0.2, random_state=42)

    # Балансировка классов с использованием метода перевыборки класса меньшего размера
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    nb = MultinomialNB()
    nb.fit(X_train_res, y_train_res)

    # Предсказание на тестовой выборке
    y_pred_nb = nb.predict(X_test)

    # Оценка качества модели
    print("Naive Bayes Classifier:")
    print(classification_report(y_test, y_pred_nb))

    # Save model
    joblib.dump(nb, "NB_MODEL.joblib.pkl", compress=9)

    return "NB_MODEL.joblib.pkl"

def learn_RF(dataset_name):
    df_features = pd.read_csv("features\\" + dataset_name, index_col='Unnamed: 0')

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(df_features.drop(columns=['word', 'labels']), df_features['labels'], test_size=0.2, random_state=42)

    # Балансировка классов с использованием метода перевыборки класса меньшего размера
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    # Тренировка модели Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_res, y_train_res)

    # Предсказание на тестовой выборке
    y_pred_rf = rf.predict(X_test)

    # Оценка качества модели
    print("Random Forest Classifier:")
    print(classification_report(y_test, y_pred_rf))

    # Save model
    joblib.dump(rf, "RF_MODEL.joblib.pkl", compress=9)

    return "RF_MODEL.joblib.pkl"

# Main function
def main():
    df = pd.read_csv("datasets_csv\\habrahabr.csv")
    # creating dataset of featers from text
    create_dataset_of_features("russia_today.csv")

    # learning SVC 
    # learn_SVC("habrahabr.csv")

    # # learnin NBC
    # learn_NBC("habrahabr.csv")

    # learning RF
    # learn_RF("habrahabr.csv")

    # from transformers import BertModel, BertTokenizer
    # # vectorizer = joblib.load("VECTORIZER.joblib.pkl")
    # # lda = LatentDirichletAllocation(n_components=1)
    # clf = joblib.load("SVM_MODEL.joblib.pkl")
    # # nb = joblib.load("NB_MODEL.joblib.pkl")
    # rf = joblib.load("RF_MODEL.joblib.pkl")
    # text_test = df["text"][1005:1007].to_list()
    # # text_test = ["Это заявление Трамп сделал на фоне новостей о том, что представители Сеула и Пхеньяна провели 3 января телефонный разговор, а глава КНДР Ким Чен Ын дал согласие возобновить диалог на высоком уровне, что положительно оценили его соседи. Ранее в своём новогоднем обращении северокорейский лидер, с одной стороны, в очередной раз пригрозил Соединённым Штатам нанести ядерный удар в случае агрессии против Пхеньяна, а с другой — призвал к диалогу с Югом. Не все рады"]
    # # G = create_graph(text_test[0], 3)
    # # nx.draw(G, with_labels = True)
    # # plt.show()
    # k_test = [get_keywords(G = create_graph(text_test_doc, window_size=3), text = text_test_doc, num_keywords=-1, max_n=3) for text_test_doc in text_test]
    # features_test = feature_extraction(documents=text_test, keyword_extr=k_test)
    # print("_"*30)
    # start = 1005

    # model_name = "DeepPavlov/rubert-base-cased"
    # model = BertModel.from_pretrained(model_name)

    # # Инициализация KeyBERT
    # kw_model_bert = KeyBERT(model=model)

    # # tr = TextRank()
    # # nlp_d = spacy.load("ru_core_news_sm")
    # # nlp_d.add_pipe(tr.PipelineComponent, name="textrank", last=True)

    # stop_words = set(stopwords.words('russian'))
    # rake = Rake(stopwords=stop_words, min_length=1, max_length=3)
    # yake = YAKE.KeywordExtractor()

    
    # with open("rezult_test\\svc_experem.txt", 'w', encoding="utf-8") as out_file:
    #     for i in range(len(text_test)):
    #         # X_new = vectorizer.transform(list(text_test[i]))
    #         # topics_lda = lda.fit_transform(X_new)
    #         predicted = clf.decision_function(list(features_test[i].values()))
    #         predicted_rf = rf.predict_proba(list(features_test[i].values()))

    #         probabilities_class_1_rf = predicted_rf[:, 1]
    #         sorted_indices_rf = np.argsort(probabilities_class_1_rf)[::-1]

    #         original_indices_rf = np.arange(len(list(features_test[i].keys())))[sorted_indices_rf[:40]]

    #         sorted_indices = np.argsort(predicted)[::-1]

    #         original_indices = np.arange(len(list(features_test[i].keys())))[sorted_indices[:40]]
    #         out_file.write(text_test[i])


    #         out_file.write('\n' + '_' * 20 + '\n')
    #         out_file.write('predicted SVC\n')
    #         for j in original_indices:
    #             out_file.write(list(features_test[i].keys())[int(j)])
    #             out_file.write(" ")
    #             out_file.write(str(predicted[int(j)]))
    #             out_file.write(" | ")

    #         out_file.write('\n' + '_' * 20 + '\n')
    #         out_file.write('predicted\n')
    #         for j in original_indices_rf:
    #             out_file.write(list(features_test[i].keys())[int(j)])
    #             out_file.write(" ")
    #             out_file.write(str(probabilities_class_1_rf[int(j)]))
    #             out_file.write(" | ")


    #         # out_file.write('\n' + '_' * 20 + '\n')
    #         # out_file.write('predicted LDA\n')
    #         # for j in range(lda.n_components):
    #         #     # Получение топ-10 слов для каждой темы
    #         #     top_words = [vectorizer.get_feature_names_out()[num] for num in lda.components_[j].argsort()[:-10 - 1:-1]]
    #         #     out_file.write(f"Topic {j}: {top_words}\n")

    #         keywords_bert = kw_model_bert.extract_keywords(text_test[i], keyphrase_ngram_range=(1, 3), stop_words=None, use_maxsum=True, nr_candidates=20, top_n=5)
    #         rake.extract_keywords_from_text(text_test[i])
    #         keywords_rake = rake.get_ranked_phrases()
    #         # doc = nlp_d(text_test[i])
    #         # keywords_PTR = [p.text for p in doc._.phrases]
    #         keywords_yake = yake.extract_keywords(text_test[i])
            
    #         out_file.write('\nbert\n')
    #         out_file.write(str({keyword: score for keyword, score in keywords_bert}))
    #         out_file.write('\nrake\n')
    #         out_file.write(str(keywords_rake))
    #         # out_file.write('\npy_text_rank\n')
    #         # out_file.write(keywords_PTR)
    #         out_file.write('\nyake\n')
    #         out_file.write(str(keywords_yake))

    #         out_file.write('\n' + '_' * 20 + '\n')
    #         out_file.write('true\n')
    #         out_file.write(df['keys'][start + i])
    #         out_file.write('\n')

if __name__ == "__main__":
    main()