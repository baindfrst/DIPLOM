import spacy
from collections import Counter

# Load the spacy model for Russian
nlp = spacy.load('ru_core_news_sm')

def get_keywords(text, num_keywords=5, max_n=3):
    # Process the text with spacy
    text = text.replace('\n', '')
    doc = nlp(text)


    # Extract keywords and n-grams using semantic analysis
    keywords = []
    for i in range(1, max_n+1):
        n_grams = [doc[j:j+i].text for j in range(len(doc)-i+1)]
        for n_gram in n_grams:
            # Check if the n-gram is a noun or proper noun and does not contain punctuation or hyphens
            if any(token.pos_ in ["NOUN", "PROPN"] for token in doc[doc.text.find(n_gram):doc.text.find(n_gram)+i]) and not any(token.is_punct or token.text == '-' for token in doc[doc.text.find(n_gram):doc.text.find(n_gram)+i]):
                # Add the n-gram to the list of keywords
                keywords.append(n_gram)

    # Sort the keywords by their frequency and select the top num_keywords
    keyword_freq = Counter(keywords)
    top_keywords = [keyword for keyword, freq in keyword_freq.most_common(num_keywords)]

    return top_keywords

text = """
Воспитанные люди должны удовлетворять следующим условиям:

… Они уважают человеческую личность, всегда снисходительны, мягкие, вежливые, уступчивые…

… Они уважают чужую собственность, а потому платят долги.

… Не лгут даже в пустяках… Они не лезут с откровенностями, когда их не спрашивают…

… Они не унижают себя с тою целью, чтобы вызвать в другом сочувствие…

… Они не суетны…

… Если имеют в себе талант, то уважают его… Они жертвуют для него всем…

… Они воспитывают в себе эстетику*…

    … Тут нужны беспрерывные дневной и ночной труд, вечное чтение, воля… Тут дорог каждый час.
"""

print(get_keywords(text))