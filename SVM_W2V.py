import spacy
from collections import Counter
from nltk import ngrams
import nltk

# Пример данных
documents = [
    "Clausal resolution in a logic of rational agency\
    A resolution based proof system for a Temporal Logic of Possible Belief is\
	presented. This logic is the combination of the branching-time temporal\
	logic CTL (representing change over time) with the modal logic KD45\
	(representing belief). Such combinations of temporal or dynamic logics\
	and modal logics are useful for specifying complex properties of\
	multi-agent systems. Proof methods are important for developing\
	verification techniques for these complex multi-modal logics.\
	Soundness, completeness and termination of the proof method are shown\
	and simple examples illustrating its use are given",

    "Local search with constraint propagation and conflict-based heuristics\
Search algorithms for solving CSP (Constraint Satisfaction Problems) usually\
	fall into one of two main families: local search algorithms and\
	systematic algorithms. Both families have their advantages. Designing\
	hybrid approaches seems promising since those advantages may be\
	combined into a single approach. In this paper, we present a new hybrid\
	technique. It performs a local search over partial assignments instead\
	of complete assignments, and uses filtering techniques and\
	conflict-based techniques to efficiently guide the search. This new\
	technique benefits from both classical approaches: a priori pruning of\
	the search space from filtering-based search and possible repair of\
	early mistakes from local search. We focus on a specific version of\
	this technique: tabu decision-repair. Experiments done on open-shop\
	scheduling problems show that our approach competes well with the best\
	highly specialized algorithms"
]

# Ключевые слова для каждого документа
keywords = [
    ['resolution based proof system', 'temporal logic', 'branching-time temporal logic', 'CTL', 'modal logic', 'KD45', 'belief', 'dynamic logics', 'multi-agent systems', 'multi-modal logics', 'rational agents', 'formal logic', 'multi-agent systems', 'temporal logic'],
    ['search algorithms', 'CSP', 'Constraint Satisfaction Problems', 'local search algorithms', 'systematic algorithms', 'partial assignments', 'filtering techniques', 'tabu decision-repair', 'constraint handling', 'search problems']
]

# Загрузка модели spaCy
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    # Разбиение текста на предложения
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    keywords = []
    for sentence in sentences:
        # Токенизация и пометка частей речи
        sent_doc = nlp(sentence)
        print(f"sent_doc = {sent_doc}")
        # Анализ зависимостей между словами
        dependencies = [(token.text, token.dep_) for token in sent_doc]
        print(f"dependencies = {dependencies}")
        # Построение триграмм
        trigrams = ngrams(sent_doc, 3)
        for i in trigrams:
            print(f"trigr:{i}")
        # Выбор кандидатов в ключевые слова на основе триграмм и зависимостей
        candidates = [trigram[1].text for trigram in trigrams if trigram[1].dep_ in ["nsubj", "dobj"]]
        print(f"candidates = {candidates}")
        # Подсчет частоты кандидатов в ключевые слова
        keyword_freq = Counter(candidates)
        print(f"keyword_freq = {keyword_freq}")
        # Выбор наиболее часто встречающихся кандидатов в ключевые слова
        sentence_keywords = [keyword for keyword, freq in keyword_freq.items() if freq > 1]
        print(f"sentence_keywords = {sentence_keywords}")
        keywords.extend(sentence_keywords)

    return keywords

# Пример использования
text = "Natural language processing (NLP) is a subfield of artificial intelligence (AI) that deals with the interaction between computers and humans through language. It involves analyzing, understanding, and generating human language using algorithms and statistical models."
keywords = extract_keywords(text)
print(keywords)