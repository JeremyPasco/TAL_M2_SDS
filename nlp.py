from models import Base, Author, Article
from sqlalchemy import func
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

#stopwords = nltk.corpus.stopwords.words('english')

stemmer = SnowballStemmer("english")

def get_data(db_session, maxAuthors = 10000, type = "abstract"):

    if type == "abstract":
        data = db_session.query(Author.id, Author.firstName, Author.lastName, Author.category,
                           func.group_concat(Article.abstract).label('text')).filter(Article.authors).group_by(Author.id).limit(maxAuthors)
    else:
        data = db_session.query(Author.id, Author.firstName, Author.lastName, Author.category,
                            func.group_concat(Article.title).label('text')).filter(Author.id, Article.id).filter(Article.authors).group_by(Author.id).limit(maxAuthors)

    return data

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def tfidf_vectorize(data, max_df = 1.0, min_df = 1, max_features = 100, stop_words = 'english', use_idf = True, tokenizer = tokenize_and_stem, ngram_range = (1,3)):
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        max_df = max_df,
        min_df = min_df,
        stop_words = stop_words,
        use_idf = use_idf,
        tokenizer = tokenizer,
        ngram_range = ngram_range
    )

    articles = [author.text for author in data]

    tfidf_matrix = tfidf_vectorizer.fit_transform(articles)
    terms = tfidf_vectorizer.get_feature_names()

    return tfidf_matrix, terms

def dist_cosine_similarity(tfidf_matrix):
    dist = 1 - cosine_similarity(tfidf_matrix)
    return dist

def _extract_vocab(data):
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for author in data:
        allwords_stemmed = tokenize_and_stem(author.text)  # for each item in 'synopses', tokenize/stem
        totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list

        allwords_tokenized = tokenize_only(author.text)
        totalvocab_tokenized.extend(allwords_tokenized)

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)

    return vocab_frame