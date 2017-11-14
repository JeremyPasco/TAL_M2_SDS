from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine, func
import os
from models import Author, Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#Connection to sqlite
#Under Windows
if os.name == 'nt':
    sqlitePath = 'sqlite:///' + os.path.dirname(__file__) + '\\sqlite.db'

#Under UNIX
else:
    sqlitePath = 'sqlite:///sqlite.db'

engine = create_engine(sqlitePath)
print("Connected to database (" + sqlitePath + ")")

Session = scoped_session(sessionmaker(bind=engine))
db_session = Session()




#Import titles for each author
data = db_session.query(Author.id, Author.firstName, Author.lastName,
                        func.group_concat(Article.title).label('text')).group_by(Author.id)




#176 authors
# index 0 to 83 : HIV authors
# index 84 to 175 : Malaria authors
titles = [author.text for author in data]





#Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=10,
    max_df = 1.0,
    min_df = 1,
    stop_words = "english",
    use_idf = True
)
tfidf_matrix = tfidf_vectorizer.fit_transform(titles)

#10 best TFIDF terms (including HIV and malaria)
terms = tfidf_vectorizer.get_feature_names()
print(terms)

#No difference between texts
print(tfidf_matrix[0]) #This is a HIV author
print(tfidf_matrix[110]) #This is a malaria author

#Implying a similarity of 1 in every cases
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
print(similarity)
