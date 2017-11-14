from import_data import db_connect, ncbi_request

db_session, engine = db_connect(mode = "dev", reset = False)

db_session, engine = db_connect(mode = "dev", reset = True)
ncbi_request(db_session, engine, term = "HIV[Title]", category = "hiv", maxArticles = 100, method = "fullName", maxAuthors = 1)
ncbi_request(db_session, engine, term = "malaria[Title]", category = "malaria", maxArticles = 100, method = "fullName", maxAuthors = 1)
ncbi_request(db_session, engine, term = "influenza[Title]", category = "influenza", maxArticles = 100, method = "fullName", maxAuthors = 1)
ncbi_request(db_session, engine, term = "hepatitis[Title]", category = "hepatitis", maxArticles = 100, method = "fullName", maxAuthors = 1)
ncbi_request(db_session, engine, term = "zika[Title]", category = "zika", maxArticles = 100, method = "fullName", maxAuthors = 1)






from nlp import get_data, tokenize_and_stem, tokenize_only, tfidf_vectorize, dist_cosine_similarity
from cluster import cluster_kmeans

data = get_data(db_session, type = "abstract")
tfidf_matrix, terms = tfidf_vectorize(data, max_features=1000)
dist = dist_cosine_similarity(tfidf_matrix)

real_clusters = [author.category for author in data]

cluster_kmeans(data, terms, tfidf_matrix, dist, real_clusters = real_clusters, nb_clusters = 5)
