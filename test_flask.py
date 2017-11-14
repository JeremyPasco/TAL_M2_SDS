from flask import Flask
from sqlalchemy import func



import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import ward, dendrogram
import re
import os
import matplotlib.pyplot as plt
#import matplotlib as mpl
#import codecs
#from sklearn import feature_extraction
#import mpld3
#nltk.download('punkt')
#nltk.download('stopwords')

# app = Flask(__name__)
# app.debug = True







from import_data import db_connect, ncbi_request

#db_session, engine = db_connect(mode = "dev", reset = True)
#ncbi_request(db_session, engine, term = "pediatric", maxArticles = 100, method = "fullName")


db_session, engine = db_connect(mode = "dev", reset = False)


from nlp import get_data, tokenize_and_stem, tokenize_only, tfidf_vectorize, dist_cosine_similarity
from cluster import cluster_kmeans

data = get_data(db_session)
tfidf_matrix, terms = tfidf_vectorize(data, max_features=100)
dist = dist_cosine_similarity(tfidf_matrix)

cluster_kmeans(data, terms, tfidf_matrix, dist, nb_clusters = 5)











for article in dbsession.query(Article):
    text = article.abstract
    text.lower()
    tokens = word_tokenize(text)

totalvocab_stemmed = []
totalvocab_tokenized = []


data = dbsession.query(Author.id, Author.firstName, Author.lastName, func.group_concat(Article.abstract).label('text')).group_by(Author.id).limit(100)
for author in data:
    text = author.text

    allwords_stemmed = tokenize_and_stem(text)  # for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list

    allwords_tokenized = tokenize_only(text)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

















import networkx as nx
M = []
for author1 in data:
    l = []
    for author2 in data:
        if author1.id == author2.id:
            l.append(1)
        else:
            nmi = normalized_mutual_info_score(labels_true, labels_pred)

    M.append(l)




G = nx.from_numpy_matrix(np.matrix(M))
M = np.array(M)


from mcl_clustering import get_graph, networkx_mcl, draw
#M, G = get_graph("example.csv")

print(" number of nodes: %s\n" % M.shape[0])

M, clusters = networkx_mcl(G)
draw(G, M, clusters)




















print(tfidf_matrix.shape)





num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]









# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

#set up cluster names using a dict
cluster_names = {0: 'Family, home, war',
                 1: 'Police, killed, murders',
                 2: 'Father, New York, brothers',
                 3: 'Dance, singing, love',
                 4: 'Killed, soldiers, captain'}

# create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=range(0,42)))

# group by cluster
groups = df.groupby('label')

# set up plot
fig, ax = plt.subplots(figsize=(17, 9))  # set size
ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

# iterate through groups to layer the plot
# note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params( \
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params( \
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelleft='off')

ax.legend(numpoints=1)  # show legend with only 1 point

# add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

plt.show()  # show the plot
plt.close()
# uncomment the below to save the plot if need be
# plt.savefig('clusters_small_noaxes.png', dpi=200)






linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=range(0,88));

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout
plt.show()


#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters

# # Insert one row via add(instance) and commit
# dbsession.add(Cafe('coffee', 'Espresso', 3.19))  # Construct a Cafe object
# # INSERT INTO cafe (category, name, price) VALUES (%s, %s, %s)
# # ('coffee', 'Espresso', 3.19)
# dbsession.commit()
#
# # Insert multiple rows via add_all(list_of_instances) and commit
# dbsession.add_all([Cafe('coffee', 'Cappuccino', 3.29),
#                    Cafe('tea', 'Green Tea', 2.99, id=8)])  # using kwarg for id
# dbsession.commit()
#
# # Select all rows. Return a list of Cafe instances
# for instance in dbsession.query(Cafe).all():
#     print(instance.category, instance.name, instance.price)
# # SELECT cafe.id AS cafe_id, cafe.category AS cafe_category,
# #   cafe.name AS cafe_name, cafe.price AS cafe_price FROM cafe
# # coffee Espresso 3.19
# # coffee Cappuccino 3.29
# # tea Green Tea 2.99
#
# # Select the first row with order_by. Return one instance of Cafe
# instance = dbsession.query(Cafe).order_by(Cafe.name).first()
# print(instance)   # Invoke __repr__()
# # SELECT cafe.id AS cafe_id, cafe.category AS cafe_category,
# #   cafe.name AS cafe_name, cafe.price AS cafe_price
# # FROM cafe ORDER BY cafe.name LIMIT %s
# # (1,)
# # Cafe(2, coffee, Cappuccino,  3.29)
#
# # Using filter_by on column
# for instance in dbsession.query(Cafe).filter_by(category='coffee').all():
#     print(instance.__dict__)   # Print object as key-value pairs
# # SELECT cafe.id AS cafe_id, cafe.category AS cafe_category,
# #   cafe.name AS cafe_name, cafe.price AS cafe_price
# # FROM cafe WHERE cafe.category = %s
# # ('coffee',)
#
# # Using filter with criterion
# for instance in dbsession.query(Cafe).filter(Cafe.price < 3).all():
#     print(instance)
# # SELECT cafe.id AS cafe_id, cafe.category AS cafe_category,
# #   cafe.name AS cafe_name, cafe.price AS cafe_price
# # FROM cafe WHERE cafe.price < %s
# # (3,)
# # Cafe(8, tea, Green Tea,  2.99)
#
# # Delete rows
# instances_to_delete = dbsession.query(Cafe).filter_by(name='Cappuccino').all()
# # SELECT cafe.id AS cafe_id, cafe.category AS cafe_category,
# #   cafe.name AS cafe_name, cafe.price AS cafe_price
# # FROM cafe WHERE cafe.name = %s
# # ('Cappuccino',)
# for instance in instances_to_delete:
#     dbsession.delete(instance)
# dbsession.commit()
# # DELETE FROM cafe WHERE cafe.id = %s
# # (2,)
#
# for instance in dbsession.query(Cafe).all():
#     print(instance)
# # SELECT cafe.id AS cafe_id, cafe.category AS cafe_category,
# #   cafe.name AS cafe_name, cafe.price AS cafe_price
# # FROM cafe
# # Cafe(1, coffee, Espresso,  3.19)
# # Cafe(8, tea, Green Tea,  2.99)




# @app.route('/')
# def hello_world():
#     return 'Hello World!'
#
#
# if __name__ == '__main__':
#     app.run()




#Pour controler le resultat:
#co authoring
#citation
#keywords