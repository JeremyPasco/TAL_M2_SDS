from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pandas as pd
from nlp import _extract_vocab


def _get_colors(n):
    _colors = [
        "#4286f4",
        "#f4415f",
        "#41f4a0",
        "#dbb313",
        "#bc13db"
    ]

    colors = {}
    for i in range(n):
        colors[i] = _colors[i]

    return colors


    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    tuple_colors = [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

    colors = {}
    i = 0
    for c in tuple_colors:
        colors[i] = '#%02x%02x%02x' % c
        i += 1

    return colors

def cluster_kmeans(data, terms, tfidf_matrix, dist, real_clusters, nb_clusters = 5, verbose = 0):
    km = KMeans(
        n_clusters = nb_clusters,
        verbose = verbose,

    )
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()

    #frame = pd.DataFrame(data, index=[clusters], columns=['id', 'firstName', 'lastNames', 'text'])
    #print(frame['cluster'].value_counts())

    vocab_frame = _extract_vocab(data)
    print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


    print("Top terms per cluster:")
    print()
    #sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    cluster_names = {}
    for i in range(nb_clusters):
        print("Cluster %d words:" % i, end='')

        temp = ""
        for ind in order_centroids[i, :6]:  # replace 6 with n words per cluster
            temp += vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0] + ', '
            print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'),
                  end=',')

        cluster_names[i] = temp
        print()  # add whitespace
        print()  # add whitespace

        #print("Cluster %d titles:" % i, end='')
        #for title in frame.ix[i]['title'].values.tolist():
            #print(' %s,' % title, end='')
        #print()  # add whitespace
        #print()  # add whitespace


    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

    xs, ys = pos[:, 0], pos[:, 1]

    cluster_colors = _get_colors(nb_clusters)

    # create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=real_clusters))

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
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(
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
    #plt.close()
    # uncomment the below to save the plot if need be
    # plt.savefig('clusters_small_noaxes.png', dpi=200)