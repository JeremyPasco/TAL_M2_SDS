from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine, func
from models import Base, Author, Article
from Bio import Entrez
import ssl
import os

def db_connect(mode = "prod", reset = False):
    if mode == "dev":
        sqlitePath = 'sqlite:///D:\\Documents\\python\\test_flask\\sqlite.db'
        #engine.echo = True
    else:
        if os.name == 'nt':
            sqlitePath = 'sqlite:///' + os.path.dirname(__file__) + '\\sqlite.db'

        else:
            sqlitePath = 'sqlite:///sqlite.db'

    engine = create_engine(sqlitePath)

    if reset:
        print("Resetting database")
        _db_drop(engine)
        _db_create(engine)

    print("Connected to database (" + sqlitePath + ")")

    Session = scoped_session(sessionmaker(bind=engine))
    db_session = Session()
    return db_session, engine


def _db_drop(engine):
    Base.metadata.drop_all(engine)

def _db_create(engine):
    Base.metadata.create_all(engine)

def _ncbi_request_ids(term = "infectious", maxArticles = 500):
    #Prepare NCBI connexion
    ssl._create_default_https_context = ssl._create_unverified_context
    Entrez.email = "Your.Name.Here@example.org"

    #Get ids from a request
    handle = Entrez.esearch(db="pubmed", retmax=maxArticles, term=term)
    records = Entrez.read(handle)
    handle.close()

    return records['IdList']

def _ncbi_abstracts_from_ids(list_ids):
    #Get abstracts from ids
    str_list_ids = ','.join(list_ids)
    handle = Entrez.efetch(db="pubmed", id=str_list_ids, retmode='xml')
    records = Entrez.read(handle)["PubmedArticle"]
    handle.close()

    return records

def _db_save_abstract(db_session, records, category, maxAuthors):
    nb_abstracts = 0
    nb_fails = 0
    nb_assocs = 0
    for record in records:
        try:
            authors = record['MedlineCitation']['Article']['AuthorList']

            a = Article(
                title=record['MedlineCitation']['Article']['ArticleTitle'],
                category=category,
                abstract='. '.join(record['MedlineCitation']['Article']['Abstract']['AbstractText'])
            )


            author_index = 0
            for author in authors:
                if maxAuthors == 0 or author_index < maxAuthors: 
                    a.authors.append(Author(
                        firstName=author['ForeName'],
                        lastName=author['LastName'],
                        category=category
                    ))
                    nb_assocs += 1
                    author_index += 1

            db_session.add(a)
            db_session.commit()
        except:
            nb_fails += 1
        else:
            nb_abstracts += 1

    return nb_abstracts, nb_assocs, nb_fails


def _db_merge_authors(db_session, method = "fullName"): #Or lastName
    authors = []
    authorNames = []
    toMerge = []

    for instance in db_session.query(Author.id, Author.fullName, Author.lastName):
        if getattr(instance, method) in authorNames:

            newId = (item for item in authors if item["nameToCompare"] == getattr(instance, method)).__next__()['id']

            toMerge.append({
                "originId": instance.id,
                "newId": newId
            })
        else:
            authors.append({"id": instance.id, "nameToCompare": getattr(instance, method)})
            authorNames.append(getattr(instance, method))

    for operation in toMerge:
        for instance in db_session.query(Article).filter(Article.authors.any(id=operation['originId'])):
            print(str(instance.id) + ': ' + str(operation['originId']) + ' -> ' + str(operation['newId']))
            originAuthor = db_session.query(Author).get(operation['originId'])
            newAuthor = db_session.query(Author).get(operation['newId'])
            instance.authors.remove(originAuthor)
            instance.authors.append(newAuthor)
            db_session.commit()

def _db_remove_orphan_authors(db_session, engine):
    #Remove authors with no article
    stmt = Author.__table__.delete().where(Author.articles == None)
    engine.execute(stmt)

def ncbi_request(db_session, engine, term = "infectious", maxArticles = 500, method = "fullName", category = "1", maxAuthors = 1):

    print()
    print('Requesting articles for "' + term + '" on Pubmed...')
    list_ids = _ncbi_request_ids(term, maxArticles)
    nb_articles = str(len(list_ids))
    print(nb_articles + ' articles found')
    print('Downloading abstracts...')
    records = _ncbi_abstracts_from_ids(list_ids)

    print()
    print('Persisting abstracts in database...')
    nb_abstracts, nb_assocs, nb_fails = _db_save_abstract(db_session, records, category, maxAuthors)
    print(str(nb_abstracts) + ' abstracts found from ' + nb_articles + ' articles')
    print(str(nb_assocs) + ' author-abstract associations created')

    print()
    print('Merging authors with same ' + method + '...')
    _db_merge_authors(db_session, method)
    print('Cleaning database...')
    _db_remove_orphan_authors(db_session, engine)
    authors = db_session.query(Author).all()
    print('Done: ' + str(len(authors)) + ' unique authors for ' + str(nb_abstracts) + ' abstracts')

    #db_session.query(Author).filter(Author.articles == None).all()
    #db_session.query(Author.fullName, func.count(Author.id).label('count')).join(Author.articles).group_by(
    #    Author.id).order_by('count DESC').all()