from Bio import Entrez
import ssl
from nltk.tokenize import word_tokenize
#nltk.download('punkt')


ssl._create_default_https_context = ssl._create_unverified_context

Entrez.email = "Your.Name.Here@example.org"

handle = Entrez.esearch(db="pubmed", retmax=100, term="infectious")
records = Entrez.read(handle)
handle.close()

list_ids = ','.join(records['IdList'])
print(list_ids)

handle = Entrez.efetch(db="pubmed", id=list_ids, retmode='xml')
records = Entrez.read(handle)["PubmedArticle"]
handle.close()

i = 0
results = []

for record in records:
    print(i)
    try:
        authors = record['MedlineCitation']['Article']['AuthorList']
        for author in authors:
            results.append({
                'lastName': author['LastName'],
                'firstName': author['ForeName'],
                'abstract': '. '.join(record['MedlineCitation']['Article']['Abstract']['AbstractText'])
            })
    except:
        print('failure...')
    else:
        print('ok')
    i+=1

#records[5]['MedlineCitation']['Article']['Abstract']['AbstractText']
#records[5]['MedlineCitation']['Article']





#Pour controler le resultat:
#co authoring
#citation
#keywords


test = results[0]['abstract']
test.lower()
word_tokenize(test)