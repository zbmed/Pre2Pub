## Pre2Pub: An algorithm for tracking the path from preprint to journal in times of an infodemic

This repo contains an algorithm that, given the Digital Object Identifier (DOI) of a preprint,
searches the corresponding journal article in three different sources: 
1) For a bio- or medRxiv preprint, the bioRxiv API is queried.  
2) If no link is found in 1. or the preprint comes from a different server, Crossref API is queried.
3) If no link is found in 2. PubMed is queried using our developed algorithm.  
  
The working mechanism of the algorithm is the following:  
1) For all preprints, we use the cleaned title (without stopwords) as search term to retrieve a list of PubMed IDs (PMIDs) via the *title* field. Retrieval maximum is set to five in order to avoid false positive results.
2) If the search via the preprint title was successful, the authors of the preprint and the authors of the matched PubMed journal articles are compared. The preprint servers provide the author names in different formats. For example, some servers provide full names with all first names, others provide only last names and first names' initials. For further authors' comparison, we defined the following common name storage schema: for each author, we store his/her last name and first names' initials. To compare the authors, we use Levenshtein ratio [1] and determine the matches using several criteria iteratively developed with the help of the training data. First, we run through the two lists of authors simultaneously and determine the Levenshtein ratio for each pair. If the value is greater than 0.9, we assume that the authors are identical.
If this is not the case, we distinguish the following four cases to consider the authors of the preprints and the authors of the journal article to be the same: first, if we find more matched than unmatched author pairs when iterating over the author lists (i.e. based on consensus); second, if the first three authors of both lists are the same; third, if the first and the last author of preprint and journal article are identical; fourth, if the first and the last author of the preprint are found in the author list of the journal article (regardless of position).
3) If the title search was not successful, we search for the list of authors in the *author* field. Retrieval maximum is set to five in both cases in order to avoid false positive results. If successful, we compare the titles of the preprint and the fetched journal articles. To do this, we generate embeddings by making use of Sentence-BERT, "a modification of the pretrained BERT network that use siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity" [2]. As pre-trained model, we load BioBERT (*dmis-lab/biobert-base-cased-v1.1*). The threshold is set to 0.95.
4) We compare the dates of publication to ensure that the date of the preprint is older than the date of the peer-reviewed article.
5) We compare the abstract of the preprint to the abstracts of the PubMed articles that passed the date check and either the author or the title check - depending on which search was successful in the beginning. We generate embeddings and apply the same method as for the title check described in (3).

**References:**  
[1] Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions and reversals. Soviet Physics Doklady, 10(8), 707--710.  
[2] Reimers N, Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. arXiv:190810084 [cs]. Published online August 27, 2019. Accessed October 4, 2021. http://arxiv.org/abs/1908.10084

## How to use

Clone the repository to your local folder: `git clone https://github.com/zbmed/Pre2Pub`  
and then change the directory: `cd Pre2Pub`  
  
Create a new virtual environment and activate it (for Linux it will be):  
`python3 -m venv venv`  
`source venv/bin/activate`  
  
Install all needed packages:  
`pip3 install -r requirements.txt`  
  
Run program:  
`usage: main.py [-h] --preprint_doi PREPRINT_DOI [--server SERVER] [--email EMAIL]`  
`Program to find a corresponding journal DOI for a preprint`  

`optional arguments:`  
`-h, --help            show this help message and exit`   
`--preprint_doi PREPRINT_DOI DOI of the preprint of interest`   
`--server SERVER       name of the preprint server; can be either biorxiv,
medrxiv or other`  
`--email EMAIL         Your email address to use the *Entrez API*; if you
send a lot of queries, this is obligatory`  
  
Example (replace the e-mail address!):  
python main.py --preprint_doi 10.1101/2020.07.06.20147199 --server medrxiv --email mail@mail.com


## Citation:
Coming soon 



