from sentence_transformers import SentenceTransformer
from Bio import Entrez
from utils.search import check_pubmed, check_bio_med_rxiv, check_crossref
from crossref_commons.retrieval import get_publication_as_json
import logging
import argparse
import transformers
import warnings
import sys
from bs4 import BeautifulSoup

# ignore info, only show errors
warnings.simplefilter("ignore")
transformers.logging.set_verbosity_error()


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(level=logging.ERROR)
    parser = argparse.ArgumentParser(description='Program to find a corresponding journal DOI for a preprint')
    parser.add_argument('--preprint_doi', required=True, type=str, help='DOI of the preprint of interest')
    parser.add_argument('--server', default="other", type=str, help='name of the preprint server; can be either '
                                                                    'biorxiv, medrxiv or other')
    parser.add_argument('--email', required=False, default=False, type=str, help='Your email address to use the '
                                                                                 '*Entrez API*; if you send a lot of '
                                                                                 'queries, this is obligatory')
    args = parser.parse_args()
    preprint_doi = args.preprint_doi  # "e.g. 10.1101/2020.07.25.20161844"
    # preprint_doi = "10.21203/rs.3.rs-692244/v1"  # "10.2139/ssrn.3899014"
    server = args.server  # "ssrn"

    found = False
    if server.lower() == "medrxiv" or server.lower() == "biorxiv":
        journal_article = check_bio_med_rxiv(preprint_doi, server)
        if journal_article:
            found = True
            print(f"information found in {server}: https://doi.org/{journal_article}")
            sys.exit()
        else:
            print(f"No publication indexed in {server}.\nSearching in Crossref:")
    # if server is different or nothing was found:
    if (server != "biorxiv" and server != "medrxiv") or not found:
        crossref = check_crossref(preprint_doi)
        if crossref:
            print(f"information found in Crossref: {crossref}")
            sys.exit()
        # if crossref fails as well we use Pre2Pub:
        else:
            print(f"No publication indexed in Crossref.\nSearching with Pre2Pub (in PubMed):")
            # get preprint information
            preprint = get_publication_as_json(preprint_doi)
            title_preprint = preprint["title"][0]
            if "abstract" in preprint.keys():
                abstract_preprint = preprint["abstract"]
                # clean abstract and remove html syntax
                abstract_preprint = BeautifulSoup(abstract_preprint, features="html.parser").get_text()
                date_preprint = preprint["posted"]["date-parts"][0]
                date_preprint = "-".join([str(i) for i in date_preprint])
                authors_preprint = []
                for author in preprint["author"]:
                    if "given" in author.keys():  # otherwise these are usually consortia
                        authors_preprint.append(author["given"]+" "+author["family"])
                authors_preprint = "; ".join(authors_preprint)
                preprint_info = {"title": title_preprint, "abstract": abstract_preprint,
                                 "date": date_preprint, "authors": authors_preprint}
                # settings:
                sbert_model = SentenceTransformer("dmis-lab/biobert-base-cased-v1.1")
                if args.email:
                    Entrez.email = args.email
                pubmed = check_pubmed(preprint_doi, preprint_info, sbert_model)
                if pubmed:
                    print(f"Information found by Pre2Pub: {pubmed}")
                else:
                    print(f"No publication found by Pre2Pub.")
            else:
                print("No abstract information available - not able to run Pre2Pub")
