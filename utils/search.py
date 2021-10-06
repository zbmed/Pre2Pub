from Bio import Entrez, Medline
from utils.similarity import AuthorSimilarity, BioBertSimilarity
from datetime import date
from crossref_commons.retrieval import get_publication_as_json
from nltk.corpus import stopwords
import requests


def esearch_pmids(title, retmax=5):
    """
    This func requests the following URL
    https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed
    &term=Adverse+Drug+Reactions+in+COVID-19+Patients%3A+Protocol+for+a+Living+Systematic+Review
    &retmax=5
    &tool=biopython
    &email=Your.Name.Here%40example.org

    This can be verified by printing handle.url
    """
    handle = Entrez.esearch(
        db="pubmed",
        term=title + " AND Journal Article[filter]",  # ensures that preprints are not found
        retmax=retmax
    )
    record = Entrez.read(handle)
    pmid_list = record["IdList"]
    handle.close()
    return pmid_list


def _entrez_fetch(id_):
    """
    Entrez function eFetch to retrieve handle and records by id.
    """
    handle = Entrez.efetch(db="pubmed", id=id_, rettype="medline", retmode="text")
    records = Medline.parse(handle)
    return handle, records


def convert_date(date_):
    if "/" in date_:
        date_converted = date_[:10].split("/")
    elif "-" in date_:
        date_converted = date_[:10].split("-")
    date_converted = date(int(date_converted[0]), int(date_converted[1]), int(date_converted[2]))
    return date_converted


def check_crossref(preprint_doi):
    """
    method that checks whether a preprint has a corresponding journal article via CrossRef
    """
    # check on CrossRef
    try:
        crossref = get_publication_as_json(preprint_doi)
        if "relation" in crossref.keys():
            keys = crossref["relation"].keys()
            if "is-preprint-of" in keys:  # if it has a publication
                # [{'id-type': 'doi', 'id': '10.3390/jcm9020538', 'asserted-by': 'subject'}]
                crossref_ids = crossref["relation"]["is-preprint-of"]
                for entry in crossref_ids:
                    if entry["id-type"] == "doi":
                        published_id = entry["id"]
                        break
                    else:
                        published_id = False  # in case there is a publication available but no DOI exist
                        # I don't know whether this edge case exists
            else:
                return False
    except:  # Connection Error
        return False

    return published_id


def check_pubmed(preprint_doi, preprint, sbert_model):
    """
    algorithm that searches for a peer-reviewed article in pubmed
    input: preprint_doi (str) and a dict for further preprint information (abstract, title, author)
    returns the DOI if found, False otherwise
    """

    # search for title in PubMed
    title_preprint = preprint["title"]
    # clean title/remove stop words
    stop_words_l = stopwords.words('english')
    title_preprint = " ".join([w for w in title_preprint.split() if not w in stop_words_l])
    try:
        pubmed = esearch_pmids(title_preprint)
    except:  # ConnectionError
        return "API connection error. Please try again later."

    author_search = False

    # if search returns no results (titles do not match), search for authors
    if not pubmed:  # search for authors:
        author_search = True
        authors_preprint = preprint["authors"]
        authors_preprint_search = authors_preprint.replace(";", " AND").replace(",", "").replace(".", "") + "[author]"
        try:
            pubmed = esearch_pmids(authors_preprint_search)
        except:
            return "API connection error. Please try again later."

    if pubmed:  # if also this list is empty, we have no match
        # now have a list of pmids and make a deep author check
        try:
            _, records = _entrez_fetch(pubmed)
        except:
            return False
        records = [{"pmid": pmid, "article": article,
                    "author_check": float(), "title_check": float(), "abstract_similarity": float()}
                   for pmid, article in zip(pubmed, records)]  # turn generator into list for multi-use

        # check that date of publication is newer than preprint
        date_preprint = convert_date(preprint["date"])
        try:
            records = [r for r in records if date_preprint < convert_date(r["article"]["EDAT"])]
        except KeyError:
            try:
                records = [r for r in records if date_preprint < convert_date(r["article"]["PD"])]
            except:
                records = records  # if date is not given, we retain all

        # if search was performed via author, compare titles, else author check
        if author_search:
            # title check:
            for i, entry in enumerate(records):
                if "TI" in entry["article"].keys():
                    title_pubmed = entry["article"]["TI"]
                    title_pubmed = " ".join([w for w in title_pubmed.split() if not w in stop_words_l])
                    compare = BioBertSimilarity.calculate_similarity(sbert_model, title_preprint, title_pubmed)
                    records[i]["title_check"] = compare  # round(compare,2)
        else:
            # author check:
            for i, article in enumerate(records):
                authors_preprint = preprint["authors"]
                if "AU" in article["article"].keys():
                    authors_pubmed = article["article"]["AU"]
                    # print(authors_pubmed, authors_preprint)
                    compare = AuthorSimilarity.is_author_correct(authors_preprint, authors_pubmed)
                    records[i]["author_check"] = compare
        for i, entry in enumerate(records):
            if entry["author_check"] or entry["title_check"] > 0.94:
                preprint_abstract = preprint["abstract"]
                if "AB" in entry["article"].keys():
                    pubmed_abstract = entry["article"]["AB"]
                    compare = BioBertSimilarity.calculate_similarity(sbert_model, preprint_abstract, pubmed_abstract)
                    records[i]["abstract_similarity"] = compare
        if records:
            found_journal = max(records, key=lambda m: m["abstract_similarity"])
            if found_journal["abstract_similarity"] >= 0.9:
                if "AID" in found_journal["article"].keys():
                    found_via_algorithm = found_journal["article"]["AID"]  # is a list of different identifiers
                    # extract doi
                    for id_ in found_via_algorithm:
                        if "doi" in id_:
                            doi = id_.replace(" [doi]", "")
                            published_info = "https://doi.org/" + doi
                            break
                        else:
                            published_info = found_via_algorithm
                # if doi is not available, return PMID
                else:
                    published_info = "https://pubmed.ncbi.nlm.nih.gov/" + found_journal["article"]["PMID"]
    if 'published_info' in locals():
        return published_info
    else:
        return False


def check_bio_med_rxiv(doi, server):
    url = "https://api.biorxiv.org/details/" + server.lower() + "/" + doi
    try:
        response = requests.get(url).json()
    except ConnectionError:
        return False
    try:
        published_info = response["collection"][0]["published"]
        found_in_server = True
    except IndexError:
        return False

    if found_in_server and published_info != "NA":
        return published_info
    else:
        return False
