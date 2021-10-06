from typing import List
import Levenshtein as lev
from nltk.corpus import stopwords
import itertools
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class BioBertSimilarity:
    def _prepare_strings_for_similarity_comparison(str1, str2):
        """
        Prepare two strings for further similarity analysis by converting the
        strings to lowercase, removing dots at the end of the sentence.
        """
        str1 = str1.lower()
        str2 = str2.lower()
        if str1.endswith('.'):
            str1 = str1[:-1]
        if str2.endswith('.'):
            str2 = str2[:-1]
        return str1, str2

    def most_similar(doc_id, similarity_matrix, matrix):
        """
        Select most similar document based on the matrix.
        Matrix can be either "Cosine Similarity" or "Euclidean Distance".
        """
        if matrix == 'Cosine Similarity':
            similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]
        elif matrix == 'Euclidean Distance':
            similar_ix = np.argsort(similarity_matrix[doc_id])
        else:
            raise NotImplemented()
        for ix in similar_ix:
            if ix == doc_id:
                continue
        return similarity_matrix[doc_id][ix]

    def calculate_similarity(sbert_model, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using BioBERT cosine similarity.
        """
        str1, str2 = BioBertSimilarity._prepare_strings_for_similarity_comparison(str1, str2)
        documents = [str1, str2]
        documents_df = pd.DataFrame(documents, columns=['documents'])
        stop_words_l = stopwords.words('english')
        documents_df['documents_cleaned'] = documents_df.documents.apply(lambda x: " ".join(
            re.sub(r'[^a-zA-Z]', ' ', w).lower() for w in x.split() if
            re.sub(r'[^a-zA-Z]', ' ', w).lower() not in stop_words_l))
        document_embeddings = sbert_model.encode(documents_df['documents_cleaned'])
        pairwise_similarities = cosine_similarity(document_embeddings)
        return BioBertSimilarity.most_similar(0, pairwise_similarities, 'Cosine Similarity')


class AuthorSimilarity:
    """
    Class to check if the authors of the matched PubMed article are the same as the preprint's authors
    Authors can be stored in two formats in original json files. This is indicated via "AUTHOR_TYPE".
    "AUTHOR_TYPE" is passed in the file constants.py
    Here "ALL" or "BIORXIV" is used to specify that.
    "ALL" has the format: "Full first name Middle name initials Full last name", e.g. "Marie C Shmidtgall"
    "BIORXIV" has the format: "Full last name, First name initial." "Malmberg, H."
    """

    # In the new version, we use only crossref because all author names are completely given there!
    ALL = 'all'
    BIORXIV = 'biorxiv'
    AUTHOR_TYPE = 'all'  # 'biorxiv' # 'all'  # for crossref its is "all"

    @staticmethod
    def keep_first_and_last_author(authors: List[str]) -> List[str]:
        """
        Keep first and last authors in the list.
        corner case: if we have just one author in the list, we gonna return list with 2 copies of the same author.
        """
        return [authors[0], authors[-1]]

    @staticmethod
    def is_author_correct(authors_preprint: str, authors_pubmed: List[str]) -> bool:
        """
        Determines if authors of the preprint and PubMed article are the same.
        Return True or False
        """
        if not authors_pubmed:
            return False

        authors_preprint = authors_preprint.split('; ')

        # compare amount of authors: if it differs by more than three, return false
        if abs(len(authors_preprint) - len(authors_pubmed)) > 3:
            return False

        if any(AuthorSimilarity.perform_checks(authors_preprint, authors_pubmed)):
            # print(f'One of checks passed')
            return True

        if AuthorSimilarity._is_first_and_last_author_present_in_another_list(authors_preprint, authors_pubmed):
            # print(f'First and last author check passed')
            return True

        authors_preprint = AuthorSimilarity._move_first_author_to_last_position(authors_preprint)

        if any(AuthorSimilarity.perform_checks(authors_preprint, authors_pubmed)):
            # print(f'One of checks passed first author moved to last position')
            return True

        if AuthorSimilarity._is_first_and_last_author_present_in_another_list(authors_preprint, authors_pubmed):
            # print(f'First and last author check passed first author moved to last position')
            return True

        return False

    @staticmethod
    def perform_checks(authors_preprint: List[str], authors_pubmed: List[str]) -> List[bool]:
        """
        Gather checks in one function.
        """
        consensus = AuthorSimilarity.authors_consensus(authors_preprint, authors_pubmed)

        consensus_first_last_authors = AuthorSimilarity.consensus_first_and_last_authors(authors_preprint,
                                                                                         authors_pubmed)
        checkers = [
            AuthorSimilarity._is_consensus_majority(consensus),
            AuthorSimilarity._is_first_n_authors_the_same(consensus, n_authors=3),
            AuthorSimilarity._is_first_and_last_author_the_same(consensus),
            AuthorSimilarity._is_first_and_last_author_the_same(consensus_first_last_authors),
        ]
        return checkers

    @staticmethod
    def _is_consensus_majority(consensus: List[bool]) -> bool:
        """
        Determine if the consensus. If the number of "True" is strictly greater than the
        number of "False", then return True
        """
        return consensus.count(True) > consensus.count(False)

    @staticmethod
    def consensus_first_and_last_authors(authors_preprint: List[str], authors_pubmed: List[str]) -> List[bool]:
        """
        Consensus based on the first and the last authors.
        """
        authors_preprint_two = AuthorSimilarity.keep_first_and_last_author(authors_preprint)
        authors_pubmed_two = AuthorSimilarity.keep_first_and_last_author(authors_pubmed)
        consensus_first_last_authors = AuthorSimilarity.authors_consensus(authors_preprint_two, authors_pubmed_two)
        return consensus_first_last_authors

    @staticmethod
    def _move_first_author_to_last_position(authors_preprint: List[str]) -> List[str]:
        """
        Move the first author in the list to the last position.
        """
        authors_preprint.insert(len(authors_preprint), authors_preprint.pop(0))
        return authors_preprint

    @staticmethod
    def _is_first_and_last_author_the_same(consensus):
        """
        Return True if the first and the last authors are the same. Based on consensus.
        """
        return True is consensus[0] == consensus[-1]

    @staticmethod
    def _is_first_n_authors_the_same(consensus, n_authors):
        """
        Check if first n authors are the same. Based on consensus.
        """
        return all(consensus[:n_authors])

    @staticmethod
    def authors_consensus(authors_preprint: List[str], authors_pubmed: List[str]) -> List[bool]:
        """
        Build a consensus by comparing pairwise authors from preprint author's list and PubMed author's list.
        True is assigned if the author is the same, False otherwise.
        Returns a list of True and False.
        """
        consensus = []
        for apre, apub in zip(authors_preprint, authors_pubmed):
            is_the_same_author = AuthorSimilarity._is_author_the_same(apre, apub)
            consensus.append(is_the_same_author)
        return consensus

    @staticmethod
    def _is_first_and_last_author_present_in_another_list(authors_preprint: List[str],
                                                          authors_pubmed: List[str]) -> bool:
        """
        Check if the first and last authors of one list is present in the second list of authors.
        """
        first, last = authors_preprint[0], authors_preprint[-1]
        is_first_author_in_list = AuthorSimilarity._is_author_in_list(first, authors_pubmed)
        is_last_author_in_list = AuthorSimilarity._is_author_in_list(last, authors_pubmed)
        if all([is_first_author_in_list, is_last_author_in_list]):
            return True
        return False

    @staticmethod
    def _is_author_in_list(author_preprint: str, authors_pubmed: List[str]) -> bool:
        """
        check if a single author from preprint authors is present in the list of pubmed authors.
        """
        assert isinstance(author_preprint, str)
        for apre, apub in itertools.product([author_preprint], authors_pubmed):
            if AuthorSimilarity._is_author_the_same(apre, apub):
                return True
        return False

    @staticmethod
    def _is_author_the_same(author_preprint: str, author_pubmed: str) -> bool:
        """
        Function to determine if the given pair of authors is the same. Uses Levenshtein Distance and a set
        threshold.
        If the calculated Levenshtein ratio exceeds the given threshold, then the authors are considered to
        be the same.
        """
        author_threshold = 0.9
        apre, apre_initials_only = AuthorSimilarity.__prepare_authors_preprint(author_preprint)
        apub = AuthorSimilarity.__prepare_authors_pubmed(author_pubmed)
        reversed_apre = ' '.join(reversed(apre.split()))
        levenstein_ratio = AuthorSimilarity.__calculate_levenstein_ratio(apre, apub, reversed_apre)

        # print(f'Reversed preprint: {reversed_apre}')
        # print(f' Comparing preprint: {apre} ----- pubmed: {apub}')
        # print(f'lev: {lev.distance(apre, apub)}')
        # print(f'lev ratio: {levenstein_ratio}')

        if apre.startswith(apub):
            return True

        if apub.startswith(apre):
            return True

        if apre_initials_only.startswith(apub):
            return True

        if reversed_apre.startswith(apub):
            return True

        if apub.startswith(reversed_apre):
            return True

        return levenstein_ratio > author_threshold

    @staticmethod
    def __calculate_levenstein_ratio(apre, apub, reversed_apre):
        """
        Calculate Levenstein ratio between a and b.
        """
        levenstein_distance_ratio = lev.ratio(apre, apub)
        levenstein_distance_ratio_reversed = lev.ratio(reversed_apre, apub)
        levenstein_ratio = max(
            levenstein_distance_ratio,
            levenstein_distance_ratio_reversed
        )
        return levenstein_ratio

    @staticmethod
    def __prepare_authors_pubmed(author_pubmed: str):
        """
        Prepare PubMed authors based on author type.
        """
        if AuthorSimilarity.AUTHOR_TYPE == AuthorSimilarity.BIORXIV:
            return AuthorSimilarity.__prepare_authors_preprint_biorxiv(author_pubmed)
        return AuthorSimilarity.__prepare_authors_pubmed_all(author_pubmed)

    @staticmethod
    def __prepare_authors_preprint(author_preprint: str):
        """
        Prepare preprint authors based on author type.
        """
        if AuthorSimilarity.AUTHOR_TYPE == AuthorSimilarity.BIORXIV:
            return AuthorSimilarity.__prepare_authors_preprint_biorxiv(author_preprint)
        return AuthorSimilarity.__prepare_authors_preprint_all(author_preprint)

    @staticmethod
    def __prepare_authors_pubmed_biorxiv(author_pubmed: str):
        """
        Prepare authors list by converting the names to the format "Last name first name initials".
        """
        apub = AuthorSimilarity.__to_lower_and_split_by_comma(author_pubmed)
        surname_position = 0
        apre, apre_initials_only = AuthorSimilarity._rearrange_surname_and_initials(apub, surname_position)
        return apre_initials_only

    @staticmethod
    def __prepare_authors_pubmed_all(author_pubmed: str):
        return AuthorSimilarity.__to_lower_and_split_by_comma(author_pubmed)

    @staticmethod
    def __to_lower_and_split_by_comma(author_pubmed):
        """
        Set string to lowercase and split by comma.
        """
        apub = author_pubmed
        apub = apub.lower()
        if apub.count(',') >= 1:
            apub = apub.split(', ')
            apub = ' '.join(apub)
        return apub

    @staticmethod
    def __prepare_authors_preprint_biorxiv(author_preprint: str):
        """
        Preprocess preprint authors - convert to lowercase, remove dots
        """

        apre = author_preprint
        apre = apre.lower()
        apre = apre.replace(', ', ' ')
        apre = apre.replace('.', '')
        surname_position = 0
        apre, apre_initials_only = AuthorSimilarity._rearrange_surname_and_initials(apre, surname_position)
        apre = ' '.join(apre)

        return apre, apre_initials_only

    @staticmethod
    def _rearrange_surname_and_initials(authors: str, surname_position: int):
        """
        Prepare authors list by converting the names to the format "Last name first name initials".
        """
        authors = authors.split(' ')
        surname = authors[surname_position]
        del authors[surname_position]
        apre_initials_only = [author[0] for author in authors if author]
        apre_initials_only.insert(0, surname)
        apre_initials_only = ' '.join(apre_initials_only)
        authors.insert(0, surname)
        return authors, apre_initials_only

    @staticmethod
    def __prepare_authors_preprint_all(author_preprint: str):
        """
        Preprocess preprint authors - convert to lowercase, remove dots
        """

        apre = author_preprint
        apre = apre.lower()
        apre = apre.replace('. ', ' ')
        apre = apre.replace('.', ' ')
        surname_position = -1
        apre, apre_initials_only = AuthorSimilarity._rearrange_surname_and_initials(apre, surname_position)
        apre = ' '.join(apre)
        return apre, apre_initials_only
