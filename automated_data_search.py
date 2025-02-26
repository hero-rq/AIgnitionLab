# Filename: automated_data_and_paper_search.py

from utils import *

import time
import arxiv
import os
import re
import io
import sys
import numpy as np
import concurrent.futures
from pypdf import PdfReader
from datasets import load_dataset
from psutil._common import bytes2human
from datasets import load_dataset_builder
from semanticscholar import SemanticScholar
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

import traceback
import concurrent.futures


#########################################################
## HuggingFace Dataset Search
#########################################################
class HFDataSearch:
    """
    Searches the Hugging Face dataset index for popular 
    datasets (filtering by likes and downloads) and 
    supports text-based queries using TF-IDF.
    """

    def __init__(self, like_thr=3, dwn_thr=50) -> None:
        """
        Class for finding relevant HuggingFace datasets.

        :param like_thr: Minimum number of 'likes' threshold.
        :param dwn_thr: Minimum number of 'downloads' threshold.
        """
        self.dwn_thr = dwn_thr
        self.like_thr = like_thr
        self.ds = load_dataset("nkasmanoff/huggingface-datasets")["train"]

        # Initialize lists to collect filtered data
        filtered_indices = []
        filtered_descriptions = []
        filtered_likes = []
        filtered_downloads = []

        # Iterate over the dataset and filter based on criteria
        for idx, item in enumerate(self.ds):
            # Get likes and downloads, handling None values
            likes = int(item['likes']) if item['likes'] is not None else 0
            downloads = int(item['downloads']) if item['downloads'] is not None else 0

            # Check if likes and downloads meet thresholds
            if likes >= self.like_thr and downloads >= self.dwn_thr:
                description = item['description']
                if isinstance(description, str) and description.strip():
                    # Collect the data
                    filtered_indices.append(idx)
                    filtered_descriptions.append(description)
                    filtered_likes.append(likes)
                    filtered_downloads.append(downloads)

        # Check if any datasets meet all criteria
        if not filtered_indices:
            print("No datasets meet the specified criteria.")
            self.ds = []
            self.descriptions = []
            self.likes_norm = []
            self.downloads_norm = []
            self.description_vectors = None
            return  # Exit constructor

        # Filter the datasets by collected indices
        self.ds = self.ds.select(filtered_indices)

        # Update local references
        self.descriptions = filtered_descriptions
        self.likes = np.array(filtered_likes)
        self.downloads = np.array(filtered_downloads)

        # Normalize likes and downloads
        self.likes_norm = self._normalize(self.likes)
        self.downloads_norm = self._normalize(self.downloads)

        # Vectorize the descriptions (TF-IDF)
        self.vectorizer = TfidfVectorizer()
        self.description_vectors = self.vectorizer.fit_transform(self.descriptions)

    def _normalize(self, arr):
        """
        Normalizes the given NumPy array to the range [0, 1].
        """
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val == 0:
            return np.zeros_like(arr, dtype=float)
        return (arr - min_val) / (max_val - min_val)

    def retrieve_ds(self, query, N=10, sim_w=1.0, like_w=0.0, dwn_w=0.0):
        """
        Retrieves the top N datasets matching the query, 
        weighted by user-defined importance factors.

        :param query: Search query string.
        :param N: Number of results to return.
        :param sim_w: Weight for cosine similarity (description-based).
        :param like_w: Weight for like counts.
        :param dwn_w: Weight for download counts.
        :return: List of top N dataset items (dictionaries).
        """
        if not self.ds or self.description_vectors is None:
            print("No datasets available to search.")
            return []

        # Vectorize query and compute similarity
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = linear_kernel(query_vector, self.description_vectors).flatten()

        # Normalize cosine similarities
        cosine_similarities_norm = self._normalize(cosine_similarities)

        # Compute final scores
        final_scores = (
            sim_w * cosine_similarities_norm +
            like_w * self.likes_norm +
            dwn_w * self.downloads_norm
        )

        # Get top N indices
        top_indices = final_scores.argsort()[-N:][::-1]
        top_indices = [int(i) for i in top_indices]
        top_datasets = [self.ds[i] for i in top_indices]

        # Check existence of train/test splits
        has_test_set = []
        has_train_set = []
        ds_size_info = []
        for i in top_indices:
            try:
                dbuilder = load_dataset_builder(
                    self.ds[i]["id"], trust_remote_code=True
                ).info
            except Exception:
                has_test_set.append(False)
                has_train_set.append(False)
                ds_size_info.append((None, None, None, None))
                continue

            if dbuilder.splits is None:
                has_test_set.append(False)
                has_train_set.append(False)
                ds_size_info.append((None, None, None, None))
                continue

            has_test = "test" in dbuilder.splits
            has_train = "train" in dbuilder.splits
            has_test_set.append(has_test)
            has_train_set.append(has_train)

            test_dwn_size, test_elem_size = None, None
            train_dwn_size, train_elem_size = None, None

            if has_test:
                test_dwn_size = bytes2human(dbuilder.splits["test"].num_bytes)
                test_elem_size = dbuilder.splits["test"].num_examples
            if has_train:
                train_dwn_size = bytes2human(dbuilder.splits["train"].num_bytes)
                train_elem_size = dbuilder.splits["train"].num_examples

            ds_size_info.append((test_dwn_size, test_elem_size, train_dwn_size, train_elem_size))

        # Annotate result dictionaries
        for idx, dataset_obj in enumerate(top_datasets):
            dataset_obj["has_test_set"] = has_test_set[idx]
            dataset_obj["has_train_set"] = has_train_set[idx]
            dataset_obj["test_download_size"] = ds_size_info[idx][0]
            dataset_obj["test_element_size"] = ds_size_info[idx][1]
            dataset_obj["train_download_size"] = ds_size_info[idx][2]
            dataset_obj["train_element_size"] = ds_size_info[idx][3]

        return top_datasets

    def results_str(self, results):
        """
        Provide results as a list of strings in a human-readable format.

        :param results: List of dataset dictionaries.
        :return: List of formatted strings describing each dataset.
        """
        result_strs = []
        for result in results:
            res_str = f"Dataset ID: {result['id']}\n"
            res_str += f"Description: {result['description']}\n"
            res_str += f"Likes: {result['likes']}\n"
            res_str += f"Downloads: {result['downloads']}\n"
            res_str += f"Has Testing Set: {result['has_test_set']}\n"
            res_str += f"Has Training Set: {result['has_train_set']}\n"
            res_str += f"Test Download Size: {result['test_download_size']}\n"
            res_str += f"Test Dataset Size: {result['test_element_size']}\n"
            res_str += f"Train Download Size: {result['train_download_size']}\n"
            res_str += f"Train Dataset Size: {result['train_element_size']}\n"
            result_strs.append(res_str)
        return result_strs


#########################################################
## SemanticScholar Paper Search
#########################################################
class SemanticScholarSearch:
    """
    Leverages the SemanticScholar Python wrapper to search 
    for relevant papers and retrieve metadata about them. 
    (Full-text retrieval not yet implemented.)
    """

    def __init__(self):
        self.sch_engine = SemanticScholar(retry=False)

    def find_papers_by_str(self, query, N=10):
        """
        Search for papers on Semantic Scholar using a query string.

        :param query: The search query.
        :param N: Number of results to retrieve.
        :return: A list of summary strings for the top papers.
        """
        paper_sums = []
        results = self.sch_engine.search_paper(
            query, limit=N, min_citation_count=3, open_access_pdf=True
        )
        for i, paper in enumerate(results):
            paper_sum = f"Title: {paper.title}\n"
            paper_sum += f"Abstract: {paper.abstract}\n"
            paper_sum += f"Citations: {paper.citationCount}\n"
            if paper.publicationDate:
                pub_year = paper.publicationDate.year
                pub_month = paper.publicationDate.month
                pub_day = paper.publicationDate.day
            else:
                pub_year = pub_month = pub_day = None

            paper_sum += (
                f"Release Date: year {pub_year}, month {pub_month}, day {pub_day}\n"
            )
            paper_sum += f"Venue: {paper.venue}\n"
            paper_sum += f"Paper ID: {paper.externalIds.get('DOI', 'N/A')}\n"
            paper_sums.append(paper_sum)

        return paper_sums

    def retrieve_full_paper_text(self, query):
        """
        Future extension: retrieve full text from Semantic Scholar 
        if available.
        """
        pass


#########################################################
## Arxiv Search
#########################################################
class ArxivSearch:
    """
    Wrapper around the `arxiv` Python package for searching
    arXiv papers and optionally retrieving full PDF text.
    """

    def __init__(self):
        self.sch_engine = arxiv.Client()

    def _process_query(self, query: str) -> str:
        """
        Process query string to fit within a size limit, 
        preserving as much info as possible.
        """
        MAX_QUERY_LENGTH = 300
        if len(query) <= MAX_QUERY_LENGTH:
            return query

        # Truncate query by words
        words = query.split()
        processed_query = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= MAX_QUERY_LENGTH:
                processed_query.append(word)
                current_length += len(word) + 1
            else:
                break

        return ' '.join(processed_query)

    def find_papers_by_str(self, query, N=20):
        """
        Search arXiv by abstract content, returning up to N results.

        :param query: Search query string
        :param N: Number of results
        :return: String listing summary info for the top papers
        """
        processed_query = self._process_query(query)
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                search = arxiv.Search(
                    query="abs:" + processed_query,
                    max_results=N,
                    sort_by=arxiv.SortCriterion.Relevance
                )

                paper_sums = []
                for r in self.sch_engine.results(search):
                    paperid = r.pdf_url.split("/")[-1]
                    pubdate = str(r.published).split(" ")[0]
                    paper_sum = f"Title: {r.title}\n"
                    paper_sum += f"Summary: {r.summary}\n"
                    paper_sum += f"Publication Date: {pubdate}\n"
                    paper_sum += f"Categories: {' '.join(r.categories)}\n"
                    paper_sum += f"arXiv paper ID: {paperid}\n"
                    paper_sums.append(paper_sum)

                time.sleep(2.0)
                return "\n".join(paper_sums)

            except Exception:
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2 * retry_count)
                    continue

        return None

    def retrieve_full_paper_text(self, query):
        """
        Download and extract full text from a PDF on arXiv.

        :param query: arXiv paper ID
        :return: String of extracted text from all pages
        """
        pdf_text = ""
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[query])))

        # Download the PDF to 'downloaded-paper.pdf'
        paper.download_pdf(filename="downloaded-paper.pdf")

        # Read the PDF
        reader = PdfReader("downloaded-paper.pdf")
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
            except Exception:
                os.remove("downloaded-paper.pdf")
                time.sleep(2.0)
                return "EXTRACTION FAILED"

            pdf_text += f"--- Page {page_number} ---\n{text}\n"

        # Clean up
        os.remove("downloaded-paper.pdf")
        time.sleep(2.0)
        return pdf_text


#########################################################
## EXECUTE_CODE Function
#########################################################
import io
import sys
import traceback
import concurrent.futures
import multiprocessing
import io
import sys
import traceback

def execute_code(code_str, timeout=60, MAX_LEN=1000):
    """
    Executes a given Python code string within a controlled 
    environment/thread, capturing its stdout.

    :param code_str: The Python code to execute as a string.
    :param timeout: Max seconds to allow code execution.
    :param MAX_LEN: Maximum length of captured stdout to return.
    :return: First MAX_LEN characters of the combined stdout or error message.
    """

    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt

    # Basic security checks
    if "load_dataset('pubmed" in code_str:
        return "[CODE EXECUTION ERROR] 'pubmed' dataset is disallowed due to excessive download times."
    if "exit(" in code_str:
        return "[CODE EXECUTION ERROR] The exit() command is not allowed."

    output_capture = io.StringIO()
    sys.stdout = output_capture

    exec_globals = globals()

    def run_code():
        try:
            exec(code_str, exec_globals)
        except Exception as e:
            output_capture.write(f"[CODE EXECUTION ERROR]: {str(e)}\n")
            traceback.print_exc(file=output_capture)

    try:
        # Run in a thread with concurrency
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_code)
            future.result(timeout=timeout)

    except concurrent.futures.TimeoutError:
        return f"[CODE EXECUTION ERROR]: Code execution exceeded {timeout} seconds."
    except Exception as e:
        return f"[CODE EXECUTION ERROR]: {str(e)}"
    finally:
        sys.stdout = sys.__stdout__

    return output_capture.getvalue()[:MAX_LEN]
