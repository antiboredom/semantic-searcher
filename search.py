"""
This is basically just a command line tool that wraps the semantic search example from:
https://www.sbert.net/examples/applications/semantic-search/README.html
"""

import sys
import argparse
import pickle
import os
import hashlib
from sentence_transformers import SentenceTransformer, util
import torch
from nltk.tokenize import sent_tokenize


def main(text, query, max_results=100, memoize=True):

    # sentence tokenize our text
    corpus = sent_tokenize(text)
    corpus = [s.strip() for s in corpus]
    corpus = [s for s in corpus if s != "" and s != '"']

    # load up the embedder
    # for a full list of pretrained models, see: https://www.sbert.net/docs/pretrained_models.html
    # embedder = SentenceTransformer("paraphrase-distilroberta-base-v1")
    embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # hash the text for memoization
    hash_object = hashlib.md5(text.encode())
    picklename = f"embeddings_{hash_object.hexdigest()}.pkl"

    # if we've already processed the text, load it
    if os.path.exists(picklename):
        with open(picklename, "rb") as filein:
            corpus_embeddings = pickle.load(filein)
    else:
        # otherwise make the embeddings
        corpus_embeddings = embedder.encode(
            corpus, convert_to_tensor=True, show_progress_bar=True
        )
        if memoize:
            with open(picklename, "wb") as fileout:
                pickle.dump(
                    corpus_embeddings, fileout, protocol=pickle.HIGHEST_PROTOCOL
                )

    # how many results should we return?
    top_k = min(max_results, len(corpus))

    # generate embeddings for the query
    query_embedding = embedder.encode(
        query, convert_to_tensor=True, show_progress_bar=False
    )

    # Find the the sentences most similar to our query based cosine similarity
    # note that top_results[0] contains the scores, and top_results[1] indices
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    indices = top_results[1]
    results = [corpus[i] for i in indices]

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Semantic Searcher")
    parser.add_argument(
        "infile", nargs="?", type=argparse.FileType("r"), default=sys.stdin
    )
    parser.add_argument("--query", "-q")
    parser.add_argument("--max", type=int, default=100)

    args = parser.parse_args()
    text = args.infile.read()

    results = main(text, args.query, args.max)

    for r in results:
        print(r)
