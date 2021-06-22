import sys
import argparse
import pickle
import os
import hashlib
from sentence_transformers import SentenceTransformer, util
import torch
from nltk.tokenize import sent_tokenize


def main(text, query, max_results=100, memoize=True):
    corpus = sent_tokenize(text)
    corpus = [s.strip() for s in corpus]
    corpus = [s for s in corpus if s != "" and s != '"']

    # embedder = SentenceTransformer("paraphrase-distilroberta-base-v1")
    embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    hash_object = hashlib.md5(text.encode())
    picklename = f"embeddings_{hash_object.hexdigest()}.pkl"

    if os.path.exists(picklename):
        with open(picklename, "rb") as filein:
            corpus_embeddings = pickle.load(filein)
    else:
        corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
        if memoize:
            with open(picklename, "wb") as fileout:
                pickle.dump(corpus_embeddings, fileout, protocol=pickle.HIGHEST_PROTOCOL)

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(max_results, len(corpus))
    query_embedding = embedder.encode(query, convert_to_tensor=True, show_progress_bar=False)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    for score, idx in zip(top_results[0], top_results[1]):
        # print(corpus[idx], "(Score: {:.4f})".format(score))
        print(corpus[idx])

    """
    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Semantic Searcher")
    parser.add_argument(
        "infile", nargs="?", type=argparse.FileType("r"), default=sys.stdin
    )
    parser.add_argument("--query", "-q")
    parser.add_argument("--max", default=100)

    args = parser.parse_args()
    text = args.infile.read()

    main(text, args.query, args.max)
