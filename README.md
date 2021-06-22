# Semantic Text Searcher

A simple utility for doing semantic text search on the command line. Give it a text file and a query, and it will return the sentences from the text file that are most similar to your query sentence.

This more or less just wraps the semantic search example from: https://www.sbert.net/examples/applications/semantic-search/README.html

## Install and Usage

Install:

```
pip install -r requirements.txt
```


Use:

```
python search.py --query [QUERY] [TEXTFILE.txt]
```

For example, to search for sentences that are most similar to "Hope exists but not for us" in War and Peace, you would run this command:

```
python search.py --query "Hope exists but not for us" warandpeace.txt
```

Note that the first time you search through a particular text it take a moment to generate text embeddings. Subsequent searches will be faster!
