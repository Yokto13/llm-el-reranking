# LLM based reranker for entity linking

This is a simple proof of concept reranker based on 8b Llama 3.1. 

The idea is that an LLM could efficiently rerank TOP entities retrieved by [some more efficient process](https://github.com/yokto13/mel). Surprisingly, this does not seem to be the case as Mewsli-9 results show weak performance. I might try a different model in the future. Part of the problem might also come from the prompt, which could benefit from some adjustments.

## How to run this

`main.py` is the entry point to the program. All variables that need to be adjusted are at the top of it.
Currently, the code expects data from my [other project](https://github.com/yokto13/mel) to work but can be certainly adjusted. To do that change `load_mewsli` and `load_candidates` in `main.py` accordingly.

## Mewsli-9 results (10 candidates)

| Lang | R@1  | Upper bound |
|------|------|-------------|
| ar   | 56.3 | TBA        |
| de   | 65.8 | TBA        |
| en   | 65.3 | TBA        |
| es   | 64.2 | TBA        |
| fa   | 65.8 | TBA        |
| ja   | 47.4 | TBA        |
| sr   | 70.5 | TBA        |
| ta   | 57.1 | TBA        |
| tr   | 55.7 | TBA        |
