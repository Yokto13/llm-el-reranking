# LLM based reranker for entity linking

This is a simple proof of concept reranker based on 8b Llama 3.1. 

The idea is that an LLM could efficiently rerank TOP entities retrieved by [some more efficient process](https://github.com/yokto13/mel). Surprisingly, this does not seem to be the case as Mewsli-9 results show weak performance. I might try a different model in the future. Part of the problem might also come from the prompt, which could benefit from some adjustments.

## How to run this

`main.py` is the entry point to the program. All variables that need to be adjusted are at the top of it.
Currently, the code expects data from my [other project](https://github.com/yokto13/mel) to work but can be certainly adjusted. To do that change `load_mewsli` and `load_candidates` in `main.py` accordingly.

## Mewsli-9 experiment

For each item in Mewsli-9, I generate 10 candidates by a custom retrieval model. These 10 candidates are then supplied to the LLM with a prompt that asks it to return the index of the correct candidate.

| Language   | R@1  | Upper bound |
|------------|------|-------------|
| Arabic     | 56.3 | 94.3       |
| German     | 65.8 | 94.8       |
| English    | 65.3 | 91.6       |
| Spanish    | 64.2 | 93.4       |
| Persian    | 65.8 | 95.5       |
| Japanese   | 47.4 | 94.2       |
| Serbian    | 70.5 | 96.1       |
| Tamil      | 57.1 | 96.4       |
| Turkish    | 55.7 | 93.4       |


