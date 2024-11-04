from tqdm import tqdm

import numpy as np
from transformers import AutoTokenizer

from reranker import Reranker
from wiki_retriever import WikiRetriever
from remap_loader import load_qids_remap


model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
retriever_tokenizer_name = (
    "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"
)
retriever_tokenized_data_dir = "/lnet/work/home-students-external/farhan/troja/outputs/workdirs/all/damuel_descs_together_tokens"
mewsli_path = "/lnet/work/home-students-external/farhan/troja/outputs/tokens_mewsli_finetuning/de/mentions_1215272744215592859.npz"
candidates_path = "de.npz"


_qids_remap = load_qids_remap("damuel_1.1-dev_qid_redirects.json.xz")


def load_mewsli(
    max_mentions: int, tokenizer: AutoTokenizer
) -> tuple[list[str], list[int]]:
    mewsli = np.load(mewsli_path)
    queries = [tokenizer.decode(tokens) for tokens in mewsli["tokens"][:max_mentions]]
    qids = [
        _qids_remap[qid] if qid in _qids_remap else qid
        for qid in mewsli["qids"][:max_mentions]
    ]
    return queries, qids


def load_candidates(
    max_mentions: int, retriever: WikiRetriever
) -> tuple[list[list[str]], list[list[int]]]:
    candidates_qids = np.load(candidates_path)["candidate_qids"][:max_mentions]
    candidates = [
        [retriever.query(qid) for qid in candidates_qids[i] if qid != -1]
        for i in range(len(candidates_qids))
    ]
    return candidates, candidates_qids


def compute_accuracy(
    reranker: Reranker,
    queries: list[str],
    mewsli_qids: list[int],
    candidates: list[list[str]],
    candidates_qids: list[list[int]],
) -> None:
    correct = 0
    upper_bound = 0

    for i, (query, candidate) in tqdm(
        enumerate(zip(queries, candidates)), total=len(queries)
    ):
        result = reranker.rerank(query, candidate)

        if mewsli_qids[i] == candidates_qids[i][result]:
            correct += 1
        if mewsli_qids[i] in candidates_qids[i]:
            upper_bound += 1
        if i % 50 == 0:
            print(f"Current accuracy: {correct / (i + 1)}")
    print(f"Accuracy: {correct / len(queries)}")
    print(f"Upper bound: {upper_bound / len(queries)}")


def main(max_mentions: int) -> None:
    reranker = Reranker(model_name)
    retriever = WikiRetriever(retriever_tokenizer_name, retriever_tokenized_data_dir)
    tokenizer = retriever.tokenizer

    queries, mewsli_qids = load_mewsli(max_mentions, tokenizer)
    candidates, candidates_qids = load_candidates(max_mentions, retriever)

    compute_accuracy(reranker, queries, mewsli_qids, candidates, candidates_qids)


if __name__ == "__main__":
    main(100000)
