from pathlib import Path
from tqdm import tqdm
from pprint import pprint

import numpy as np

from reranker import Reranker
from wiki_retriever import WikiRetriever
from remap_loader import load_qids_remap


model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
retriever_tokenizer_name = (
    "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"
)
retriever_tokenized_data_dir = "/lnet/work/home-students-external/farhan/troja/outputs/workdirs/all/damuel_descs_together_tokens"
mewsli_path = "/lnet/work/home-students-external/farhan/troja/outputs/tokens_mewsli_finetuning/es/mentions_1307770978027216442.npz"
candidates_path = "es.npz"


_qids_remap = load_qids_remap("damuel_1.1-dev_qid_redirects.json.xz")


def main(max_mentions: int) -> None:
    reranker = Reranker(model_name)

    retriever = WikiRetriever(retriever_tokenizer_name, retriever_tokenized_data_dir)

    mewsli = np.load(mewsli_path)
    queries = [
        retriever.tokenizer.decode(tokens) for tokens in mewsli["tokens"][:max_mentions]
    ]
    mewsli_qids = [
        _qids_remap[qid] if qid in _qids_remap else qid
        for qid in mewsli["qids"][:max_mentions]
    ]

    candidates_qids = np.load(candidates_path)["candidate_qids"][:max_mentions]

    candidates = [
        [retriever.query(qid) for qid in candidates_qids[i] if qid != -1]
        for i in range(len(candidates_qids))
    ]

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


if __name__ == "__main__":
    main(100000)
