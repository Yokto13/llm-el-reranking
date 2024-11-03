import os
import tqdm

from transformers import AutoTokenizer
import numpy as np

from remap_loader import load_qids_remap

_qids_remap = load_qids_remap("damuel_1.1-dev_qid_redirects.json.xz")


print(len(_qids_remap))


class WikiRetriever:
    def __init__(self, model_name: str, data_path: str):
        self.model_name = model_name
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.qid_to_tokens = {}
        self.data = {}
        self.load_data()

    def load_data(self):
        for file in tqdm.tqdm(os.listdir(self.data_path)):
            if not file.endswith(".npz"):
                continue
            file_path = os.path.join(self.data_path, file)
            d = np.load(file_path)
            qids, tokens_list = d["qids"], d["tokens"]
            for qid, tokens in zip(qids, tokens_list):
                if qid in _qids_remap:
                    qid = _qids_remap[qid]
                self.qid_to_tokens[qid] = tokens

    def query(self, qid: int) -> str:
        if qid in _qids_remap:
            print(qid, _qids_remap[qid])
            qid = _qids_remap[qid]
        if qid not in self.data:
            self.data[qid] = self.tokenizer.decode(self.qid_to_tokens[qid])
        return self.data[qid]
