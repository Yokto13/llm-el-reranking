"""
    DaMuEL (which I use for the KB) and Mewsli-9 where scrapped in different timeframes. 
    Due to the continous changes of the Wikidata, the QIDs are not the same.
    This script loads the remapping dict that allows us to map all QIDs to the same 'space'.
"""

import json
import lzma
from pathlib import Path


def _load_json_file(filepath: str | Path) -> dict:
    if str(filepath).endswith(".xz"):
        with lzma.open(filepath, "r") as f:
            return json.loads(f.read())
    with open(filepath, "r") as f:
        return json.loads(f.read())


def _convert_qid_keys_to_int(qid_map: dict) -> dict[int, int]:
    return {int(k[1:]): int(v[1:]) for k, v in qid_map.items()}


def load_qids_remap(filepath: str | Path) -> dict[int, int]:
    qid_map = _load_json_file(filepath)
    converted = _convert_qid_keys_to_int(qid_map)
    return converted
