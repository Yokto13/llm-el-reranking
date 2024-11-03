import torch
from transformers import pipeline

import os

token = os.environ.get("HFTOKEN")
os.environ["HF_HOME"] = "/lnet/work/home-students-external/farhan/troja/hf_cache"

SYSTEM_MESSAGE = {
    "role": "system",
    "content": """You are an expert in entity linking. You are given a query and a list of candidates.
    The query is part of a text where a certain entity is denoted with [M] tags.
    Candidates are short descriptions of entities.
    Your task is to return **ID** corresponding to the candidate that corresponds to the entity in the query.
    Answer number and shore explanation of your choice.

    The input will be in the following format:
    Query: <query>
    Candidates: 
    1. <candidate1>
    2. <candidate2>
    3. <candidate3>
    ...

    ANSWER FORMAT: Start <ID> End

    Commentary:
        <commentary>

    EXAMPLE:
    Query: To save Troy, [M] Paris [M] had to be sacrificed.
    Candidates: 
    1. [M] Paris [M] Paris (French pronunciation: [paʁi] ⓘ) is the capital and largest city of France. 
    2. [M] Paris [M] Paris (řecky: Πάρις) je postava z řecké mytologie, syn Priama, krále Tróje, a jeho manželky Hekabé.
    3. [M] Paris Hilton [M] Paris Whitney Hilton (born February 17, 1981)[3][4] is an American media personality, businesswoman, socialite, model, singe

    Expected output: 
    Start 2 End
    Commentary:
        The query is about mythology because it talks about Troy.
        The first candidate is about France, the second is about mythology and the third is about a celebrity.
        The correct answer is 2 because it is about Paris from mythology.
    """,
}


device = "cuda" if torch.cuda.is_available() else "cpu"


class Reranker:
    def __init__(self, model_name):
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            token=token,
            device=device,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

    def rerank(self, query: str, candidates: list[str]) -> int:
        query_message_text = f"Query: {query}"
        candidates_text = "\n".join(
            [f"{i + 1}. {c}\n " for i, c in enumerate(candidates)]
        )
        messages = [
            SYSTEM_MESSAGE,
            {
                "role": "user",
                "content": query_message_text + "\n Candidates: \n" + candidates_text,
            },
        ]

        outputs = self.pipe(
            messages,
            max_new_tokens=8,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
            # cache_implementation="static",
        )

        content = outputs[0]["generated_text"][-1]["content"]

        try:
            content = content.split("Start")[1].split("End")[0]
        except IndexError:
            print("Failed to parse result")
            return 0

        content = content.strip()
        content = content.lower()
        if "<id>" in content:
            content = content.replace("<id>", "")
        if "id" in content:
            content = content.replace("id", "")
        content = "".join((c for c in content if c.isdigit()))

        try:
            res = int(content) - 1
            if res >= 0 and res < len(candidates):
                return res
            raise ValueError(f"Invalid result: {res}")
        except ValueError:
            print("Failed to parse result")
            return 0
