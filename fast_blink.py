import logging

from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import uvicorn

from blink.annotator import Annotator
import blink.my_main_dense_coverage as main_dense
import argparse

app = FastAPI()

logger = logging.getLogger(__name__)

models_path = "models/"  # the path where you stored the BLINK models

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 5,
    "biencoder_model": models_path + "biencoder_wiki_large.bin",
    "biencoder_config": models_path + "biencoder_wiki_large.json",
    "entity_catalogue": models_path + "entity.jsonl",
    "entity_encoding": models_path + "all_entities_large.t7",
    "crossencoder_model": models_path + "crossencoder_wiki_large.bin",
    "crossencoder_config": models_path + "crossencoder_wiki_large.json",
    "fast": False,  # set this to be true if speed is a concern
    "output_path": "logs/",  # logging directory,
    "faiss_index": "hnsw",
    "index_path": models_path + "faiss_hnsw_index4.pkl"
}

args = argparse.Namespace(**config)

models = None
models = main_dense.load_models(args, logger=logger)

my_annotator = Annotator()


def _annotate(annotator, sentence):
    mentions = annotator.extract_name(sentence)
    samples = []
    for mention in mentions:
        record = {}
        record["label"] = "unknown"
        record["label_id"] = -1
        # LOWERCASE EVERYTHING !
        record["context_left"] = sentence[:mention["start_pos"]].lower()
        record["context_right"] = sentence[mention["end_pos"]:].lower()
        record["mention"] = mention["entity"].lower()
        record["start_pos"] = int(mention["start_pos"])
        record["end_pos"] = int(mention["end_pos"])
        # record["sent_idx"] = mention["sent_idx"]
        samples.append(record)
    return samples


def _annotate_multiple(annotator, sentence_list):
    len_lst = []
    ant_lst = []
    for sentence in sentence_list:
        annotation = _annotate(annotator=annotator, sentence=sentence)
        len_lst.append(len(annotation))
        ant_lst.extend(annotation)
    return len_lst, ant_lst


class MyData(BaseModel):
    data: str


@app.post("/get_mentions")
def get_mentions(sentence: MyData):
    annotation = _annotate(my_annotator, sentence.data)
    return annotation


class Passage(BaseModel):
    id: Optional[str]
    path: Optional[str]
    text: str


@app.post("/link_mentions_crossencoder_multiple")
def link_mentions_crossencoder_multiple(text_lst: List[Passage]):
    text_lst = jsonable_encoder(text_lst)

    # logging.debug(text_lst)
    sentences_list = [i["text"] for i in list(text_lst)]

    print(sentences_list)
    idx_count_list, annotations_list = _annotate_multiple(my_annotator, sentence_list=sentences_list)
    # print("XXXXXXXXXXX",idx_count_list,"YYYYYYYYYYYY", annotations_list,"ZZZZZZZZZZZZZ")
    logging.info(idx_count_list)
    _, _, _, predictions, scores, my_result = main_dense.run(args, logger, *models, test_data=annotations_list)
    # print("AAAAAAAAAAA",my_result,"BBBBBBBBBBBBB")
    pointer = 0
    ret = []
    for sentence_annotation_count, input_sentence, in zip(idx_count_list, text_lst):
        obj = input_sentence.copy()
        obj["entities"] = (my_result[pointer:(sentence_annotation_count + pointer)]).copy()
        # print(obj)
        pointer += sentence_annotation_count
        ret.append(obj)
    logging.info(obj)

    return ret


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
