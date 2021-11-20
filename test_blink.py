import blink.my_main_dense_coverage as main_dense
import argparse
from BLINK.blink.annotator import Annotator

def _annotate(input_sentences):

    annotator = Annotator()
    sentence = input_sentences[0]
    mentions = annotator.extract_name(sentence)

    samples = []
    for mention in mentions:
        record = {}
        record["label"] = "unknown"
        record["label_id"] = -1
        # LOWERCASE EVERYTHING !
        record["context_left"] = sentence[0:mention["start_pos"]].lower()
        record["context_right"] = sentence[mention["end_pos"]:-1].lower()
        record["mention"] = mention["entity"].lower()
        record["start_pos"] = int(mention["start_pos"])
        record["end_pos"] = int(mention["end_pos"])
        #record["sent_idx"] = mention["sent_idx"]
        samples.append(record)
    return samples


models_path = "models/" # the path where you stored the BLINK models

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 10,
    "biencoder_model": models_path+"biencoder_wiki_large.bin",
    "biencoder_config": models_path+"biencoder_wiki_large.json",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"crossencoder_wiki_large.json",
    "fast": False, # set this to be true if speed is a concern
    "output_path": "logs/", # logging directory,
    "faiss_index":"hnsw",
    "index_path":models_path+"faiss_hnsw_index4.pkl"
}

args = argparse.Namespace(**config)

models = main_dense.load_models(args, logger=None)

data_to_link = [ {
                    "id": 0,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "".lower(),
                    "mention": "Shakespeare".lower(),
                    "context_right": "'s account of the Roman general Julius Caesar's murder by his friend Brutus is a meditation on duty.".lower(),
                },
                {
                    "id": 1,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "Shakespeare's account of the Roman general".lower(),
                    "mention": "Julius Caesar".lower(),
                    "context_right": "'s murder by his friend Brutus is a meditation on duty.".lower(),
                }
                ]

#_, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
a, b, c, predictions, scores,my_result = main_dense.run(args, None, *models, test_data=data_to_link)

print(my_result)

annotation = _annotate("Pick and sterilize the correct tool to cut the bamboo. The tool you use will depend on how thick and hearty your bamboo is. If you have thin bamboo, you may be able to use a sharp knife. If your bamboo is heartier, you may have to use a handsaw. Whatever tool you end up using, sterilize it first with household disinfectants, such as diluted bleach or rubbing alcohol.")

a, b, c, predictions, scores,my_result = main_dense.run(args, None, *models, test_data=annotation)

print(my_result)