#-*- encoding: utf-8 -*-
import datasets
import json
import re
from typing import Dict, List, Set
from collections import defaultdict

from .analyze_multiwoz import normalize, restore_common_abbr, normalize_state_value

path_woz22 = "./version22/data/"
data_files = {
    "conf10": path_woz22 + "MultiWOZ_1.0/data.json",
    "conf20": path_woz22 + "MultiWOZ_2.0/data.json",
    "conf21": path_woz22 + "MultiWOZ_2.1/data.json",
    "conf22": path_woz22 + "MultiWOZ_2.2/data.json",
    "conf23": "./version23/data.json",
    "conf24": "./version24/data/data.json"
}
val_files = {
    "conf10": path_woz22 + "MultiWOZ_1.0/valListFile.json",
    "conf20": path_woz22 + "MultiWOZ_2.0/valListFile.json",
    "conf21": path_woz22 + "MultiWOZ_2.1/valListFile.txt",
    "conf22": path_woz22 + "MultiWOZ_2.1/valListFile.txt",
    "conf23": path_woz22 + "MultiWOZ_2.1/valListFile.txt",
    "conf24": "./version24/data/valListFile.json"
}

test_files = {
    "conf10": path_woz22 + "MultiWOZ_1.0/testListFile.json",
    "conf20": path_woz22 + "MultiWOZ_2.0/testListFile.json",
    "conf21": path_woz22 + "MultiWOZ_2.1/testListFile.txt",
    "conf22": path_woz22 + "MultiWOZ_2.1/testListFile.txt",
    "conf23": path_woz22 + "MultiWOZ_2.1/testListFile.txt",
    "conf24": "./version24/data/testListFile.json"
}
dialogue_action_files = {
    "conf10": None,
    "conf20": path_woz22 + "MultiWOZ_2.0/dialogue_acts.json",
    "conf21": path_woz22 + "MultiWOZ_2.0/dialogue_acts.json",
    "conf22": path_woz22 + "MultiWOZ_2.2/dialog_acts.json",
    "conf23": "./version23/dialogue_acts.json",
    "conf24": "./version24/data/dialogue_acts.json"
}
SCHEMA_22_PATH = "./version22/data/MultiWOZ_2.2/schema.json"
ONTOLOGY_24_PATH = "./version24/data/ontology.json"


MAPPING_PAIR_FILE = "./version22/utils/mapping.pair"
## following codes are copied from multiwoz2.1 preprocess

replacements = []
def load_from_file(path):
    ret = None
    with open(path, "r") as fin:
        ret = json.load(fin)
    assert(ret is not None)
    return ret

class MultiWOZ(datasets.GeneratorBasedBuilder):
    version=datasets.Version("1.2.0"), 
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="conf20",
            description="MultiWOZ2.0"
        ),
        datasets.BuilderConfig(
            name="conf21", 
            description="MultiWOZ2.1"
        ),
        datasets.BuilderConfig(
            name="conf22",
            description="MultiWOZ2.2"
        ),
        datasets.BuilderConfig(
            name="conf23",
            description="MultiWOZ2.3"
        ),
        datasets.BuilderConfig(
        name="conf24", 
        description="MultiWOZ2.4"
        )
    ]
    DEFAULT_CONFIG_NAME = "conf21"
    domain_slot_type = dict()
    num_context_turn = 5

    def _info(self):
        features = datasets.Features(
            {
                "dialogue_id": datasets.Value("string"),
                "turn_id": datasets.Value("int32"),
                "context": datasets.Value("string"),
                "sys_utt": datasets.Value("string"),
                "usr_utt": datasets.Value("string"),
                "prev_states": datasets.Sequence(
                    datasets.Features({
                    "slot_name": datasets.Value("string"),
                    "slot_value": datasets.Sequence(datasets.Value("string"))
                    }),
                ),
                "states": datasets.Sequence(
                    datasets.Features({
                    "slot_name": datasets.Value("string"),
                    "slot_value": datasets.Sequence(datasets.Value("string"))
                    }),
                ),
            }
        )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        data_file = dl_manager.download_and_extract(data_files[self.config.name])
        val_list_file = dl_manager.download_and_extract(val_files[self.config.name])
        test_list_file = dl_manager.download_and_extract(test_files[self.config.name])
        schema_file = dl_manager.download_and_extract(SCHEMA_22_PATH)
        ontology_file = dl_manager.download_and_extract(ONTOLOGY_24_PATH)
        data = json.load(open(data_file))
        val_dialogue_ids = set([dial_id.strip() for dial_id in open(val_list_file).readlines()])
        test_dialogue_ids = set([dial_id.strip() for dial_id in open(test_list_file).readlines()])
        train_dialogue_ids = set(data.keys()) - val_dialogue_ids - test_dialogue_ids
        
        global replacements
        replacement_file = dl_manager.download_and_extract(MAPPING_PAIR_FILE)
        with open(replacement_file, "r") as fin:
            for line in fin.readlines():
                tok_from, tok_to = line.replace('\n', '').split('\t')
                replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

        schema = load_from_file(schema_path)
        for d in schema:
            for s in d["slots"]:
                k = s["name"]
                self.domain_slot_type[k] = s["is_categorical"]

        return [
            datasets.SplitGenerator(
                name=spl_enum,
                gen_kwargs={
                    "data_path": data_file,
                    "dialogue_ids": dialogue_ids,
                    "schema_path": schema_file,
                    "ontology_path": ontology_file
                },
            )
            for spl_enum, dialogue_ids in [
                (datasets.Split.TRAIN, train_dialogue_ids),
                (datasets.Split.VALIDATION, val_dialogue_ids),
                (datasets.Split.TEST, test_dialogue_ids),
            ]
        ]

    def _generate_examples(self, data_path, dialogue_ids, schema_path, ontology_path):
        dialogues = load_from_file(data_path)
        ontology = load_from_file(ontology_path)
        id_ = -1
        res = None
        for dial_id, dial in dialogues.items():
            if dial_id not in dialogue_ids:
                continue
            prev_states, context, sys_utt, usr_utt = dict(), [], "none", "none"
            for turn_id, turn in enumerate(dial["log"]):
                role = "usr" if turn_id % 2 == 0  else "sys"
                utterance = restore_common_abbr(normalize(turn['text'].strip(), replacements))
                if role == "usr":
                    usr_utt = utterance
                else:
                    sys_utt = utterance
                    states = self.extract_states(turn["metadata"])
                    cur_states = {
                        "slot_name": [k for k, _ in states.items()],
                        "slot_value": [val for _, val in states.items()]
                    }
                    state_value_num = sum([len(val) for _, val in states.items()])
                    if len(prev_states) == 0:
                        prev_states = {
                            "slot_name": [k for k, _ in states.items()],
                            "slot_value": [ [] for _, val in states.items()]
                        }
                    id_ += 1
                    res = {
                        "dialogue_id": dial_id,
                        "turn_id": turn_id,
                        "context": " ".join(context[-self.num_context_turn:]),
                        "usr_utt": usr_utt,
                        "sys_utt": sys_utt,
                        "prev_states": prev_states,
                        "states": cur_states
                    }
                    context.append(usr_utt + " [end of user text] " + sys_utt + " [end of system text] ")
                    prev_states = cur_states
                    # remove empty state
                    if state_value_num == 0 or len(usr_utt) == 0:
                        id_ -= 1
                    else:
                        yield id_, res

    def extract_states(self, metadata):
        # metadata -> domain -> semi -> slots -> value
            """
            an example:
            "hotel": {
                    "book": {
                        "booked": [],
                        "stay": "3",
                        "day": "tuesday",
                        "people": "6"
                    },
                    "semi": {
                        "name": "not mentioned",
                        "area": "not mentioned",
                        "parking": "yes",
                        "pricerange": "cheap",
                        "stars": "not mentioned",
                        "internet": "not mentioned",
                        "type": "hotel"
                    }
            """
            belief_state = dict()
            normalize_slot = lambda x: "".join([" " + i.lower() if "A"<=i<="Z" else i for i in x])
            for domain, item in metadata.items():
                for k in ["semi", "book"]:
                    for slot, value in item[k].items():
                        if k == "semi" and slot == "pricerange":
                            slot = "price range"
                        if k == "book":
                            if slot == "booked":
                                continue
                            slot = "book " + slot
                        slot = normalize_slot(slot)
                        domain_slot = domain + "-" + slot
                        value = normalize_state_value(domain, slot, value.strip(), replacements, remove_none=True)
                        belief_state[domain_slot] = value
            return belief_state
