#-*- encoding: utf-8 -*-
import json
from typing import Dict, List, Set

import datasets
import sys
base_path = "./version22/data"
path_woz10 = base_path + "/MultiWOZ_1.0"
path_woz20 = base_path + "/MultiWOZ_2.0"
path_woz21 = base_path + "/MultiWOZ_2.1"
path_woz22 = base_path + "/MultiWOZ_2.2"
path_woz23 = "./version23"
path_woz24 = "./version24/data"
_URL_LIST = [
    ("dialogue_acts",path_woz22 + "/dialog_acts.json")
]
_URL_LIST += [
    (
        f"train_{i:03d}",
        path_woz22 + f"/train/dialogues_{i:03d}.json",
    )
    for i in range(1, 18)
]
_URL_LIST += [
    (
        f"dev_{i:03d}",
        path_woz22 + f"/dev/dialogues_{i:03d}.json",
    )
    for i in range(1, 3)
]
_URL_LIST += [
    (
        f"test_{i:03d}",
        path_woz22 + f"/test/dialogues_{i:03d}.json",
    )
    for i in range(1, 3)
]

_URLs22 = dict(_URL_LIST)
SCHEMA_22_PATH = "./version22/data/MultiWOZ_2.2/schema.json"
ONTOLOGY_24_PATH="./version24/data/ontology.json"

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
    DEFAULT_CONFIG_NAME = "conf22"

    def _info(self):
        features = datasets.Features(
            {
                "dialogue_id": datasets.Value("string"),
                "turn_id": datasets.Value("int32"),
                "sys_utt": datasets.Value("string"),
                "usr_utt": datasets.Value("string"),
                "states_dict": datasets.Sequence(
                    datasets.Features(
                        {
                            "slot_name": datasets.Value("string"),
                            "slot_value": datasets.Value("string"),
                        }
                    ),
                )
            }
        )
        # features = datasets.Features(
        #     {
        #         "dialogue_id": datasets.Value("string"),
        #         "services": datasets.Sequence(datasets.Value("string")),
        #         "turns": datasets.Sequence(
        #             {
        #                 "turn_id": datasets.Value("string"),
        #                 "speaker": datasets.ClassLabel(names=["USER", "SYSTEM"]),
        #                 "utterance": datasets.Value("string"),
        #                 "frames": datasets.Sequence(
        #                     {
        #                         "service": datasets.Value("string"),
        #                         "state": {
        #                             "active_intent": datasets.Value("string"),
        #                             "requested_slots": datasets.Sequence(datasets.Value("string")),
        #                             "slots_values": datasets.Sequence(
        #                                 {
        #                                     "slots_values_name": datasets.Value("string"),
        #                                     "slots_values_list": datasets.Sequence(datasets.Value("string")),
        #                                 }
        #                             ),
        #                         },
        #                         "slots": datasets.Sequence(
        #                             {
        #                                 "slot": datasets.Value("string"),
        #                                 "value": datasets.Value("string"),
        #                                 "start": datasets.Value("int32"),
        #                                 "exclusive_end": datasets.Value("int32"),
        #                                 "copy_from": datasets.Value("string"),
        #                                 "copy_from_value": datasets.Sequence(datasets.Value("string")),
        #                             }
        #                         ),
        #                     }
        #                 ),
        #                 "dialogue_acts": datasets.Features(
        #                     {
        #                         "dialog_act": datasets.Sequence(
        #                             {
        #                                 "act_type": datasets.Value("string"),
        #                                 "act_slots": datasets.Sequence(
        #                                     datasets.Features(
        #                                         {
        #                                             "slot_name": datasets.Value("string"),
        #                                             "slot_value": datasets.Value("string"),
        #                                         }
        #                                     ),
        #                                 ),
        #                             }
        #                         ),
        #                         "span_info": datasets.Sequence(
        #                             {
        #                                 "act_type": datasets.Value("string"),
        #                                 "act_slot_name": datasets.Value("string"),
        #                                 "act_slot_value": datasets.Value("string"),
        #                                 "span_start": datasets.Value("int32"),
        #                                 "span_end": datasets.Value("int32"),
        #                             }
        #                         ),
        #                     }
        #                 ),
        #             }
        #         ),
        #     }
        # )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        if self.config.name == "conf20":
            pass
        elif self.config.name == "conf21":
            data_file = dl_manager.download_and_extract(path_woz21 + "/data.json")
            val_list_file = dl_manager.download_and_extract(path_woz21 + "/valListFile.json")
            test_list_file = dl_manager.download_and_extract(path_woz21 + "/testListFile.json")
            data = json.load(open(data_file))
            val_dialogue_ids = [dial_id.strip() for dial_id in open(val_list_file).readlines()]
            test_dialogue_ids = [dial_id.strip() for dial_id in open(test_list_file).readlines()]
            train_dialogue_ids = list(set(data.keys()) - set(val_dialogue_ids + test_dialogue_ids))
            return [
                datasets.SplitGenerator(
                    name=spl_enum,
                    gen_kwargs={
                        "filepaths": data_file,
                        "split": spl,
                        "dialogue_ids": dialogue_ids
                    },
                )
                for spl, spl_enum, dialogue_ids in [
                    ("train", datasets.Split.TRAIN, train_dialogue_ids),
                    ("dev", datasets.Split.VALIDATION, val_dialogue_ids),
                    ("test", datasets.Split.TEST, test_dialogue_ids),
                ]
            ]
        elif self.config.name == "conf22":
            data_files = dl_manager.download_and_extract(_URLs22)
            self.stored_dialogue_acts = json.load(open(data_files["dialogue_acts"]))
            return [
                datasets.SplitGenerator(
                    name=spl_enum,
                    gen_kwargs={
                        "filepaths": data_files,
                        "split": spl,
                    },
                )
                for spl, spl_enum in [
                    ("train", datasets.Split.TRAIN),
                    ("dev", datasets.Split.VALIDATION),
                    ("test", datasets.Split.TEST),
                ]
            ]
        elif self.config.name == "conf23":
            pass
        elif self.config.name == "conf24":
            pass

    def _generate_examples(self, config):
        if self.config.name == "conf22":
            filepaths, split = config["file_path"], config["split"]
            id_ = -1
            file_list = [fpath for fname, fpath in filepaths.items() if fname.startswith(split)]
            for filepath in file_list:
                dialogues = json.load(open(filepath))
                for dialogue in dialogues:
                    id_ += 1
                    mapped_acts = self.stored_dialogue_acts.get(dialogue["dialogue_id"], {})
                    res = {
                        "dialogue_id": dialogue["dialogue_id"],
                        "services": dialogue["services"],
                        "turns": [
                            {
                                "turn_id": turn["turn_id"],
                                "speaker": turn["speaker"],
                                "utterance": turn["utterance"],
                                "frames": [
                                    {
                                        "service": frame["service"],
                                        "state": {
                                            "active_intent": frame["state"]["active_intent"] if "state" in frame else "",
                                            "requested_slots": frame["state"]["requested_slots"]
                                            if "state" in frame
                                            else [],
                                            "slots_values": {
                                                "slots_values_name": [
                                                    sv_name for sv_name, sv_list in frame["state"]["slot_values"].items()
                                                ]
                                                if "state" in frame
                                                else [],
                                                "slots_values_list": [
                                                    sv_list for sv_name, sv_list in frame["state"]["slot_values"].items()
                                                ]
                                                if "state" in frame
                                                else [],
                                            },
                                        },
                                        "slots": [
                                            {
                                                "slot": slot["slot"],
                                                "value": "" if "copy_from" in slot else slot["value"],
                                                "start": slot.get("start", -1),
                                                "exclusive_end": slot.get("exclusive_end", -1),
                                                "copy_from": slot.get("copy_from", ""),
                                                "copy_from_value": slot["value"] if "copy_from" in slot else [],
                                            }
                                            for slot in frame["slots"]
                                        ],
                                    }
                                    for frame in turn["frames"]
                                    if (
                                        "active_only" not in self.config.name
                                        or frame.get("state", {}).get("active_intent", "NONE") != "NONE"
                                    )
                                ],
                                "dialogue_acts": {
                                    "dialog_act": [
                                        {
                                            "act_type": act_type,
                                            "act_slots": {
                                                "slot_name": [sl_name for sl_name, sl_val in dialog_act],
                                                "slot_value": [sl_val for sl_name, sl_val in dialog_act],
                                            },
                                        }
                                        for act_type, dialog_act in mapped_acts.get(turn["turn_id"], {})
                                        .get("dialog_act", {})
                                        .items()
                                    ],
                                    "span_info": [
                                        {
                                            "act_type": span_info[0],
                                            "act_slot_name": span_info[1],
                                            "act_slot_value": span_info[2],
                                            "span_start": span_info[3],
                                            "span_end": span_info[4],
                                        }
                                        for span_info in mapped_acts.get(turn["turn_id"], {}).get("span_info", [])
                                    ],
                                },
                            }
                            for turn in dialogue["turns"]
                        ],
                    }
                    yield id_, res
