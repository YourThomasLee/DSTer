#-*- encoding: utf8 -*- 
import matplotlib
from collections import defaultdict
import json
import math
import pdb
import pandas as pd

from trainer.data_loaders import WOZDataLoader


def load_domain_slot_type(file_path):
    ret = dict()
    schema = json.load(open(file_path))
    patterns = [("leaveat", "leave at"), ("pricerange", "price range"),
                ("bookpeople", "book people"), ("bookstay", "book stay"),
                ("booktime", "book time"), ("arriveby", "arrive by"),
                ("bookday", "book day")]
    for d in schema:
        for s in d["slots"]:
            k = s["name"].lower()
            for source,target in patterns:
                k = k.replace(source, target)
            ret[k] = s["is_categorical"]
    return ret

def load_data(data_dir, schema_file, version, batch_size):
    domain_slot_type = load_domain_slot_type(schema_file) # load slots type
    data_loader = WOZDataLoader(data_dir, batch_size, shuffle=False, 
                 validation_split=0.1, num_workers=0, 
                 version=version, training=True) # load train, valid datasets
    valid_loader = data_loader.split_validation()
    test_loader = WOZDataLoader(data_dir, batch_size, shuffle=False, 
                 validation_split=0.1, num_workers=0, 
                 version=version, training=False) #load test dataset
    return domain_slot_type, data_loader, valid_loader, test_loader

def collect_labels(data_loader):
    ret = dict()
    for idx, item in enumerate(data_loader):
        for k,vals in item['cur_states'].items():
            if k not in ret:
                ret[k] = defaultdict(int)
            for v in vals:
                ret[k][v] += 1
    return ret

def analyze_labels(train_loader, valid_loader, test_loader, slot_types):
    train_labels = collect_labels(train_loader)
    valid_labels = collect_labels(valid_loader)
    test_labels = collect_labels(test_loader)
    assert(set(train_labels.keys()) == set(valid_labels.keys()) and set(valid_labels.keys()) == set(test_labels.keys()))
    label_freq = {k: [0, 0, 0] for k in train_labels}
    for ds in label_freq:
        label_freq[ds][0] = sum([v for k, v in train_labels[ds].items() if k != "none"])
        label_freq[ds][1] = sum([v for k, v in valid_labels[ds].items() if k != "none"])
        label_freq[ds][2] = sum([v for k, v in test_labels[ds].items() if k != "none"])
    
    def entropy(l):
        s = sum(l)
        l = [i/s for i in l]
        return -sum([i * math.log(i) for i in l])


    def cross_entropy(d1, d2):
        # -p1 log(p2)
        all_candidates = set([k for k in d1 if k != "none"] + [l for l in d2 if l !=  "none"])
        s1 = sum([d1.get(k, 0) + 1 for k in all_candidates if k != "none"]) #laplace transformation
        s2 = sum([d2.get(k, 0) + 1 for k in all_candidates if k != "none"])
        crs_ent = 0
        for k in all_candidates:
            crs_ent += -(d1[k] + 1) / s1 * math.log((d2[k] + 1) / s2)
        return crs_ent

    label_ent = {k: [0, 0, 0] for k in train_labels}
    for ds in label_ent:
        label_ent[ds][0] = entropy([v for k, v in train_labels[ds].items() if k != "none"])
        label_ent[ds][1] = entropy([v for k, v in valid_labels[ds].items() if k != "none"])
        label_ent[ds][2] = entropy([v for k, v in test_labels[ds].items() if k != "none"])
    
    label_crs = {k: [] for k in train_labels if slot_types[k]}
    value_domain = dict()
    for ds in train_labels:
        value_domain[ds] = dict()
        value_domain[ds].update(train_labels[ds])
        value_domain[ds].update(valid_labels[ds])
        value_domain[ds].update(test_labels[ds])
        if ds not in label_crs:
            continue
        label_crs[ds] = [cross_entropy(valid_labels[ds], train_labels[ds]),
                         cross_entropy(test_labels[ds], train_labels[ds]),
                         cross_entropy(test_labels[ds], valid_labels[ds])]
    slot_values = {k: list(v.keys()) for k,v in value_domain.items()}
    json.dump(slot_values, open("slot_values.json", "w"))
    return label_freq, label_ent, label_crs

if __name__ == "__main__":
    data_path = "./data/multiwoz"
    schema_file = "./data/multiwoz/version22/data/MultiWOZ_2.2/schema.json"
    slot_types, train_loader, valid_loader, test_loader = load_data(data_path, schema_file, version = "21", batch_size = 32)
    freq, ent, crs = analyze_labels(train_loader, valid_loader, test_loader, slot_types)
    import json, pdb
    pdb.set_trace()
    pd.DataFrame(freq).to_csv("freq.csv")
    pd.DataFrame(ent).to_csv("entropy.csv")
    pd.DataFrame(crs).to_csv("cross_entropy.csv")
    
