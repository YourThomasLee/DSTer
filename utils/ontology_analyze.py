#!/usr/bin python 
#-*- encoding: utf-8 -*-
import util

def illegal_state_value(onto_given, onto_generated):
    """
    onto_given: domain-slot -> value
    onto_generated: domain-slot -> value -> count
    """
    diff_dict = dict()
    for domain_slot, values in onto_generated.items():
        if domain_slot not in onto_given.keys():
            diff_dict[domain_slot] = values
            # diff_dict[domain_slot].pop("")
            print("illegal domain slot: ", domain_slot)
            continue
        else:
            if domain_slot not in diff_dict:
                diff_dict[domain_slot] = dict()
            for val, cnt in values.items():
                if val == "none" or len(val) == 0:
                    continue
                for v in val.split("|"):
                    if val not in onto_given[domain_slot]:
                        diff_dict[domain_slot][val] = cnt
    diff_dict = {k: v for k,v in diff_dict.items() if len(v.items()) > 0}
    return diff_dict

if __name__ == "__main__":
    onto_given = util.read_json("data/domain_data/ontology_given.json")
    onto_generated = util.read_json("data/domain_data/ontology_generated.json")
    diff = illegal_state_value(onto_given, onto_generated)
    util.write_json(diff, "data/domain_data/ontology_diff.json")