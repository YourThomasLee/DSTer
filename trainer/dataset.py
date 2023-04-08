#-*- encoding: utf-8 -*-

from turtle import ontimer
from collections import defaultdict as ddict
from torch.utils.data import Dataset
from copy import deepcopy as dc
import nltk
import re
import joblib

# user package
from utils import util
from trainer import vocab

TYPOS_CORRECT = 0 # count how many times of correcting typo errors in program
GENERAL_TYPO = {
    # type
    "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports", 
    "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall", 
    "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture", 
    "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
    # area
    "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east", 
    "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre", 
    "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north", 
    "centre of town":"centre", "cb30aq": "none",
    # price
    "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate", "moderately": "moderate",
    # day
    "next friday":"friday", "monda": "monday", "thur": "thursday",
    # parking
    "free parking":"free",
    # internet
    "free internet":"yes",
    # star
    "4 star":"4", "4 stars":"4", "0 star rarting":"none",
    # others 
    "dont care": "dontcare", "y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none", "not given": "none",
    '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none", "dont care": "dontcare", "don't care": "dontcare", "doesn't care": "dontcare",
    "w": "none",
    # LBZ adding
    #restaurant name
    "not(hamilton lodge)": "not hamilton lodge",
    "golden house                            golden house": "golen house",
    # taxi-leave at
    "0700": "07:00", "300": "03:00", "1615": "16:15", "20.00": "20:00", "16.30": "16:30", "21:4": "21:04", "1530": "15:30", "1145": "11:45", "1545": "15:45", "1745": "17:45", "1830": "18:30",
    "`1": "1",
    "02:45.": "02:45",
    "5:45": "05:45",
    "1:15": "01:15",
    "3:00": "03:00",
    "4:15": "04:15",
    "8:30": "08:30",
    "3:45": "03:45",
    "8:15": "08:15",
    "9:30": "09:30",
    "3:15": "03:15",
    "9:00": "09:00",
    "1:00": "01:00",
    "5:15": "05:15",
    "4:45": "04:45",
    "21:04": "21:04",
    "9:15": "09:15",
    "6:00": "06:00",
    "1700": "17:00",
    "5:30": "05:30",
    "1730": "17:30",
    "9:45": "09:45",
    "2:00": "02:00",
    "1:00": "01:00",
    "9:15": "09:15",
    "8:45": "08:45",
    "8:30": "08:30",
    "1030": "10:30",
    "7:54": "07:54",
    "2:30": "02:30",
    "9:30": "09:30",
    "13.29": "13:29",
    "1700": "17:00",
    "8:00": "08:00",
    "6:55": "06:55",
    "15.45": "15:45",
    "8:30": "08:30",
    "9:30": "09:30",
    "15.32": "15:32",
    "11.45": "11:45",
    "after 5:45 pm": "17:45",
    "09;45": "09:45",
    "11.24": "11:24",
    "11.45": "11:45",
    "18.15": "18:15",
    # hotel book people
    "six": "6",
    "3.": "3",
    }

class MultiWOZ:
    """
    MultiWOZ(8 domains) data loading using BaseDataLoader
    Description: Multi-Domain Wizard-of-Oz dataset (MultiWOZ), a fully-labeled collection of human-human written conversations spanning over multiple domains and topics. At a size of 10k dialogues, it is at least one order of magnitude larger than all previous annotated task-oriented corpora.
        - There are 3,406 single-domain dialogues that include booking if the domain allows for that and 7,032 multi-domain dialogues consisting of at least 2 up to 5 domains. To enforce reproducibility of results, the corpus was randomly split into a train, test and development set. The test and development sets contain 1k examples each. Even though all dialogues are coherent, some of them were not finished in terms of task description. Therefore, the validation and test sets only contain fully successful dialogues thus enabling a fair comparison of models. There are no dialogues from hospital and police domains in validation and testing sets.
        - 2.0为多领域的对话数据集, 人-人对话数据集, 对话和标注都由人完成, 存在延迟标注(标注轮次较晚), 多重标注(一个值被标注在多个槽位中), 错误标注(值和槽并不匹配), 语法错误, 标注错漏.
        - 2.1: 观察到2.0版本存在对话状态和对话话语中存在错误标注/噪声, 另外统一了已有的前人对数据集的扩充(例如对话动作扩充等等); 
            1. 对于话语上的自由性, 许多值并没有和ontology中的设定值完全一样, 对于槽值进行了规整;
            2. 使用Levenshtein distance小于3的值作为候选槽值, 然后构建完槽值后对句子进行匹配替换, 生成新的对话语句
            3. 对每个槽增加了简短的说明, 考虑支持小样本/迁移(冷启动)等状况下模型设计和评估
            4. 对话动作的标注(主要就是合并了前人工作的成果)
        - 2.2: 进一步修正17.3%的对话的状态; 修订了ontology去除大量非合法的值. 另外对一些槽的值进行话语上的区间标注
            1. 修复以下错误: 过早标注, 数据库标注(话语中未被提及, 只是包含在数据库中), 书写错误, 隐含的时间处理; 
    woz data url: https://github.com/budzianowski/multiwoz.git
    dialogue state tracking: 
        formulation: $State_n = f(State_{n-1}, history_{n-1}, utterance_{n}, action_{n-1})$
    task description: 
        数据: 领域, 槽位, 领域候选数据(正排数据如何组织, 依赖关系如何建模)
        机制设计: 对话状态系统机制的设计(数据交融, 模型设计和推理, 线上如何使用, 在线如何训练)
    此处只考虑文本对话数据: 对话, 状态
    # dial_id领域: MUL, PMUL为多领域, SNG, SSNG, WOZ为单领域
    # 对话中只有系统话语包含了对话动作的标注, 用户话语无动作标注
    # police/hospital领域包含很少的样本, 因为这两个领域的标注过于简单, 同时验证集和测试集没有这两个领域的样本

    
    从ontology diff 上看， 2.1版本的标签数据基本能对上了，但2.0版本的标签还有很多的噪声，考虑跑完模型再清洗一遍标签
    """
    def __init__(self, 
        original_data_dir,
        data_dir,
        state_domain_card):
        # self.state_vocab = vocab.Vocab()
        self.text_vocab = vocab.Vocab()
        self.text_vocab.add_word("system")
        self.text_vocab.add_word("user")
        self.original_data_dir = original_data_dir
        self.data_dir =  data_dir

        if "2.0" in original_data_dir:
            version = 20
        elif "2.1" in original_data_dir:
            version = 21
            
        elif "2.2" in original_data_dir:
            version = 22

        onto_file_name = self.original_data_dir + "/domain_data/" + "ontology.json" #
        self.onto_given = self.process_ontology(onto_file_name)
        self.onto_generated = dict()

        train_file_name =  self.data_dir + "/train%s.json" % version
        vali_file_name = self.data_dir + "/vali%s.json" % version
        test_file_name = self.data_dir + "/test%s.json" % version
        if util.is_exist(train_file_name) and util.is_exist(vali_file_name) and util.is_exist(test_file_name):
            train_data = util.read_json(train_file_name)
            vali_data = util.read_json(vali_file_name)
            test_data = util.read_json(test_file_name)
            self.onto_generated = util.read_json(self.data_dir + "/domain_data/" + "ontology_generated.json")
            self.text_vocab = joblib.load(self.data_dir + "/" + "vocab.joblib")
        else:
            train_data, vali_data, test_data = self.read_multiwoz()
            test_data = sorted(test_data, key = lambda x: x[1])
            util.write_json(train_data,train_file_name)
            util.write_json(vali_data, vali_file_name)
            util.write_json(test_data, test_file_name)
            util.ensure_dir(self.data_dir + "/domain_data/")
            util.write_json(self.onto_generated, self.data_dir + "/domain_data/" + "ontology_generated.json")
            joblib.dump(self.text_vocab, self.data_dir + "/" + "vocab.joblib")
        ontology = self.onto_generated
        self.train_data = Datapool(train_data, self.text_vocab, ontology, state_domain_card)
        self.vali_data = Datapool(vali_data, self.text_vocab, ontology, state_domain_card)
        self.test_data = Datapool(test_data, self.text_vocab, ontology, state_domain_card)
        
    def process_ontology(self, onto_file_name):
        """
        process given ontology
        """
        if util.is_exist(self.data_dir + "/domain_data/" + "ontology_given.json"):
            return util.read_json(self.data_dir + "/domain_data/" + "ontology_given.json")
        ontology = dict()
        normalize_slot = lambda x: "".join([i if not "A"<=i<="Z" else " " + i.lower() for i in x])
        for k,v in util.read_json(onto_file_name).items():
            if "2.0" in self.original_data_dir:
                domain, slot = k.split("-")
            elif "2.1" in self.original_data_dir:
                domain, sclass, slot = k.split("-")
                if "semi" not in sclass:
                    slot = sclass + " " + slot
                if slot == "pricerange":
                    slot = "price range"
            slot = normalize_slot(slot)
            domain_slot = domain + "-" + slot
            ontology[domain_slot] = v
        util.ensure_dir(self.data_dir + "/domain_data/")
        util.write_json(ontology, self.data_dir + "/domain_data/" + "ontology_given.json")
        return ontology
    
    def fix_general_label_error(self, domain, slot, value):
        """
        process label value
        """
        if len(value) == 0:
            return ""
        
        if value in GENERAL_TYPO.keys():
            # general typo
            global TYPOS_CORRECT
            TYPOS_CORRECT += 1
            value = GENERAL_TYPO[value]
        # miss match slot and value
        if  domain == "hotel" and (slot == "type" and value in ["nigh", "moderate -ly priced", "bed and breakfast", "centre", "venetian", "intern", "a cheap -er hotel"] or \
            slot == "internet" and value == "4" or \
            slot == "price range" and value == "2") or \
            domain == "attraction" and slot == "type" and value in ["gastropub", "la raza", "galleria", "gallery", "science", "m"] or \
            "area" in slot and value in ["moderate"] or \
            "day" in slot and value == "t":
            value = "none"
        elif domain == "hotel" and slot == "type" and value in ["hotel with free parking and free wifi", "4", "3 star hotel"]:
            value = "hotel"
        elif domain == "hotel" and slot == "star" and value == "3 star hotel":
            value = "3"
        elif "area" in slot:
            if value == "no": value = "north"
            elif value == "we": value = "west"
            elif value == "cent": value = "centre"
        elif "day" in slot:
            if value == "we": value = "wednesday"
            elif value == "no": value = "none"
        elif "price" in slot and value == "ch":
            value = "cheap"
        elif "internet" in slot and value == "free":
            value = "yes"
        
        # some out-of-define classification slot values
        if  domain == "restaurant" and slot == "area" and value in ["stansted airport", "cambridge", "silver street"] or \
            domain == "attraction" and slot == "area" and value in ["norwich", "ely", "museum", "same area as hotel"]:
            value = "none"
        if domain == "hotel" and slot == 'name' and value in ["no", "yes"]:
            value = "none"
        if domain == "restaurant" and slot == 'name' and value in ["no", "yes"]:
            value = "none"
        # remove nonsense
        if "2.0" in self.original_data_dir and "hotel" not in domain and "book" not in slot and len(value) <= 2:
            value = "none"
        return value

    def normalize_state_value(self, domain, slot, value, remove_none = True):
        if "|" in value:
            # we do not fix multivalue label here
            value = "|".join([self.fix_general_label_error(domain, slot, item.strip()) for item in value.split("|")]).strip()
        else:
            # fix some general errors
            value = self.fix_general_label_error(domain, slot, value)
        if remove_none and value == "none":
            # remove none value
            value = ""
        return value

    def extract_state(self, metadata):
        # 从原始数据中抽取出对应的对话状态
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
        normalize_slot = lambda x: "".join([i if not "A"<=i<="Z" else " " + i.lower() for i in x])
        for domain, item in metadata.items():
            for slot, value in item["semi"].items():
                if slot == "pricerange":
                    slot = "price range"
                slot = normalize_slot(slot)
                domain_slot = domain + "-" + slot
                value = self.normalize_state_value(domain, slot, value.lower().strip())
                belief_state[domain_slot] = value
                if domain_slot not in self.onto_generated:
                    self.onto_generated[domain_slot] = ddict(int)
                if "|" in value:
                    for i in value.split("|"):
                        if len(i.strip()) == 0: continue
                        self.onto_generated[domain_slot][i] = self.onto_generated[domain_slot][i] + 1
                elif len(value.strip()) > 0:
                    self.onto_generated[domain_slot][value] = self.onto_generated[domain_slot][value] + 1
            for slot, value in item["book"].items():
                # fix "book" slot name
                if slot == "booked":
                    continue
                slot_new = normalize_slot("book " + slot)
                domain_slot = domain + "-" + slot_new
                value = self.normalize_state_value(domain, slot_new, value.lower().strip())
                belief_state[domain_slot] = value
                if domain_slot not in self.onto_generated:
                    self.onto_generated[domain_slot] = ddict(int)
                if "|" in value:
                    for i in value.split("|"):
                        if len(i.strip()) == 0: continue
                        self.onto_generated[domain_slot][i] = self.onto_generated[domain_slot][i] + 1
                elif len(value.strip()) > 0:
                    self.onto_generated[domain_slot][value] = self.onto_generated[domain_slot][value] + 1
        belief_state = {k: v for k,v in belief_state.items() if len(v) > 0}
        return belief_state

    def dialog2turns(self, dial_id, dialog):
        """
        extract a turns list from a dialog
        """
        # list schema: 
        #   - dial_id: dialogue ID
        #   - turn_id: turn index
        #   - role: user or system
        #   - last_state: belief state previous turn 
        #   - dialogue history: dialogue utterance string
        #   - utterance: dialogue text
        #   - now_state: current dialogue belief state
        turn_list = list()
        history_text = list()
        pre_belief_state = dict()
        belief_state = dict()
        for turn_id, turn in enumerate(dialog["log"]):
            role = "user" if turn_id % 2 == 0  else "system"
            if role == "system":
                # system turn: update belief
                belief_state = self.extract_state(turn['metadata'])
            utterance = util.restore_common_abbr(turn['text'].lower().strip())
            self.text_vocab.add_sentence(utterance)
            if role == "system":
                turn_list.append((dial_id, turn_id, role, dc(pre_belief_state), dc(history_text), dc(belief_state)))
                pre_belief_state = dc(belief_state)
            history_text.append(utterance)
        return turn_list
    
    def read_multiwoz(self):
        """
        import multiwoz 20,21,22 data
        """
        if "2.2" in self.original_data_dir:
            # TODO: read multiWOZ 2.2 data
            return 0
        data_file = self.original_data_dir + "/" + "data.json"
        val_list_file = self.original_data_dir + "/" + "valListFile.json"
        test_list_file = self.original_data_dir + "/" + "testListFile.json"
        
        val_list = [i for i in util.read_file(val_list_file)]
        test_list = [i for i in util.read_file(test_list_file)]
        train_data, val_data, test_data = list(), list(), list() 

        data = util.read_json(data_file)
        # print("type: %s length: %d" % (type(data), len(data)))
        for dial_id, dialog in data.items():
            # transformation from a dialogue to turns list
            turns_list = self.dialog2turns(dial_id, dialog)
            # 验证集
            if dial_id in val_list:
                val_data += turns_list
                continue
            # 测试集
            elif dial_id in test_list:
                test_data += turns_list
                continue
            # 训练集
            train_data += turns_list
        check_none = lambda x: (x in self.onto_generated.keys() and len(self.onto_generated[x]) <= 1)
        if check_none('bus-arrive by'):
            self.onto_generated.pop('bus-arrive by')
        if check_none('bus-book people'):
            self.onto_generated.pop('bus-book people')
        if check_none('train-book ticket'):
            self.onto_generated.pop('train-book ticket')
        print("the result of generated ontology - original ontology: ", list(set(self.onto_generated.keys()) -  set(self.onto_given.keys())))
        # 2.0: {'bus-arrive by', 'train-book ticket', 'bus-book people'}
        # 2.1: {'bus-book people', 'bus-arrive by', 'train-book ticket'}
        print("the result of original ontology - generated ontology: ", list(set(self.onto_given.keys()) -  set(self.onto_generated.keys())))
        print("the quantity of label values:", [(domain_slot, len(values)) for domain_slot, values in self.onto_generated.items()])
        print("text vocabulary words quantity: ", self.text_vocab.n_words)
        # 2.0: set()
        # 2.1: set()
        return train_data, val_data, test_data


class Datapool(Dataset):
    """
    pass
    """
    def __init__(self, sample_list, vocab, ontology, state_domain_card, mode = "train"):
        self.sample_list = sample_list
        self.text_vocab = vocab
        self.ontology = ontology
        self.mode = mode
        self.state_domain_card = state_domain_card

    def state_transform(self, state):
        numeric_state = []
        # 排个序
        domain_slots = sorted([k for k in self.ontology.keys()])
        values_dict = {k: sorted(list(self.ontology[k].keys())) for k in domain_slots}
        for domain_slot in domain_slots:
            values = values_dict[domain_slot]
            n = max(self.state_domain_card) + 1
            temp = [0.0 for i in range(n)]
            if domain_slot in state and "none" not in state[domain_slot]:
                for v in state[domain_slot].split("|"):
                    if v not in values:
                        import pdb
                        pdb.set_trace()
                    # assert(v in values)
                    temp[values.index(v.strip()) + 1] = 1.0
            else:
                temp[0] = 1.0 # none/unknow
            numeric_state.append(temp)
        assert(len(numeric_state) == 35) #35 * 256
        return numeric_state

    def text_transform(self, text_list):
        pattern = r'''(?x) # set flag to allow verbose regexps
            (?:[A-Z]\.)+[A-Z]   # abbreviations, e.g. U.S.A
            | \w+(?:-\w+)* # words with optional internal hyphens
            | \$?\d+(?:\.\d+)?\%? # currency and percentages, e.g. $12.40, 82%
            | \.\.\.      # ellipsis
            |(?:[.,;"'?():-_`!])  # these are separate tokens; includes ], [
        '''
        token_list = []
        role_list = []
        for idx, val in enumerate(text_list):
            tokens = nltk.regexp_tokenize(val,pattern)
            token_list += tokens
            role_list += [ "system" if idx & 1 else "user" ] * len(tokens)
        assert(len(token_list) == len(role_list))
        numeric_text = [self.text_vocab.get_index(i) for i in token_list]
        numeric_role = [self.text_vocab.get_index(i) for i in role_list]
        return numeric_role, numeric_text

    def train_transform(self, sample):
        previous_state = self.state_transform(sample[3])
        current_state = self.state_transform(sample[5])
        utterance_list = sample[4]
        history_text = utterance_list[:-2]
        current_text = utterance_list[-2:]
        history_roles, history_text = self.text_transform(history_text)
        current_roles, current_text = self.text_transform(current_text)
        return {
            "dial_id": sample[0],
            "turn_id": sample[1],
            "previous_state": previous_state,
            "current_state": current_state,
            "history_text": history_text,
            "history_roles": history_roles,
            "current_text": current_text,
            "current_roles": current_roles,
        }

    def val_transform(self):
        pass

    def __getitem__(self, index):
        # list schema: 
        #   - dial_id: dialogue ID
        #   - turn_id: turn index
        #   - role: user or system
        #   - last_state: belief state previous turn 
        #   - dialogue history: dialogue utterance string
        #   - now_state: current dialogue belief state
        sample = self.sample_list[index]
        if self.mode == "train":
            sample = self.train_transform(sample)
        elif self.mode == "valid":
            sample = self.valid_transform(sample)
        return sample

    def __len__(self):
        return len(self.sample_list)  