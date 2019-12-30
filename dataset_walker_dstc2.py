import os, json, re
import math
from CONSTANT import CUR_DIR,ONTO_DIR,TRAIN_DEV_DIR,TEST_DIR
class dataset_walker(object):
    def __init__(self,dataset,labels=False,dataroot=None):
        #parameters description:
        #   dataset: dstc2_train or dstc2_dev or destc2_test
        #   labels: a boolean variable for determine wheter this class load the lable.json or not, True--load, False--Not
        #   dataroot: the root path of data corresponding to content of dstc2_train.flist
        self.dataset                =   dataset
        self.dataset_session_lists  =   [ os.path.join(dataroot,dataset+'.flist') ]
        self.labels                 =   labels
        assert(dataroot!=None)
        self.dataroot               =   os.path.join(os.path.abspath(dataroot))
        # load dataset (list of calls)
        self.session_list           =   []
        for dataset_session_list in self.dataset_session_lists :
            with open(dataset_session_list) as f:
                for line in f:
                    line                =   line.strip()
                    if (line in self.session_list):
                        raise RuntimeError('Call appears twice: %s' % (line))
                    self.session_list.append(line)  
        
    def __iter__(self):
        for session_id in self.session_list:
            session_id_list = session_id.split('/')
            session_dirname = os.path.join(self.dataroot,*session_id_list)
            applog_filename = os.path.join(session_dirname,'log.json')
            if (self.labels):
                labels_filename = os.path.join(session_dirname,'label.json')
                if (not os.path.exists(labels_filename)):
                    raise RuntimeError('Cant score : cant open labels file %s' % (labels_filename))
            else:
                labels_filename = None
            call = Call(applog_filename,labels_filename)
            call.dirname = session_dirname
            yield call
    def __len__(self, ):
        return len(self.session_list)
    

class Call(object):
    def __init__(self,applog_filename,labels_filename):
        self.applog_filename = applog_filename
        self.labels_filename = labels_filename
        f = open(applog_filename)
        self.log = json.load(f)
        f.close()
        if (labels_filename != None):
            f = open(labels_filename)
            self.labels = json.load(f)
            f.close()
        else:
            self.labels = None

    def __iter__(self):
        if (self.labels_filename != None):
            for (log,labels) in zip(self.log['turns'],self.labels['turns']):
                yield (log,labels)
        else: 
            for log in self.log['turns']:
                yield (log,None)
                
    def __len__(self, ):
        return len(self.log['turns'])

replace_un_informable_slots = []
def label2vec(goal_label, method_label, request_label,ontology)->list:
    '''
    Parameters:
        1. goal
        2. method
        3. requests
    Return Value:
        1. a label vector {'food':_ , 'pricerange':_ , 'name':_ , 'area':_ ,'method':_ , 'request_label': 1*8 list }
    '''
    label_vec=dict() 
    for slot in ['food', 'pricerange', 'name', 'area']:
        if slot in goal_label and goal_label[slot] in ontology['informable'][slot]:
            label_vec[slot]=ontology['informable'][slot].index(goal_label[slot])
        else:
            # the max index is for the special value: "none"
            label_vec[slot]=len(ontology['informable'][slot])
    label_vec['method']=ontology['method'].index(method_label)
    reqVec = [0.0] * len(ontology['requestable'])
    for req in request_label:
        reqVec[ontology['requestable'].index(req)] = 1
    label_vec['request_label']=reqVec
    # print('goal_label,method_label,request_label : ',goal_label, method_label, request_label)
    # print('informable:', ontology['informable'].keys(), 'requestable:', ontology['requestable'], 'method: ', ontology['method'],sep='\n')
    # print(label_vec)
    return label_vec

def gen_turn_nbest_data(turn:dict, label:dict,ontology:dict)->dict:
    turnData = dict()
    # process user_input : exp scores
    user_input = turn["input"]["live"]["asr-hyps"]
    slu_sum=sum([math.exp(float(i['score'])) for i in user_input])
    for asr_pair in user_input:
        asr_pair['score'] = math.exp(float(asr_pair['score']))/slu_sum
    # process machine_output : replace un-informable value with tags
    machine_output = turn["output"]["dialog-acts"]
    # generate labelIdx
    label_vec = label2vec(label['goal-labels'], label['method-label'], label['requested-slots'],ontology)
    turnData["slu_input"] = user_input
    turnData["dialogue_act"] = machine_output# it is needed the vectorization
    turnData["label"] = label_vec
    return turnData

def extract_data(dataset:dataset_walker,ontology:dict)->list:
    data=[]
    for call in dataset:
        fileData = dict()
        fileData["session-id"] = call.log["session-id"]
        fileData["turns"] = list()
        # print ("session-id:", call.log["session-id"])
        for turn, label in call:
            turnData = gen_turn_nbest_data(turn, label, ontology)
            fileData["turns"].append(turnData)
        data.append(fileData)
    return data

if __name__=='__main__':
    # CUR_DIR,ONTO_DIR,TRAIN_DEV_DIR,TEST_DIR,
    with open(ONTO_DIR,'r') as fin:
        ontology=json.load(fin)
    print('ontology load finished!, ontology.keys(): ', ontology.keys())
    train_dataset   =   dataset_walker('dstc2_train',   labels=True,    dataroot=TRAIN_DEV_DIR)
    dev_dataset     =   dataset_walker('dstc2_dev',     labels=True,    dataroot=TRAIN_DEV_DIR)
    test_dataset    =   dataset_walker('dstc2_test',    labels=True,    dataroot=TEST_DIR)
    data            =   extract_data(train_dataset,     ontology)
    
