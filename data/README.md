# Dialogue state tracking data collection and preprocess

## 1. The objective of file in this directory

The process of  existing method to get model's input from original data is so different that researchers have to pay so much attention to the pre-process details. The objective of files in this directory is try to provide an unified way to fetch and use dialogue data.

This directory stores the public datasets used to evaluate the models in tracking dialogue states which represents the dialogue state of user in utterance for specific target like booking restaurant, searching for attraction area, and looking for a great movie. 

## 2. Access provided to datasets

Here I listed public datasets that are designed for dialogue state tracking evaluation

-  [DSTC2 & DSTC3](https://github.com/matthen/dstc): 
- [Sim-M, Sim-R, Sim-Gen](https://github.com/google-research-datasets/simulated-dialogue):
- [MultiWOZ 1.0 & 2.0 & 2.1 & 2.2](https://github.com/budzianowski/multiwoz/tree/master) & [2.3](https://github.com/lexmen318/MultiWOZ-coref) & [2.4](https://github.com/smartyfh/MultiWOZ2.4): 
- [SGD](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue):
- [CrossWOZ](https://github.com/thu-coai/CrossWOZ):
- [RiSAWOZ](https://github.com/terryqj0107/RiSAWOZ)：

and there are already data loaders for MultiWOZ:  
- [multi-woz2.2 loader](https://huggingface.co/datasets/multi_woz_v22)
- [multiwoz_dst](https://huggingface.co/datasets/adamlin/multiwoz_dst)
- [multiwoz_all_versions](https://huggingface.co/datasets/pietrolesci/multiwoz_all_versions/tree/main)


Leader board: [multi-woz 2.0](https://paperswithcode.com/sota/multi-domain-dialogue-state-tracking-on); [multi-woz 2.1](https://paperswithcode.com/sota/multi-domain-dialogue-state-tracking-on-1); [multi-woz 2.2](https://paperswithcode.com/sota/multi-domain-dialogue-state-tracking-on-2)



file description:

- `prepare_env.sh`: fetch datasets from network space (you should make sure that your configuration of git account is correct, because this script depends it. Sometimes this file fails to fetch files from github for the network situation. Taking it into consideration, I recommend you run it in debug mode `sh -x prepare_env.sh`) 
- `./multiwoz/multiwoz.py`: this python script provides a common interface to get MultiWOZ dataset in an unified way, so that you can use different versions/types of dataset as following: 
  
  ```python
  from datasets import load_dataset
  data = dialogue_data("./data/multiwoz/", "conf22")
  ```





schema design(TODO):

class dialogue_data

| attributes | types | description |
| ---------- | ----- | ----------- |
|            |       |             |
|            |       |             |
|            |       |             |

If you have any questions in using this file, please propose an issue. I will respond it as soon as possible!
