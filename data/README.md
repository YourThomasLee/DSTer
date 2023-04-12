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
- `./multiwoz/multiwoz.py`: this python script provides a common interface to get dataset in an unified way, so that you can use different versions/types of dataset as following: 
  
  ```python
  from datasets import load_dataset
  data = load_dataset("./data/multiwoz/", "conf22")
  ```

Now this repository supports

- MultiWOZ 2.1





## Preprocess

### multiwoz 2.1





## Label analysis

### multi-woz 2.1

Frequency statistics

| frequency | taxi-leave at | taxi-destination | taxi-departure | taxi-arrive by | restaurant-food | restaurant-price range | restaurant-name | restaurant-area | restaurant-book time | restaurant-book day | restaurant-book people | hotel-name | hotel-area | hotel-parking | hotel-price range | hotel-stars | hotel-internet | hotel-type | hotel-book stay | hotel-book day | hotel-book people | attraction-type | attraction-name | attraction-area | train-leave at | train-destination | train-day | train-arrive by | train-departure | train-book people |
| --------- | ------------- | ---------------- | -------------- | -------------- | --------------- | ---------------------- | --------------- | --------------- | -------------------- | ------------------- | ---------------------- | ---------- | ---------- | ------------- | ----------------- | ----------- | -------------- | ---------- | --------------- | -------------- | ----------------- | --------------- | --------------- | --------------- | -------------- | ----------------- | --------- | --------------- | --------------- | ----------------- |
| train     | 2216          | 4091             | 4090           | 1809           | 16945           | 15400                  | 10753           | 15573           | 9206                 | 9306                | 9354                   | 11017      | 12149      | 9009          | 10975             | 9900        | 8722           | 11456      | 9180            | 9151           | 9126              | 11096           | 7718            | 10587           | 8223           | 16168             | 15709     | 7982            | 15866           | 6360              |
| valid     | 273           | 481              | 483            | 212            | 2280            | 2110                   | 1399            | 2151            | 1222                 | 1231                | 1243                   | 1461       | 1549       | 1081          | 1322              | 1360        | 1107           | 1449       | 1209            | 1224           | 1208              | 1390            | 1031            | 1342            | 1126           | 2271              | 2224      | 1151            | 2255            | 880               |
| test      | 328           | 587              | 575            | 233            | 2078            | 1966                   | 1630            | 2033            | 1311                 | 1311                | 1333                   | 1403       | 1258       | 1065          | 1427              | 1188        | 1125           | 1293       | 1047            | 1048           | 1031              | 1660            | 1351            | 1662            | 1244           | 2689              | 2645      | 1591            | 2590            | 1141              |

Slot labels entropy

| entropy    | taxi-leave at | taxi-destination | taxi-departure | taxi-arrive by | restaurant-food | restaurant-price range | restaurant-name | restaurant-area | restaurant-book time | restaurant-book day | restaurant-book people | hotel-name | hotel-area | hotel-parking | hotel-price range | hotel-stars | hotel-internet | hotel-type | hotel-book stay | hotel-book day | hotel-book people | attraction-type | attraction-name | attraction-area | train-leave at | train-destination | train-day | train-arrive by | train-departure | train-book people |
| ---------- | ------------- | ---------------- | -------------- | -------------- | --------------- | ---------------------- | --------------- | --------------- | -------------------- | ------------------- | ---------------------- | ---------- | ---------- | ------------- | ----------------- | ----------- | -------------- | ---------- | --------------- | -------------- | ----------------- | --------------- | --------------- | --------------- | -------------- | ----------------- | --------- | --------------- | --------------- | ----------------- |
| train      | 0.08          | 5.29             | 5.08           | 0.05           | 3.08            | 1.21                   | 4.71            | 1.47            | 0.02                 | 1.95                | 2.08                   | 3.41       | 1.72       | 0.49          | 1.25              | 1.17        | 0.37           | 0.79       | 1.57            | 1.95           | 2.08              | 2.29            | 4.35            | 1.43            | 0.16           | 1.94              | 1.94      | 0.12            | 2.0             | 2.04              |
| validation | 0.06          | 4.62             | 4.28           | 0.13           | 3.07            | 1.2                    | 4.49            | 1.53            | 0                    | 1.9                 | 2.06                   | 3.37       | 1.72       | 0.38          | 1.27              | 1.11        | 0.41           | 0.74       | 1.62            | 1.94           | 2.07              | 2.34            | 4.05            | 1.50            | 0.11           | 1.79              | 1.9       | 0.09            | 2.09            | 1.98              |
| test       | 0.20          | 4.72             | 4.6            | 0              | 2.76            | 1.14                   | 4.53            | 1.29            | 0.03                 | 1.93                | 2.07                   | 3.37       | 1.7        | 0.46          | 1.31              | 1.35        | 0.34           | 0.73       | 1.54            | 1.92           | 2.06              | 2.25            | 4.16            | 1.43            | 0.18           | 1.92              | 1.94      | 0.06            | 2.02            | 2.02              |

cross entropy

|              | restaurant-price range | restaurant-area | restaurant-book day | restaurant-book people | hotel-area | hotel-parking | hotel-price range | hotel-stars | hotel-internet | hotel-type | hotel-book stay | hotel-book day | hotel-book people | attraction-type | attraction-area | train-destination | train-day | train-departure | train-book people |
| ------------ | ---------------------- | --------------- | ------------------- | ---------------------- | ---------- | ------------- | ----------------- | ----------- | -------------- | ---------- | --------------- | -------------- | ----------------- | --------------- | --------------- | ----------------- | --------- | --------------- | ----------------- |
| valid\|train | 1.226                  | 1.54            | 1.952               | 2.076                  | 1.737      | 0.418         | 1.282             | 1.122       | 0.426          | 0.748      | 1.636           | 1.971          | 2.077             | 2.407           | 1.512           | 1.853             | 1.950     | 2.149           | 2.021             |
| test\|train  | 1.162                  | 1.318           | 1.949               | 2.080                  | 1.729      | 0.534         | 1.331             | 1.404       | 0.35           | 0.736      | 1.572           | 1.977          | 2.080             | 2.330           | 1.450           | 1.991             | 1.95      | 2.081           | 2.059             |
| test\|valid  | 1.159                  | 1.335           | 1.954               | 2.112                  | 1.741      | 0.50          | 1.326             | 1.41        | 0.376          | 0.732      | 1.581           | 1.975          | 2.105             | 2.363           | 1.461           | 2.03              | 1.954     | 2.085           | 2.061             |

