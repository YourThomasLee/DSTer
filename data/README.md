# Dialogue state tracking data collection and preprocess

The preprocess of dialogue state tracking models for get model's input from original data is so different that researchers have to pay so much attention to the preprocess details. The objective of files in this directory is try to provide an unified way to fetch and use dialogue data.

This directory stores the public datasets used to evaluate the models in tracking dialogue states which represents the dialogue state of user in utterance for specific target like booking restaurant, searching for attraction area, and looking for a great movie. 

function description:

- `prepare_env.sh`: fetch datasets from network space (you should make sure that your configuration of git account is correct, because this script depends it) 

- `dialogue_data.py`: this python script provides a common interface to get the data in the same way, so that you can use different versions/types of dataset like:
  
  ```python
  from data import dialogue_data
  name = "MultiWOZ"
  version = "2.0"
  data = dialogue_data(name, version)
  ```

schema design(TODO):

class dialogue_data

| attributes | types | description |
| ---------- | ----- | ----------- |
|            |       |             |
|            |       |             |
|            |       |             |

If you have any questions in using this file, please propose an issue. I will respond it as soon as possible!
