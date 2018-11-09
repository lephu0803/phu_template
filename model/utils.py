import json
import logging

class Params():
    ''' Use for load parameter from json file

    params = Params(json_path)
    params.learning_rate = 0.5
    '''

    def __init__(self,json_path):
        self.update(json_path)

    def save(self,json_path):
        '''Save param to json'''
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    def update(self, json_path):
        with open(json_path, 'w') as f:
            params = json.load(f)
            self.__dict__.update(params)
    
    @property
    def dict(self):
        return self.__dict__
def set_logger(log_path):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:

        file_handle = logging.FileHandler(log_path)
        file_handle.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handle)

        # logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(messeger)s'))
        logger.addHandler(stream_handler)

def save_dict_to_json(d,json_path):

    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d,f,indent=4)

            

