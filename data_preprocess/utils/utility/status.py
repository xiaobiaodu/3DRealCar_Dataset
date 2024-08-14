import os
import sys
import argparse
import collections
try:
    from utils.logging import logging
except:
    import logging

PIPELINE_STAGES = [
    'upload', 'dataset', 'segmentation', 
    'pcd_clean', 'pcd_standard', 'pcd_rescale', 'processed', 
    'trained', 'training']

class DatasetStatus:

    def __init__(self, default=False):
        for stage in PIPELINE_STAGES:
            setattr(self, stage, default)

class DatasetStatusCount(DatasetStatus):

    def __init__(self, default=0):
        super().__init__(default=default)
    
    def update(self, stage):
        setattr(self, stage, getattr(self, stage) + 1)

class DatasetInfo:

    def __init__(self, dataset_info, status_info):
        # dataset_info: "name:code"
        # status_info: "name:status"
        self._dataset_info = dataset_info
        self._inv_dataset_info = {v:k for k,v in self._dataset_info.items()}
        self._status_info = status_info
    
    def name2info(self, name):
        if name in self._dataset_info.keys() and name in self._status_info.keys():
            return self._dataset_info[name], self._status_info[name]
        else:
            logging.error(f'dataset {name} NOT found in dataset info!')
            raise KeyError

    def code2info(self, code):
        if code in self._inv_dataset_info.keys() and code in self._inv_dataset_info.keys():
            name = self._inv_dataset_info[code]
            return self.name2info(name)
        else:
            logging.error(f'code {code} NOT found in dataset info!')
            raise KeyError
        
    def name2code(self, name):
        if name in self._dataset_info.keys():
            return self._dataset_info[name]
        else:
            logging.error(f'dataset {name} NOT found in dataset info!')
            raise KeyError

    def keys(self):
        return sorted(self._dataset_info.keys(), key=lambda k: self.name2code(k))

def get_dataset_status(upload_dir, dataset_dir, dataset_name, exp_name, model_dir, processed_type):
    processed_dataset_dir = os.path.join(dataset_dir, dataset_name, processed_type + '_processed')
    status = DatasetStatus()
    status.upload=os.path.exists(os.path.join(upload_dir, dataset_name, '.upload'))
    status.dataset=os.path.exists(os.path.join(processed_dataset_dir, '.dataset'))
    status.segmentation=os.path.exists(os.path.join(processed_dataset_dir, '.segmentation'))
    status.pcd_clean=os.path.exists(os.path.join(processed_dataset_dir, 'pcd_clean/.processed'))
    status.pcd_standard=os.path.exists(os.path.join(processed_dataset_dir, 'pcd_standard/.processed'))
    status.pcd_rescale=os.path.exists(os.path.join(processed_dataset_dir, 'pcd_rescale/.processed'))
    status.processed=os.path.exists(os.path.join(processed_dataset_dir, '.processed'))
    status.trained=os.path.exists(os.path.join(model_dir, dataset_name, exp_name, '.trained'))
    status.training=os.path.exists(os.path.join(model_dir, dataset_name, exp_name, '.training'))
    return status

def get_dataset_info(dataset_info_fn, upload_dir):
    dataset_infos = {}
    status_infos = {}
    for line in open(dataset_info_fn).readlines():
        line = line.strip().split()
        if len(line) < 3:
            continue
        k = line[0]
        v = line[1]
        f = line[2]
        v = get_dataset_name(upload_dir=upload_dir, dataset_name=v)
        dataset_infos[v] = k
        status_infos[v] = f
    return DatasetInfo(
        dataset_info=dataset_infos, 
        status_info=status_infos)

def get_dataset_name(upload_dir, dataset_name):
    if os.path.exists(os.path.join(upload_dir, dataset_name + '_anonymous')):
        return dataset_name + '_anonymous'
    else:
        return dataset_name
