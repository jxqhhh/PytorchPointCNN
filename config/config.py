import torch
import time
import math


base_model='modelnet_x3_l4'


# Note: dataset to use is determined by base_model
dataset_available = ["ModelNet40","ScanNetSeg","ShapeNetParts"]
dataset_to_path = {
    "ModelNet40": {
        "train": "/home/jxq/repository/ModelNet40/train_files.txt",
        "test": "/home/jxq/repository/ModelNet40/test_files.txt"
    },
    "ScanNetSeg":{
        "train":2,
        "test":2
    },
    "ShapeNetParts":{
        "train": "/home/jxq/repository/ModelNet40/train_files.txt",
        "test": "/home/jxq/repository/ModelNet40/test_files.txt"
    }
}

base_model_available=['modelnet_x3_l4','scannet_x8_2048_k8_fps','shapenet_x8_2048_fps']

base_model_to_dataset={
    "modelnet_x3_l4":"ModelNet40",
    "shapenet_x8_2048_fps":"ShapeNetParts"
}

base_model_to_task={
    "modelnet_x3_l4": "cls",
    "shapenet_x8_2048_fps": "seg"
}

dataset=base_model_to_dataset[base_model]
task=base_model_to_task[base_model]

base_model_to_dataset_setting={
    "modelnet_x3_l4":{
        "sample_num":1024,
        "data_dim":6,
        "use_extra_features" : False,
        "with_X_transformation" : True,
        "with_normal_feature":True,
        "sorting_method" : None,
        "rotation_range" : [0,math.pi,0,'u'],
        "rotation_order":'rxyz',
        "scaling_range" : [0.1, 0.1, 0.1, 'g'],
        "jitter":0,
        "jitter_val": 0.0,
        "rotation_range_val": [0, 0, 0, 'u'],
        "scaling_range_val": [0, 0, 0, 'u']
    },
    "scannet_x8_2048_fps":{
        "sample_num":2048,
        "data_dim":3,
        "with_X_transformation" : True,
        "sorting_method" : None,
        "rotation_range" : [math.pi / 72, math.pi, math.pi / 72, 'u'],
        "rotation_order":'rxyz',
        "scaling_range" : [0.05, 0.05, 0.05, 'g'],
        "jitter":0,
        "jitter_val": 0.0,
        "rotation_range_val": [0, 0, 0, 'u'],
        "scaling_range_val": [0, 0, 0, 'u']
    },
    "shapenet_x8_2048_fps":{
        "sample_num": 2048,
        "data_dim": 3,
        "with_X_transformation": True,
        "sorting_method": None,
        "rotation_range": [0, 0, 0, 'u'],
        "rotation_order": 'rxyz',
        "scaling_range": [0.0, 0.0, 0.0, 'g'],
        "jitter": 0.001,
        "jitter_val": 0.0,
        "rotation_range_val": [0, 0, 0, 'u'],
        "scaling_range_val": [0, 0, 0, 'u']
    }
}
assert dataset in dataset_available
dataset_dir = dataset_to_path[dataset]
dataset_setting=base_model_to_dataset_setting[base_model]


# configuration file
project_name = "PyG_PointCNN"
description = ""


# device can be either "cuda" or "cpu"
num_workers = 4
available_gpus = ""
if available_gpus!="":
    device = torch.device("cuda")
    use_gpu=True
else:
    device=torch.device("cpu")
    use_gpu=False
print_freq = 10
daemon_mode = False
backup_code = True


time_stamp = '{}'.format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
short_time_stamp = "{}".format(time.strftime('%y%m%d_%H%M%S'))
dropout = True
dropmax = False
without_bn = True
first_kernel = 7
# pretrained_model = None
pretrained_model = ""
# TRAIN or VAL or VIS or TEST
process = "TRAIN"


class train:
    root_dir = dataset_to_path[dataset]
    if isinstance(root_dir, dict):
        root_dir = root_dir['train']
    resume = False
    resume_epoch = None

################## DO NOT CHANGE ############################


if base_model in base_model_available:
    instance_name = "{}{}".format(base_model, description)
else:
    instance_name = "{}{}_{}".format(base_model, description, first_kernel) + \
                    "{}".format('_dropout' if dropout else '')

comment = "{}_{}".format(instance_name, short_time_stamp)

if process != 'TRAIN':
    comment = pretrained_model
    daemon_mode = False
    backup_code = False

result_root = "../rst-{}".format(project_name)
result_sub_folder = "{}/{}".format(result_root, comment)
ckpt_file = "{}/ckpt.pth".format(result_sub_folder)


#############################################################


class validation:
    root_dir = dataset_to_path[dataset]
    if isinstance(root_dir, dict):
        root_dir = root_dir['test']
    pretrained_model = ckpt_file


class test:
    root_dir = dataset_to_path[dataset]
    if isinstance(root_dir, dict):
        root_dir = root_dir['test']
    pretrained_model = ckpt_file

if base_model=="modelnet_x3_l4":
    train.batch_size=128
    train.num_epochs=1024
    train.optimizer="ADAM"
    train.epsilon=1e-2
    train.learning_rate_base=0.01
    train.decay_steps=8000
    train.decay_rate=0.5
    train.learning_rate_min=1e-6
    train.weight_decay=1e-5
    validation.batch_size=128
    test.batch_size=128
    validation.step_val=5
elif base_model=="shapenet_x8_2048_fps":
    train.batch_size=16
    train.num_epochs=1024
    train.optimizer="ADAM"
    train.epsilon=1e-3
    train.learning_rate_base=0.005
    train.decay_steps= 20000
    train.decay_rate= 0.9
    train.learning_rate_min=0.00001
    train.weight_decay=0.0
    validation.batch_size=16
    validation.step_val=5
    test.batch_size=16
else:
    print("parameter not specified")
    raise NotImplementedError
