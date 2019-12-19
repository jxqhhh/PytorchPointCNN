import torch.nn as nn
from .XConv import XConv,View
import math
import torch
import sys
sys.path.append("..")
import config
#from torch_geometric.nn import fps

class PointCNN(nn.Module):

    sampling_method = ['random', 'fps']  # "random": random sampling; "fps": farthest point sampling

    def __init__(self, setting:dict):
        super(PointCNN, self).__init__()
        self.setting=setting
        self.fts_is_None=self.setting['fts_is_None']
        self.xconv_params=self.setting['xconv_params']
        if not self.fts_is_None:
            self.features_hd_generator=nn.Sequential(nn.Linear(self.setting['data_dim']-3,\
                self.setting['xconv_params'][0]['C']),nn.ELU(),View((-1,self.setting['xconv_params'][0]['C'])),\
                nn.BatchNorm1d(num_features=self.setting["xconv_params"][0]['C']),\
                View((-1,setting["sample_num"],self.setting['xconv_params'][0]['C'])))
        self.net = nn.Sequential()
        self.with_X_transformation=setting['with_X_transformation']
        self.fc_params=setting['fc_params']
        self.links=[]

        self.sampling = setting['sampling']
        prev_P=setting["sample_num"]

        self.xconv_output_dim=[]
        param_dict={"xconv_output_dim":self.xconv_output_dim,"xconv_params":self.setting["xconv_params"]}
        if "xdconv_params" in setting:
            param_dict["xdconv_params"]=self.setting["xdconv_params"]
        if self.fts_is_None:
            self.xconv_output_dim.append(0)
        else:
            self.xconv_output_dim.append(self.setting["data_dim"]-3)

        for layer_idx, layer_param in enumerate(self.setting['xconv_params']):
            K = layer_param['K']  # K：在inverse density sampling时，需要根据每个点到与其最近的K个点的平均距离估算该点的概率密度，从而实现inverse denstiy sampling；这个K也是XConv操作里的K
            D = layer_param['D']  # D：Dilation
            if layer_param['P']>0:
                P = layer_param['P']  # P：表示下一层要选用的样本点的数量
            elif layer_param["P"]==-1:
                P=prev_P
            else:
                print("P should either be positive integer or eqaul to -1!")
                #exit()()
            prev_P=P
            C = layer_param['C']  
            with_global = (setting["with_global"] and layer_idx == len(self.xconv_params) - 1)
            self.links.append(layer_param['links'])
            if layer_idx == 0:
                C_pts_fts = C // 2 if self.fts_is_None else C // 4 # //表示除后向下取整数
                depth_multiplier = 4
            else:
                C_prev = self.xconv_params[layer_idx - 1]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = math.ceil(C / C_prev)
            last_layer_flag = (not "xdconv_params" in setting) and (layer_idx == (len(self.setting['xconv_params'])-1) )
            self.net.add_module("layer{}".format(layer_idx),module=XConv(setting=param_dict,idx=layer_idx,K=K,dilation=D,
                P=P,C=C,C_pts_fts=C_pts_fts,depth_multiplier=depth_multiplier,with_X_transfomation=self.with_X_transformation, with_global=with_global,
                sorting_method=None).to(config.device))
            if last_layer_flag and with_global:
                self.xconv_output_dim.append(C+C//4)
            else:
                self.xconv_output_dim.append(C)


        if "xdconv_params" in setting:
            for layer_idx, layer_param in enumerate(setting["xdconv_params"]):
                K = layer_param['K']
                D = layer_param['D']
                pts_layer_idx = layer_param['pts_layer_idx']
                qrs_layer_idx = layer_param['qrs_layer_idx']
                P = setting["xconv_params"][qrs_layer_idx]['P']
                C = setting["xconv_params"][qrs_layer_idx]['C']
                C_prev = setting["xconv_params"][pts_layer_idx]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = 1
                with_global=False
                last_layer_flag = (layer_idx == (len(self.setting['xdconv_params']) - 1))
                self.net.add_module("xdconv_layer{}".format(layer_idx),module=XConv(setting=param_dict,idx=len(self.setting['xconv_params'])+layer_idx,
                    K=K,dilation=D,P=P,C=C,C_pts_fts=C_pts_fts,depth_multiplier=depth_multiplier,with_X_transfomation=self.with_X_transformation,
                    with_global=with_global,sorting_method=None).to(config.device))

                layer_name = "xdconv_dense{}".format(layer_idx)
                self.net.add_module(name=layer_name, module=nn.Sequential(nn.Linear(self.xconv_output_dim[qrs_layer_idx+1]+C, C).to(config.device),
                                                                          nn.BatchNorm1d(num_features=C).to(config.device)).to(config.device))
                self.xconv_output_dim.append(C)


        for layer_idx, layer_param in enumerate(self.fc_params):
            C = layer_param['C']
            if layer_idx>0:
                if "xdconv_params" in setting:
                    P=setting["xconv_params"][setting["xdconv_params"][-1]["qrs_layer_idx"]]["P"]
                else:
                    P=setting["xconv_params"][-1]["P"]
                self.net.add_module(name="dense{}".format(layer_idx),module=nn.Sequential(nn.Linear(prev_C,C),View((-1,C)),nn.BatchNorm1d(num_features=C),View((-1,P,C))).to(config.device))
            else:
                if "xdconv_params" in setting:
                    prev_C=setting["xconv_params"][setting["xdconv_params"][-1]["qrs_layer_idx"]]["C"]
                    P=setting["xconv_params"][setting["xdconv_params"][-1]["qrs_layer_idx"]]["P"]
                else:
                    prev_C=self.setting["xconv_params"][-1]['C']
                    P=setting["xconv_params"][-1]["P"]
                    if self.setting["with_global"]:
                        prev_C += (prev_C//4)
                self.net.add_module(name="dense{}".format(layer_idx),module=nn.Sequential(nn.Linear(prev_C,C),View((-1,C)),nn.BatchNorm1d(num_features=C),View((-1,P,C))).to(config.device))
            self.net.add_module(name="fc_dropout_{}".format(layer_idx),module=nn.Dropout(layer_param['dropout_rate']).to(config.device))
            prev_C=layer_param['C']

        self.net.add_module(name="logits",module=nn.Linear(self.fc_params[-1]["C"], self.setting["num_classes"], bias=True).to(config.device))


    def forward(self, input):
        pts=input[:,:,:3]
        if input.shape[-1]>3:
            fts=input[:,:,3:]
        else:
            fts=None
        layer_pts=[pts]
        if self.fts_is_None:
            layer_fts=[fts]
        else:
            layer_fts = [self.features_hd_generator.forward(fts)]
        for layer_idx,layer_param in enumerate(self.xconv_params):
            P = layer_param['P']
            if P == -1 or (layer_idx > 0 and P == self.setting["xconv_params"][layer_idx - 1]['P']) or (layer_idx == 0 and P== self.setting["sample_num"]):
                qrs = layer_pts[-1]
            else:
                if self.sampling == 'random':
                    qrs = pts[:, :P, :]
                    qrs_idx = [i for i in range(P)]
                elif self.sampling == 'fps':
                    batch=torch.arange(0,pts.shape[0],1).view(-1,1).repeat(1,pts.shape[1]).view(-1).to(config.device)
                    qrs_idx=fps(pts.view(-1,3),batch,ratio=P/pts.shape[1])
                    qrs=pts.view(-1,3)[qrs_idx].view(pts.shape[0],-1,3)
                else:
                    print("Unknown sampling method!")
                    raise NotImplementedError
            pts=layer_pts[-1]
            fts=layer_fts[-1]
            #XConv!
            fts=self.net._modules["layer{}".format(layer_idx)].forward(pts=pts,fts=fts,qrs=qrs)
            layer_pts.append(qrs)
            fts_list = []
            if self.links[layer_idx]:
                if self.sampling=='random':
                    for link in self.links[layer_idx]:
                        fts_from_link = layer_fts[link]
                        if fts_from_link is not None:
                            fts_slice = fts_from_link[:,qrs_idx,:]
                            fts_list.append(fts_slice)
                elif self.sampling=='fps':
                    raise NotImplementedError
            if fts_list:
                fts_list.append(fts)
                layer_fts.append(torch.cat(fts_list,dim=-1))
            else:
                layer_fts.append(fts)



        if 'xdconv_params' in self.setting:
            for layer_idx, layer_param in enumerate(self.setting["xdconv_params"]):
                pts_layer_idx = layer_param['pts_layer_idx']
                qrs_layer_idx = layer_param['qrs_layer_idx']
                C = self.setting["xconv_params"][qrs_layer_idx]['C']
                pts = layer_pts[pts_layer_idx + 1]
                fts = layer_fts[pts_layer_idx + 1] if layer_idx == 0 else layer_fts[-1]
                qrs = layer_pts[qrs_layer_idx + 1]
                fts_qrs = layer_fts[qrs_layer_idx + 1]
                fts_xdconv = self.net._modules["xdconv_layer{}".format(layer_idx)].forward(pts=pts,qrs=qrs,fts=fts)
                fts_concat = torch.cat([fts_xdconv, fts_qrs], dim=-1)

                layer_name = "xdconv_dense{}".format(layer_idx)
                fts_fuse = self.net._modules[layer_name].forward(fts_concat).transpose(1,2)
                fts_fuse = self.net._modules[layer_name+"bn"].forward(fts_fuse).transpose(1,2)
                layer_pts.append(qrs)
                layer_fts.append(fts_fuse)

        fc_layers=[layer_fts[-1]]
        for layer_idx, layer_param in enumerate(self.fc_params):
            fc_layers.append(self.net._modules["dense{}".format(layer_idx)].forward(fc_layers[-1]))
            fc_layers.append(self.net._modules["fc_dropout_{}".format(layer_idx)].forward(fc_layers[-1]))

        if self.setting['task']=='cls':
            if self.net.training:
                input_of_logits_layer = fc_layers[-1]
            else:
                input_of_logits_layer = fc_layers[-1].mean(dim=1, keepdim=True)
            logits=self.net._modules["logits"].forward(input_of_logits_layer)
            return logits
        elif self.setting['task']=='seg':
            logits = self.net._modules["logits"].forward(fc_layers[-1])
            return logits
        else:
            raise NotImplementedError

    def fps(self, pts, fts, qrs):
        '''

        :param pts: coordinate matrix
        :return:
        '''
        raise NotImplementedError
        pass

def modelnet_x3_l4()->PointCNN:
    setting={}
    setting["num_classes"] = 40
    x = 3
    xconv_param_name = ('K', 'D', 'P', 'C', 'links')
    setting["xconv_params"] = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                    [(8, 1, -1, 16 * x, []),
                     (12, 2, 384, 32 * x, []),
                     (16, 2, 128, 64 * x, []),
                     (16, 3, 128, 128 * x, [])]]
    setting["with_global"] = False

    fc_param_name = ('C', 'dropout_rate')
    setting["fc_params"] = [dict(zip(fc_param_name, fc_param)) for fc_param in
                 [(128 * x, 0.0),
                  (64 * x, 0.8)]]

    setting["sampling"] = 'random'
    setting["sample_num"]=config.dataset_setting["sample_num"]

    setting["data_dim"] = 6
    setting["task"]="cls"
    ###### Do not change this
    setting['fts_is_None'] = config.dataset_setting["data_dim"]<=3
    if not setting['fts_is_None']:
        setting['fts_is_None'] = not config.dataset_setting["use_extra_features"]
    ###### Do not change this
    setting["with_X_transformation"] = True
    return PointCNN(setting)

def shapenet_x8_2048_fps()->PointCNN:
    setting={}
    setting["num_classes"] = 50
    x = 8
    xconv_param_name = ('K', 'D', 'P', 'C', 'links')
    setting["xconv_params"] = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                               [(8, 1, 2048, 32 * x, []),
                                (12, 2, 768, 32 * x, []),
                                (16, 2, 384, 64 * x, []),
                                (16, 6, 128, 128 * x, [])]]
    xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
    setting["xdconv_params"] = [dict(zip(xdconv_param_name, xdconv_param)) for xdconv_param in
                     [(16, 6, 3, 3),
                      (16, 6, 3, 2),
                      (12, 6, 2, 1),
                      (8, 6, 1, 0),
                      (8, 4, 0, 0)]]
    setting["with_global"] = True
    fc_param_name = ('C', 'dropout_rate')
    setting["fc_params"] = [dict(zip(fc_param_name, fc_param)) for fc_param in
                            [(32 * x, 0.0),
                             (32 * x, 0.5)]]

    setting["sampling"] = 'fps'
    setting["sample_num"] = config.dataset_setting["sample_num"]

    setting["data_dim"] = 3
    setting["task"] = "seg"
    ###### Do not change this
    setting['fts_is_None'] = config.dataset_setting["data_dim"] <= 3
    if not setting['fts_is_None']:
        setting['fts_is_None'] = not config.dataset_setting["use_extra_features"]
    ###### Do not change this
    setting["with_X_transformation"] = True
    return PointCNN(setting)


def scannet_x8_2048_fps4()->PointCNN:
    setting={}
    setting["num_classes"] = 21
    x = 8
    xconv_param_name = ('K', 'D', 'P', 'C', 'links')
    setting["xconv_params"] = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                    [(8, 1, -1, 32 * x, []),
                    (12, 2, 768, 64 * x, []),
                    (16, 2, 384, 96 * x, []),
                    (16, 4, 128, 128 * x, [])]]
    setting["with_global"] = True

    xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
    setting["xdconv_params"] = [dict(zip(xdconv_param_name, xdconv_param)) for xdconv_param in
                     [(16, 4, 3, 3),
                      (16, 2, 3, 2),
                      (12, 2, 2, 1),
                      (8, 2, 1, 0)]]

    fc_param_name = ('C', 'dropout_rate')
    setting["fc_params"] = [dict(zip(fc_param_name, fc_param)) for fc_param in
                 [(32 * x, 0.0),
                  (32 * x, 0.5)]]

    setting["sampling"] = 'fps'
    setting["sample_num"]=config.dataset_setting["sample_num"]

    setting["data_dim"] = 3
    setting["task"]="seg"
    ###### Do not change this
    setting['fts_is_None'] = config.dataset_setting["data_dim"]<=3
    if not setting['fts_is_None']:
        setting['fts_is_None'] = not config.dataset_setting["use_extra_features"]
    ###### Do not change this
    setting["with_X_transformation"] = True
    return PointCNN(setting)
