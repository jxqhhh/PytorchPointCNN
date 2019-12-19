from __future__ import division


import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import sys
sys.path.append("..")
import config

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(self.shape)

class XConv(torch.nn.Module):

    def __init__(self, setting, idx, K, dilation, P, C, C_pts_fts, depth_multiplier, with_X_transfomation,
                 with_global, sorting_method=None, late_bn=False):
        '''
        :param parent: dict，存储所属的PointCNN结构中的"xconv_params"和"xdconv_params"和"xconv_output_dim"
        :param idx: XConv是其所属的PointCNN结构中的第idx个XConv层（idx从0开始计数，包括xdconv）
        :param K: hyperparameter of KNN
        :param C: dimension of feature
        :param dilation: dilation rate of KNN
        :param C_pts_fts: we will individually lift each point into C_pts_fts ("pts_fts" means features from points) dimension (corresponding to Step 2 of the XConv operator)
        :param P: num of representative points
        :param with_X_transfomation
        :param depth_multiplier:
        :param norm
        :param de_norm
        '''
        super(XConv, self).__init__()
        self.setting=setting
        self.idx=idx
        self.K = K
        self.dilation = dilation
        self.sorting_method = sorting_method
        self.with_X_transformation = with_X_transfomation
        self.P = P
        self.C = C
        self.C_pts_fts = C_pts_fts
        self.depth_multiplier = depth_multiplier
        self.late_bn=late_bn
        self.add_module(name='dense1',
                        module=torch.nn.Sequential(torch.nn.Linear(3, self.C_pts_fts), torch.nn.ELU()))
        self.add_module(name='BN1', module=torch.nn.BatchNorm1d(self.C_pts_fts, momentum=0.99))
        self.add_module(name='dense2',
                        module=torch.nn.Sequential(torch.nn.Linear(self.C_pts_fts, self.C_pts_fts), torch.nn.ELU()))
        self.add_module(name='BN2', module=torch.nn.BatchNorm1d(self.C_pts_fts, momentum=0.99))
        if self.with_X_transformation:
            self.add_module(name='x_trans_conv1', module=torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, \
                                                                                             out_channels=self.K * self.K,
                                                                                             kernel_size=(1, self.K),bias=False),
                                                                             torch.nn.ELU(), torch.nn.BatchNorm2d(
                    num_features=self.K * self.K)))

            self.add_module(name='x_trans_depthConv1', module=torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=self.K,out_channels=self.K*self.K,kernel_size=(1, self.K),groups=K,bias=False),
                torch.nn.ELU(), torch.nn.BatchNorm2d(num_features=self.K * self.K)))
            self.add_module(name='x_trans_depthConv2', module=torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=self.K,out_channels=self.K*self.K,kernel_size=(1, self.K),groups=K,bias=False),
                torch.nn.ELU(), torch.nn.BatchNorm2d(num_features=self.K * self.K)))

        self.with_global= ( with_global and idx == ( len(self.setting["xconv_params"]) - 1 ) )
        if self.idx<len((self.setting["xconv_params"])):
            in_channels_fts_conv=self.C_pts_fts+self.setting["xconv_output_dim"][-1]
        else:
            in_channels_fts_conv=self.C_pts_fts+self.setting["xconv_output_dim"][self.setting["xdconv_params"][self.idx-len((self.setting["xconv_params"]))]["qrs_layer_idx"]+1]


        self.add_module(name="fts_conv",module=torch.nn.Sequential(torch.nn.Conv2d(in_channels=in_channels_fts_conv,
                out_channels=in_channels_fts_conv*self.depth_multiplier,kernel_size=(1, self.K),groups=in_channels_fts_conv,bias=False).to(device=config.device),
                torch.nn.Conv2d(in_channels=self.depth_multiplier * in_channels_fts_conv,out_channels=self.C, kernel_size=(1, 1),bias=False)
                .to(device=config.device),torch.nn.ELU().to(device=config.device)).to(device=config.device))
        self.add_module(name="fts_conv_BN",module=torch.nn.BatchNorm1d(num_features=self.C))
        if self.with_global:
            self.add_module(name="dense3",module=torch.nn.Sequential(torch.nn.Linear(3, self.C // 4).to(device=config.device),
                torch.nn.ELU().to(device=config.device),View(shape=(-1,self.C//4)),
                torch.nn.BatchNorm1d(num_features=self.C // 4).to(device=config.device), \
                torch.nn.Linear(self.C // 4, self.C // 4).to(device=config.device), torch.nn.ELU().to(device=config.device),
                torch.nn.BatchNorm1d(num_features=self.C // 4).to(device=config.device),
                View(shape=(-1,self.P,self.C//4))))

    def forward(self, pts, fts, qrs):
        N = pts.shape[0]  # batch size
        point_num=pts.shape[1]
        # xconv operation
        _, indices_dilated = self.knn_indices_general(qrs, pts, True)
        indices = indices_dilated[:, :, ::self.dilation, :]  # indices of K neaerest(dilation: d) points
        indices=(indices.view(-1,2)[:,1].cpu()+torch.arange(0,N*point_num,point_num).view(-1,1).repeat(1,self.P*self.K).view(-1)).cpu().numpy()
        if self.sorting_method is not None:
            raise NotImplementedError

        nn_pts=(pts.contiguous().view(-1,3))[indices].view(N,self.P,self.K,3) # coordinates of nearest-neighbour points
        nn_pts_center = qrs.unsqueeze(dim=2)  # (N, P, 1, 3) # coordinates of queries
        nn_pts_local_origin = nn_pts - nn_pts_center  # (N, P, K, 3) # relative coordinates

        knn_pts_len=torch.norm(nn_pts_local_origin,dim=3,keepdim=False).detach() # (N,P,K) # stop gradient here!
        nn_pts_max_len=torch.unsqueeze(torch.mean(knn_pts_len,dim=-1,keepdim=True),dim=-1) # (N,P,1,1)
        nn_pts_local=nn_pts_local_origin/nn_pts_max_len


        nn_fts_from_pts_0 = self._modules['BN1'].forward(
            (self._modules['dense1'].forward(nn_pts_local.view(-1, 3)))).view(N, self.P, self.K, self.C_pts_fts)
        nn_fts_from_pts = self._modules['BN2'].forward(
            self._modules['dense2'].forward(nn_fts_from_pts_0.view(-1, self.C_pts_fts))).view(N, self.P, self.K,\
            self.C_pts_fts)  # shape: (N,P,K,C_pts_fts)

        if fts is None:
            nn_fts_input = nn_fts_from_pts  # no concat!
        else:
            nn_fts_from_prev=(fts.contiguous().view(N*point_num,-1))[indices].contiguous().view(N,self.P,self.K,-1)# the F matrix
            nn_fts_input = torch.cat([nn_fts_from_pts, nn_fts_from_prev],
                                     dim=-1)

        if self.with_X_transformation:
            ######################## X-transformation #########################
            nn_pts_local = nn_pts_local.transpose(1, 3).transpose(2, 3)  # (N,3,P,K)
            X_0 = self._modules["x_trans_conv1"].forward(nn_pts_local)
            X_0_KK = X_0.view(N, self.K, self.K, self.P).transpose(1, 2).transpose(2, 3)# (N,K,P,K)

            X_1 = self._modules['x_trans_depthConv1'].forward(X_0_KK)  # (N,K*K,P,1)
            X_1_KK = X_1.view(N, self.K, self.K, self.P).transpose(1,2).transpose(2, 3)  # (N,K,P,K)
            X_2 = self._modules['x_trans_depthConv2'].forward(X_1_KK)  # (N,K*K,P,1)
            X_2_KK = X_2.view(N, self.K, self.K, self.P).transpose(1,2).transpose(2, 3)  # (N,K,P,K)
            X_2_KK = X_2_KK.transpose(1, 2).transpose(2,3)  # (N,P,K,K) # output of Step 4 of algorithm 1
            fts_X = torch.matmul(X_2_KK, nn_fts_input)  # output of Step 5 of algorithm 1
            ###################################################################
        else:
            fts_X = nn_fts_input

        fts_conv_3d = self._modules['fts_conv'].forward(fts_X.transpose(1, 3).transpose(2, 3)).transpose(1,2).contiguous().view(-1,self.C)
        fts_conv_3d = self._modules["fts_conv_BN"].forward(fts_conv_3d).view(N,self.P,self.C) # (N,P,C)


        if self.late_bn:
            raise NotImplementedError

        if self.with_global:
            fts_global = self._modules['dense3'].forward(qrs)
            return torch.cat([fts_global, fts_conv_3d], dim=-1)
        else:
            return fts_conv_3d


    def knn_indices_general(self, queries, points, sort=True):
        """

        :param queries:
        :param points:
        :param sort:
        :param unique:
        :return: (distances, indices): indicies is of shape (N,P,K,2); distances is of shape (N,P,K)
        """
        k = self.K * self.dilation
        queries_shape = queries.shape
        batch_size = queries_shape[0]
        point_num = queries_shape[1]
        Distance = XConv.batch_distance_matrix_general(queries, points)
        distances, point_indices = torch.topk(-Distance, k=k, sorted=sort)  # (N, P, K)
        batch_indices = (torch.arange(batch_size).to(config.device).view(-1, 1, 1, 1)).repeat(1, point_num, k, 1)
        indices = torch.cat([batch_indices, torch.unsqueeze(point_indices, dim=3)], dim=3)
        distances = -distances
        return distances, indices  # shape of indices is (N, P, K, 2)

    # A shape is (batch_size, P_A, C), B shape is (batch_size, P_B, C)
    # D shape is (N, P_A, P_B)
    @staticmethod
    def batch_distance_matrix_general(A, B):
        r_A = torch.sum(A * A, dim=-1).unsqueeze_(dim=-1)
        r_B = torch.sum(B * B, dim=-1).unsqueeze_(dim=-1)
        m = torch.matmul(A, torch.transpose(B, -1, -2))
        D = r_A - 2 * m + torch.transpose(r_B, -1, -2)
        return D

    '''
    @staticmethod
    # add a big value to duplicate columns
    def prepare_for_unique_top_k(D, A):
        indices_duplicated = XConv.find_duplicate_columns(A)
        D += D.max().item() * indices_duplicated

    # A shape is (N, P,C)
    @staticmethod
    def find_duplicate_columns(A):
        N = A.shape[0]
        P = A.shape[1]
        indices_duplicated = torch.ones((N, 1, P))
        for idx in range(N):
            _, indices = np.unique(A[idx].cpu().numpy, return_index=True, axis=0)
            indices_duplicated[idx, :, indices] = 0
        return indices_duplicated

    '''
