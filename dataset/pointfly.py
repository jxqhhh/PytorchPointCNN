import random

import numpy as np
import torch
from transforms3d.euler import euler2mat

def augment(points, xforms, range=None):
    """

    :param points: points xyz matrix with shape [N1,N2,3], where N1 denotes num of samples and N2 denotes num of points in every sample
    :param xforms:
    :param range: range for jitter
    :return:
    """
    xforms=torch.Tensor(xforms)
    points_xformed = torch.matmul(points, xforms)
    if range is None:
        return points_xformed

    jitter_data = range*torch.distributions.Normal(loc=0.0, scale=1.0).sample(points_xformed.size())
    jitter_clipped = torch.clamp(jitter_data, -5 * range, 5 * range)
    return points_xformed + jitter_clipped


def gauss_clip(mu, sigma, clip):
    v = random.gauss(mu, sigma)
    v = max(min(v, mu + clip * sigma), mu - clip * sigma)
    return v


def uniform(bound):
    return bound * (2 * random.random() - 1)


def scaling_factor(scaling_param, method):
    try:
        scaling_list = list(scaling_param)
        return random.choice(scaling_list)
    except:
        if method == 'g':
            return gauss_clip(1.0, scaling_param, 3)
        elif method == 'u':
            return 1.0 + uniform(scaling_param)


def rotation_angle(rotation_param, method):
    try:
        rotation_list = list(rotation_param)
        return random.choice(rotation_list)
    except:
        if method == 'g':
            return gauss_clip(0.0, rotation_param, 3)
        elif method == 'u':
            return uniform(rotation_param)


def get_xforms(xform_num, rotation_range=(0, 0, 0, 'u'), scaling_range=(0.0, 0.0, 0.0, 'u'), order='rxyz'):
    xforms = np.empty(shape=(xform_num, 3, 3))
    rotations = np.empty(shape=(xform_num, 3, 3))
    for i in range(xform_num):
        rx = rotation_angle(rotation_range[0], rotation_range[3])
        ry = rotation_angle(rotation_range[1], rotation_range[3])
        rz = rotation_angle(rotation_range[2], rotation_range[3])
        rotation = euler2mat(rx, ry, rz, order)

        sx = scaling_factor(scaling_range[0], scaling_range[3])
        sy = scaling_factor(scaling_range[1], scaling_range[3])
        sz = scaling_factor(scaling_range[2], scaling_range[3])
        scaling = np.diag([sx, sy, sz])

        xforms[i, :] = scaling * rotation
        rotations[i, :] = rotation
    return xforms, rotations


# the returned indices will be used by tf.gather_nd
def get_indices(batch_size, sample_num, point_num, pool_setting=None):
    if not isinstance(point_num, np.ndarray):
        point_nums = np.full((batch_size), point_num)
    else:
        point_nums = point_num
    indices = []
    for i in range(batch_size):
        pt_num = point_nums[i]
        if pool_setting is None:
            pool_size = pt_num
        else:
            if isinstance(pool_setting, int):
                pool_size = min(pool_setting, pt_num)
            elif isinstance(pool_setting, tuple):
                pool_size = min(random.randrange(pool_setting[0], pool_setting[1] + 1), pt_num)
        if pool_size > sample_num:
            choices = np.random.choice(pool_size, sample_num, replace=False)
        else:
            choices = np.concatenate((np.random.choice(pool_size, pool_size, replace=False),
                                      np.random.choice(pool_size, sample_num - pool_size, replace=True)))
        if pool_size < pt_num:
            choices_pool = np.random.choice(pt_num, pool_size, replace=False)
            choices = choices_pool[choices]
        choices = np.expand_dims(choices, axis=1)
        choices_2d = np.concatenate((np.full_like(choices, i), choices), axis=1)
        indices.append(choices_2d)
    return np.stack(indices)


def global_norm(pts):
    pts_data = pts[:, :, :3]
    pts_data = pts_data - np.mean(pts_data, axis=1, keepdims=True)
    pts_data /= np.linalg.norm(pts_data, axis=-1, keepdims=True).max(axis=1, keepdims=True)
    rst = pts_data
    if pts.shape[1] > 3:
        pts_normal = pts[:, :, 3:]
        rst = np.concatenate([pts_data, pts_normal], axis=-1)
    return rst

'''
def gather_nd(params, indices, name=None):

    //the input indices must be a 2d tensor in the form of [[a,b,..,c],...],
    //which represents the location of the elements.

    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1
    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)
    return torch.take(params, idx)
'''