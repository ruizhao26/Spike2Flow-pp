import os
import h5py
import numpy as np
import os.path as osp
import torch
import random
from datasets.ds_utils import *
import json
import time

TEST_SCENES = [
    '2016-09-02_000170',
    'ball_000000',
    'car_tuebingen_000024',
    'car_tuebingen_000103',
    'car_tuebingen_000145',
    'horses__000028',
    'kids__000002',
    'motocross_000108',
    'skatepark_000034',
    'spitzberglauf_000009',
    'tubingen_05_09_000012'
]


class Augmentor:
    def __init__(self, crop_size, do_flip):
        # spatial augmentation params
        self.crop_size = crop_size
        self.spatial_aug_prob = 0.8

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.2

    def spatial_transform(self, spk_list, flow_list=None):
        y0 = np.random.randint(0, spk_list.shape[1] - self.crop_size[0])
        x0 = np.random.randint(0, spk_list.shape[2] - self.crop_size[1])
        do_lr_flip = np.random.rand() < self.h_flip_prob
        do_ud_flip = np.random.rand() < self.v_flip_prob

        if self.do_flip:
            if do_lr_flip:
                spk_list = np.flip(spk_list, axis=2)
            if do_ud_flip:
                spk_list = np.flip(spk_list, axis=1)
        spk_list = spk_list[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            
        if flow_list != None:
            for ii, flow in enumerate(flow_list):
                flow = flow.transpose([2, 0, 1])
                if self.do_flip:
                    if do_lr_flip:
                        flow = np.flip(flow, axis=2)
                        flow[0,:,:] = -flow[0,:,:]
                    if do_ud_flip:
                        flow = np.flip(flow, axis=1)
                        flow[1,:,:] = -flow[1,:,:]
                flow = flow[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
                flow_list[ii] = flow

        return spk_list, flow_list

    def __call__(self, spk_list, flow_list=None):
        spk_list, flow_list = self.spatial_transform(spk_list, flow_list)
        spk_list = np.ascontiguousarray(spk_list)
        flow_list = [np.ascontiguousarray(flow) for flow in flow_list]
        return spk_list, flow_list


class H5Loader_rssf_train(torch.utils.data.Dataset):
    """
        self.cfg:                       config data
        self.pair_step:                 step for sampling the dataset to get a sample in a mini-batch
        self.spike_sub_stream_num:      number of spike sub-stream for a sample in a minibatch
    """
    def __init__(self, cfg):
        self.cfg = cfg

        self.pair_step = self.cfg['loader']['pair_step']
        self.spike_sub_stream_num = 7
    
        # Augmentor
        self.augmentor = Augmentor(crop_size=self.cfg['loader']['crop_size'], do_flip=self.cfg['loader']['do_flip'])
        self.img_size_dict = json.load(open(cfg['data']['image_size_json']))
        self.samples = self.collect_samples()
        print('dataset RSSF - Training, samples num: {:d}'.format(len(self.samples)))

    def confirm_exist(self, path_list_list):
        for pl in path_list_list:
            for p in pl:
                if not osp.exists(p):
                    print(p)
                    return 0
        return 1

    def collect_samples(self):
        spike_path = osp.join(self.cfg['data']['rssf_path'], 'crop_dataset', 'dsft_up10', 'eta_0.80_gamma_60_alpha_0.70')
        scene_list = sorted(os.listdir(spike_path))
        samples = []
        for scene in scene_list:
            # select the scene
            if scene in TEST_SCENES:
                continue

            spike_dir = osp.join(spike_path, scene)
            flow1_dir = osp.join(self.cfg['data']['rssf_path'], 'crop_dataset', 'flow1', scene)
            flow2_dir = osp.join(self.cfg['data']['rssf_path'], 'crop_dataset', 'flow2', scene)
            flow3_dir = osp.join(self.cfg['data']['rssf_path'], 'crop_dataset', 'flow3', scene)
            spike_crop_list = sorted(os.listdir(osp.join(spike_dir)))
            spike_path_list = sorted(os.listdir(osp.join(spike_dir, '00')))
            
            for st in range(2, len(spike_path_list)-(self.spike_sub_stream_num), self.pair_step):
                data_range = range(st, st+self.spike_sub_stream_num)
                spikes_path_list = [osp.join(spike_dir, '00', spike_path_list[ii]) for ii in data_range]
                flow1_path = osp.join(flow1_dir, '00', spike_path_list[st+2][:-3]+'.flo')
                flow2_path = osp.join(flow2_dir, '00', spike_path_list[st+2][:-3]+'.flo')
                flow3_path = osp.join(flow3_dir, '00', spike_path_list[st+2][:-3]+'.flo')
                
                #### 读入70 spike frame
                #### 估计光流 20 -> 30, 20 -> 40, 20 -> 50

                if(self.confirm_exist([spikes_path_list, [flow1_path, flow2_path, flow3_path]])):
                    s = {}
                    s['spikes_paths'] = spikes_path_list
                    s['spike_crop_list'] = spike_crop_list
                    s['flow1_path'] = flow1_path
                    s['flow2_path'] = flow2_path
                    s['flow3_path'] = flow3_path
                    s['scene'] = scene
                    samples.append(s)
        return samples

    def replace_crop(self, path, crop_id_str):
        path_split = path.split(os.sep)
        path_split[-2] = crop_id_str
        return os.sep.join(path_split)

    def _load_sample(self, s):
        data = {}
        ## 随机选择crop并替换crop
        crop_id_str = random.choice(s['spike_crop_list'])
        s['spikes_paths'] = [self.replace_crop(p, crop_id_str) for p in s['spikes_paths']]
        s['flow1_path'] = self.replace_crop(s['flow1_path'], crop_id_str)
        s['flow2_path'] = self.replace_crop(s['flow2_path'], crop_id_str)
        s['flow3_path'] = self.replace_crop(s['flow3_path'], crop_id_str)

        h5files = [h5py.File(p) for p in s['spikes_paths']]
        data['spikes'] = [np.array(f['dsft']) for f in h5files]
        data['spikes'] = np.concatenate(data['spikes'], axis=0)
        data['flows'] = [read_gen(s['flow{:d}_path'.format(ii)]).astype(np.float32) for ii in range(1,4)]
        
        # Augmentation
        data['spikes'], data['flows'] = self.augmentor(data['spikes'], data['flows'])
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = self._load_sample(self.samples[index])
        return data


class H5Loader_rssf_test(torch.utils.data.Dataset):
    """
        self.cfg:                       config data
        self.pair_step:                 step for sampling the dataset to get a sample in a mini-batch
        self.spike_sub_stream_num:      number of spike sub-stream for a sample in a minibatch
    """
    def __init__(self, cfg, ds_type='test', scene=None, valid=False, crop_len=200):
        self.cfg = cfg
        self.type = ds_type
        self.scene = scene
        self.valid = valid

        ####### Settings of the RSSF-Eval #######
        self.crop_len = crop_len
        self.h = 768
        self.w = 1024
        #########################################

        self.pair_step = 1
        self.spike_sub_stream_num = 7
        
        self.img_size_dict = json.load(open(cfg['data']['image_size_json']))
        self.samples = self.collect_samples()
        print('scene {:25s}, samples num: {:d}'.format(self.scene, len(self.samples)))

    def confirm_exist(self, path_list_list):
        for pl in path_list_list:
            for p in pl:
                if not osp.exists(p):
                    print(p)
                    return 0
        return 1

    def collect_samples(self):
        spike_path = osp.join(self.cfg['data']['rssf_path'], 'dsft_up10', 'eta_0.80_gamma_60_alpha_0.70')
        scene_list = sorted(os.listdir(spike_path))
        samples = []
        for scene in scene_list:
            if scene != self.scene:
                continue

            spike_dir = osp.join(spike_path, scene)
            flow1_dir = osp.join(self.cfg['data']['rssf_path'], 'flow1', scene)
            flow2_dir = osp.join(self.cfg['data']['rssf_path'], 'flow2', scene)
            flow3_dir = osp.join(self.cfg['data']['rssf_path'], 'flow3', scene)
            spike_path_list = sorted(os.listdir(spike_dir))

            crop_len = self.crop_len
            
            if self.valid:
                end_idx = crop_len + 1
            else:
                end_idx = len(spike_path_list)-(self.spike_sub_stream_num)
            
            for st in range(2, end_idx, self.pair_step):
                data_range = range(st, st+self.spike_sub_stream_num)
                spikes_path_list = [osp.join(spike_dir, spike_path_list[ii]) for ii in data_range]
                flow1_path = osp.join(flow1_dir, spike_path_list[st+2][:-3]+'.flo')
                flow2_path = osp.join(flow2_dir, spike_path_list[st+2][:-3]+'.flo')
                flow3_path = osp.join(flow3_dir, spike_path_list[st+2][:-3]+'.flo')
                
                #### 读入70 spike frame
                #### 估计光流 20 -> 30, 20 -> 40, 20 -> 50
                
                if(self.confirm_exist([spikes_path_list, [flow1_path, flow2_path, flow3_path]])):
                    s = {}
                    s['spikes_paths'] = spikes_path_list
                    s['flow1_path'] = flow1_path
                    s['flow2_path'] = flow2_path
                    s['flow3_path'] = flow3_path
                    s['scene'] = scene
                    samples.append(s)
        return samples

    def central_crop(self, x, channel_first=True):
        if channel_first:
            c, h, w = x.shape
        else:
            h, w, c = x.shape
        crop_h = (h - self.h) // 2 if h > self.h else 0
        crop_w = (w - self.w) // 2 if w > self.w else 0
        if channel_first:
            if crop_h > 0:
                x = x[:, crop_h:-crop_h, :]
            if crop_w > 0:
                x = x[:, :, crop_w:-crop_w]
        else:
            if crop_h > 0:
                x = x[crop_h:-crop_h, :, :]
            if crop_w > 0:
                x = x[:, crop_w:-crop_w, :]
        return x

    def _load_sample(self, s):
        data = {}
        h5files = [h5py.File(p) for p in s['spikes_paths']]
        data['spikes'] = [self.central_crop(np.array(f['dsft']), channel_first=True) for f in h5files]

        data['flow1'] = self.central_crop(read_gen(s['flow1_path']).astype(np.float32), channel_first=False)
        data['flow2'] = self.central_crop(read_gen(s['flow2_path']).astype(np.float32), channel_first=False)
        data['flow3'] = self.central_crop(read_gen(s['flow3_path']).astype(np.float32), channel_first=False)
        data['flows'] = [data['flow1'], data['flow2'], data['flow3']]
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = self._load_sample(self.samples[index])
        return data


def get_test_datasets(cfg, valid=False, crop_len=50):
    test_datasets = {}
    for ss in TEST_SCENES:
        cur_dataset = H5Loader_rssf_test(cfg, scene=ss, valid=valid, crop_len=crop_len)
        test_datasets[ss] = cur_dataset
    return test_datasets
