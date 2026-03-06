# -*- coding: utf-8 -*-
import argparse
import os
import time
import cv2
import os.path as osp
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import pprint
from utils import *
from logger import *
from configs.yml_parser import YAMLParser
from easydict import EasyDict

from torch.cuda.amp import autocast, GradScaler

from model.get_model import get_model
from datasets.h5_loader_rssf import *
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--configs', '-c', type=str, default='./configs/spike2flow.yml')
parser.add_argument('--save_dir', '-sd', type=str, default='./outputs')
parser.add_argument('--batch_size', '-bs', type=int, default=6)
parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
parser.add_argument('--num_workers', '-j', type=int, default=12)
parser.add_argument('--start-epoch', '-se', type=int, default=0)
parser.add_argument('--pretrained', '-prt', type=str, default=None)
parser.add_argument('--print_freq', '-pf', type=int, default=100)
parser.add_argument('--vis_path', '-vp', type=str, default='./vis')
parser.add_argument('--model_iters', '-mit', type=int, default=8)
parser.add_argument('--no_warm', '-nw', type=int, default=1)
parser.add_argument('--eval', '-e', action='store_true')
parser.add_argument('--save_name', '-sn', type=str, default=None)
parser.add_argument('--warm_iters', '-wi', type=int, default=3000)
parser.add_argument('--eval_vis', '-ev', type=str, default='eval_vis')
parser.add_argument('--crop_len', '-clen', type=int, default=200)
parser.add_argument('--with_valid', '-wv', type=bool, default=True)
parser.add_argument('--decay_interval', '-di', type=int, default=10)
parser.add_argument('--decay_factor', '-df', type=float, default=0.7)
parser.add_argument('--valid_vis_freq', '-vvf', type=float, default=10)
parser.add_argument('--compile_model', action='store_true')
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--seed', type=int, default=2728)

parser.add_argument('--logs_file_name', type=str, default='spike2flow')
parser.add_argument('--model_name', type=str)

parser.add_argument('--weight_light_loss', '-wll', type=float, default=0.2)
args = parser.parse_args()

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
cfg_parser = YAMLParser(args.configs)
cfg = cfg_parser.config

n_iter = 0


print('model: ', args.model_name)


if args.print_freq != None:
    cfg['train']['print_freq'] = args.print_freq
if args.batch_size != None:
    cfg['loader']['batch_size'] = args.batch_size
if args.epochs != None:
    cfg['loader']['n_epochs'] = args.epochs


##########################################################################################################
## Configs
writer_root = 'logs/{:s}/'.format(args.logs_file_name)
os.makedirs(writer_root, exist_ok=True)
writer_path = writer_root + args.model_name + '.txt'
##########################################################################################################
# Set Writers for each writer
writer = open(writer_path, 'a')
##################################
# format-print argparse
for k, v in vars(args).items():
    vv = pprint.pformat(v)
    ostr = '{:s} : {:s}'.format(k, vv)
    writer.write(ostr + '\n')
##################################


################################# warm up ###################################
warmup = WarmUp(ed_it=args.warm_iters, st_lr=1e-7, ed_lr=args.learning_rate)
#############################################################################


##########################################################################################################
## Train
def train(cfg, train_loader, model, optimizer, scaler, epoch, log, writer, train_writer):
    ######################################################################
    ## Init
    global n_iter
    batch_time = AverageMeter(precision=3)
    data_time = AverageMeter(precision=3)
    losses = AverageMeter(i=1, precision=6, names=['Loss'])
    
    model.train()
    end = time.time()

    ######################################################################
    ## Training Loop
    for ww, data in enumerate(train_loader, 0):
        if (not args.no_warm) and (n_iter <= args.warm_iters):
            warmup.adjust_lr(optimizer=optimizer, cur_it=n_iter)
        
        st1 = time.time()
        spikes = data['spikes']
        # spikes = [spk.cuda() for spk in spikes]
        # spks = torch.cat(spikes, dim=1)
        
        spks = spikes.cuda().float()

        flow1gt = data['flows'][0].cuda()
        flow2gt = data['flows'][1].cuda()
        flow3gt = data['flows'][2].cuda()

        flowgt = [flow1gt, flow2gt, flow3gt]
        if ww != 0:
            data_time.update(time.time() - end)

        flow = model(spks=spks, iters=args.model_iters)

        ## compute loss
        loss, loss_deriv_dict = supervised_loss(flow, flowgt)
        flow_mean = loss_deriv_dict['flow_mean']
        
        # record loss
        losses.update(loss.item())

        if ww % 10 == 0:
            train_writer.add_scalar('total_loss', loss.item(), n_iter)
            train_writer.add_scalar('flow_mean', loss_deriv_dict['flow_mean'], n_iter)

        ## compute gradient and optimize
        optimizer.zero_grad()

        if cfg['train']['mixed_precision']:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # record elapsed time
        if ww != 0:
            batch_time.update(time.time() - end)
        end = time.time()
        
        n_iter += 1
        if n_iter % cfg['train']['vis_freq'] == 0:
                vis_flow_batch(flow[-1], args.vis_path, suffix=args.model_name, max_batch=16)
 
        ## output logs
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        ostr = 'Epoch: [{:d}] [{:d}/{:d}],  Iter: {:d}  '.format(epoch, ww, len(train_loader), n_iter-1)
        ostr += 'Time: {},  Data: {},  Loss: {}, Flow mean {:.4f}, lr {:.7f}'.format(batch_time, data_time, losses, flow_mean, cur_lr)
        # log.info(ostr)
        if ww % cfg['train']['print_freq'] == 0:
            writer.write(ostr + '\n')
            print(ostr)

        end = time.time()
    return


##########################################################################################################
## valid
def validation(cfg, test_datasets, model, writer, log):
    global n_iter
    data_time = AverageMeter()
    metrics_name = ['AEE1', 'AEE2', 'AEE3', 'PO_1', 'PO_2', 'PO_3']
    all_metrics = AverageMeter(i=len(metrics_name), precision=4, names=metrics_name)

    model_time = AverageMeter()
    end = time.time()
    epe1_dict = {}
    epe2_dict = {}
    epe3_dict = {}
    po_1_dict = {}
    po_2_dict = {}
    po_3_dict = {}
    len_dict = {}

    # switch to evaluate mode
    model.eval()

    i_set = 0
    for scene, cur_test_set in test_datasets.items():
        i_set += 1
        cur_test_loader = torch.utils.data.DataLoader(
            cur_test_set,
            pin_memory = False,
            drop_last = False,
            batch_size = 1,
            shuffle = False,
            num_workers = args.num_workers)

        
        cur_all_metrics = AverageMeter(i=len(metrics_name), precision=4, names=metrics_name)
        cur_model_time = AverageMeter()

        cur_eval_root = osp.join(args.eval_vis, args.model_name, scene)
        os.makedirs(cur_eval_root, exist_ok=True)

        for ww, data in enumerate(cur_test_loader, 0):
            spikes = data['spikes']
            spikes = [spk.cuda() for spk in spikes]
            spks = torch.cat(spikes, dim=1).float()
            flow1gt = data['flows'][0].cuda().permute([0,3,1,2])
            flow2gt = data['flows'][1].cuda().permute([0,3,1,2])
            flow3gt = data['flows'][2].cuda().permute([0,3,1,2])

            data_time.update(time.time() - end)
            with torch.no_grad():
                st = time.time()
                flow = model(spks=spks, iters=args.model_iters)
                mtime = time.time() - st

            if ww % args.valid_vis_freq == 0:
                flow_vis = flow_to_image(flow[-1][0][0].permute([1,2,0]).cpu().numpy())
                cur_vis_path = osp.join(cur_eval_root, '{:03d}.png'.format(ww))
                cv2.imwrite(cur_vis_path, flow_vis)

            # epe
            epe1 = torch.norm(flow[-1][0] - flow1gt, p=2, dim=1).mean().item()
            epe2 = torch.norm(flow[-1][1] - flow2gt, p=2, dim=1).mean().item()
            epe3 = torch.norm(flow[-1][2] - flow3gt, p=2, dim=1).mean().item()
            po_1 = calculate_error_rate(flow[-1][0], flow1gt)
            po_2 = calculate_error_rate(flow[-1][1], flow2gt)
            po_3 = calculate_error_rate(flow[-1][2], flow3gt)
            res_list = [epe1, epe2, epe3, po_1, po_2, po_3]
            all_metrics.update(res_list)
            cur_all_metrics.update(res_list)

            if ww != 0:
                cur_model_time.update(mtime)
                model_time.update(mtime)
        

        epe1_dict[scene] = cur_all_metrics.avg[0]
        epe2_dict[scene] = cur_all_metrics.avg[1]
        epe3_dict[scene] = cur_all_metrics.avg[2]
        po_1_dict[scene] = cur_all_metrics.avg[3]
        po_2_dict[scene] = cur_all_metrics.avg[4]
        po_3_dict[scene] = cur_all_metrics.avg[5]

        len_dict[scene] = cur_test_set.__len__()
        ostr = 'Scene[{:02d}]: {:30s}  EPE1: {:.4f}  EPE2: {:.4f}  EPE3: {:.4f}  PO_1: {:.4f}  PO_2: {:.4f}  PO_3: {:.4f} AvgTime: {:.4f}'.format(
            i_set, scene, cur_all_metrics.avg[0], cur_all_metrics.avg[3], cur_all_metrics.avg[1], cur_all_metrics.avg[4], cur_all_metrics.avg[2], cur_all_metrics.avg[5], cur_model_time.avg[0])
        writer.write(ostr + '\n')
        print(ostr)
        time.sleep(0.1)
    
    ostr0 = 'All EPE1/PO1: {:.4f}/{:.4f}  All EPE2/PO2: {:.4f}/{:.4f}  All EPE3/PO3: {:.4f}/{:.4f}  AvgTime: {:.4f}'.format(
        all_metrics.avg[0], all_metrics.avg[3], all_metrics.avg[1], all_metrics.avg[4], all_metrics.avg[2], all_metrics.avg[5], model_time.avg[0])
    a_epe1, b_epe1, c_epe1 = get_class_metric(epe1_dict, len_dict)
    a_epe2, b_epe2, c_epe2 = get_class_metric(epe2_dict, len_dict)
    a_epe3, b_epe3, c_epe3 = get_class_metric(epe3_dict, len_dict)
    a_po1, b_po1, c_po1 = get_class_metric(po_1_dict, len_dict)
    a_po2, b_po2, c_po2 = get_class_metric(po_2_dict, len_dict)
    a_po3, b_po3, c_po3 = get_class_metric(po_3_dict, len_dict)
    ostr1 = 'EPE1/PO1: Class A: {:.4f}, {:.4f}  Class B: {:.4f}, {:.4f}  Class C: {:.4f}, {:.4f}'.format(a_epe1, a_po1, b_epe1, b_po1, c_epe1, c_po1)
    ostr2 = 'EPE2/PO2: Class A: {:.4f}, {:.4f}  Class B: {:.4f}, {:.4f}  Class C: {:.4f}, {:.4f}'.format(a_epe2, a_po2, b_epe2, b_po2, c_epe2, c_po2)
    ostr3 = 'EPE3/PO3: Class A: {:.4f}, {:.4f}  Class B: {:.4f}, {:.4f}  Class C: {:.4f}, {:.4f}'.format(a_epe3, a_po3, b_epe3, b_po3, c_epe3, c_po3)
    writer.write(ostr0 + '\n')
    writer.write(ostr1 + '\n')
    writer.write(ostr2 + '\n')
    writer.write(ostr3 + '\n')
    print(ostr0)
    print(ostr1)
    print(ostr2)
    print(ostr3)

    return



if __name__ == '__main__':
    set_seeds(args.seed)

    ##########################################################################################################
    # Create save path and logs
    timestamp1 = datetime.datetime.now().strftime('%m-%d')
    timestamp2 = datetime.datetime.now().strftime('%H%M%S')

    save_folder_name = 'a_{:s}_b{:d}_{:s}'.format(args.model_name, args.batch_size, timestamp2)

    save_path = osp.join(args.save_dir, timestamp1, save_folder_name)
    make_dir(save_path)
    ostr = '=>Save path: ' + save_path
    writer.write(ostr + '\n')
    train_writer = SummaryWriter(save_path)

    make_dir(args.vis_path)
    make_dir(args.eval_vis)


    ##########################################################################################################
    ## Create model
    model_dict =  {
            "dropout": 0.0,
            "mixed_precision": cfg['train']['mixed_precision'],
            "corr_levels": 3,
            "corr_radius": 3,
            }
    model_dict = EasyDict(model_dict)
    model = get_model(model_dict, model_name=args.model_name)

    if args.compile_model and (torch.__version__ >= '2.0.0'):
        ostr = 'Start compile the model'
        writer.write(ostr + '\n')
        st = time.time()
        torch.compile(model)
        ostr = 'Finish compiling the model  Time {:.2f}s'.format(time.time() - st)
        writer.write(ostr + '\n')

    if args.pretrained != None:
        network_data = torch.load(args.pretrained)
        if 'model' in network_data.keys():
            network_data = network_data['model']
        ostr = '=> using pretrained model {:s}'.format(args.pretrained)
        writer.write(ostr + '\n')
        ostr = '=> model params: {:.6f}M'.format(model.num_parameters()/1e6)
        writer.write(ostr + '\n')
        model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
        model.load_state_dict(network_data)
    else:
        network_data = None
        ostr = '=> train from scratch'
        writer.write(ostr + '\n')
        model.init_weights()
        ostr = '=> model params: {:.6f}M'.format(model.num_parameters()/1e6)
        writer.write(ostr + '\n')
        model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()


    cudnn.benchmark = True

    ##########################################################################################################
    ## Create Optimizer
    assert(cfg['optimizer']['solver'] in ['Adam', 'SGD'])
    ostr = '=> settings {:s} solver'.format(cfg['optimizer']['solver'])
    writer.write(ostr + '\n')
    param_groups = [{'params': model.module.parameters(), 'weight_decay': cfg['model']['flow_weight_decay']}]
    optimizer = torch.optim.Adam(param_groups, args.learning_rate, betas=(cfg['optimizer']['momentum'], cfg['optimizer']['beta']))

    if args.pretrained != None:
        network_data = torch.load(args.pretrained)
        if 'optimizer' in network_data.keys():
            optimizer.load_state_dict(network_data['optimizer'])
            osrt = "=> using optimizer ckpt {:s}".format(args.pretrained)

    if cfg['train']['mixed_precision']:
        scaler = GradScaler()
    

    ##########################################################################################################
    ## Dataset
    if not args.eval:
        train_set = H5Loader_rssf_train(cfg)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            drop_last = True,
            batch_size = cfg['loader']['batch_size'],
            shuffle = True,
            # pin_memory = True,
            # prefetch_factor = 6,
            num_workers = args.num_workers)
    
    if args.eval:
        test_datasets = get_test_datasets(cfg, valid=True, crop_len=args.crop_len)
        validation(cfg=cfg, test_datasets=test_datasets, model=model, writer=writer, log=None)
    else:
        test_datasets = get_test_datasets(cfg, valid=True, crop_len=args.crop_len)
        epoch = args.start_epoch
        while(True):
            train(
                cfg=cfg,
                train_loader=train_loader,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                log=None,
                writer=writer,
                train_writer=train_writer)
            epoch += 1

            if epoch % args.decay_interval == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * args.decay_factor

            if args.with_valid:
                if epoch % 20 == 0:
                    validation(
                        cfg=cfg, 
                        test_datasets=test_datasets, 
                        model=model, 
                        writer=writer,
                        log=None)

                # Save Model
                if epoch % 10 == 0:
                    model_save_name = '{:s}_epoch{:03d}.pth'.format(args.model_name, epoch)
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    if cfg['train']['mixed_precision']:
                        checkpoint["scaler"] = scaler.state_dict()
                    torch.save(checkpoint, osp.join(save_path, model_save_name))

            if epoch >= cfg['loader']['n_epochs']:
                break