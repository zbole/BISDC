from typing import Optional

import torch, glob, os, numpy as np
import sys
sys.path.append('../')

from util.log import logger

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = max(base_lr * (multiplier ** (epoch // step_epoch)), clip)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))  # area_intersection: K, indicates the number of members in each class in intersection
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def checkpoint_restore(model, exp_path, exp_name, use_cuda=True, epoch=0, dist=False, f='', second_model_path=None):
    # 将模型转到CPU
    if use_cuda:
        model.cpu()

    # 如果没有指定文件，选择加载最新的模型
    if not f:
        if epoch > 0:
            f = os.path.join(exp_path, exp_name + '-%09d' % epoch + '.pth')
            assert os.path.isfile(f)
        else:
            f = sorted(glob.glob(os.path.join(exp_path, exp_name + '-*.pth')))
            if len(f) > 0:
                f = f[-1]
                epoch = int(f[len(exp_path) + len(exp_name) + 2: -4])

    if len(f) > 0:
        logger.info('Restore from ' + f)
        checkpoint = torch.load(f)

        # 去掉'module.'前缀（如果存在）
        for k, v in checkpoint.items():
            if 'module.' in k:
                checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
            break

        # 加载模型1的权重，strict=False 允许部分不匹配
        try:
            if dist:
                model.module.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        except RuntimeError as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise RuntimeError(f"Failed to load checkpoint {f}. Some layers were not loaded.")

    # 如果第二个模型路径提供了，我们使用它填充缺失的权重
    if second_model_path is not None:
        # 加载第二个模型的权重
        logger.info(f"Loading missing parameters from the second model: {second_model_path}")
        second_checkpoint = torch.load(second_model_path)

        # 打印出 second_checkpoint 的内容，查看有哪些键
        logger.info(f"Keys in second checkpoint: {second_checkpoint.keys()}")

        # 根据第二个模型的结构调整这里
        # 比如，如果第二个模型权重存储在 'state_dict' 键下，则使用： second_state_dict = second_checkpoint['state_dict']
        second_state_dict = second_checkpoint  # 假设第二个模型没有嵌套，需要直接使用 second_checkpoint

        # 获取第一个模型的state_dict
        model_state_dict = model.state_dict()

        # 筛选缺失的参数，并用第二个模型填充
        missing_params = {k: v for k, v in second_state_dict.items() if k not in model_state_dict}
        for name, param in missing_params.items():
            if name in model_state_dict:
                model_state_dict[name] = param
            else:
                print(f"Layer {name} is not in the model, skipping.")

        # 更新模型的state_dict
        model.load_state_dict(model_state_dict)
        logger.info("Successfully loaded missing parameters from the second model.")

    # 将模型转到GPU（如果需要）
    if use_cuda:
        model.cuda()

    return epoch + 1


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def is_multiple(num, multiple):
    return num != 0 and num % multiple == 0


def checkpoint_save(model, exp_path, exp_name, epoch, save_freq=16, use_cuda=True):
    f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
    logger.info('Saving ' + f)
    model.cpu()
    torch.save(model.state_dict(), f)
    if use_cuda:
        model.cuda()

    #remove previous checkpoints unless they are a power of 2 or a multiple of 16 to save disk space
    epoch = epoch - 1
    f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
    if os.path.isfile(f):
        if not is_multiple(epoch, save_freq) and not is_power2(epoch):
            os.remove(f)


def load_model_param(model, pretrained_dict, prefix=""):
    # suppose every param in model should exist in pretrain_dict, but may differ in the prefix of the name
    # For example:    model_dict: "0.conv.weight"     pretrain_dict: "FC_layer.0.conv.weight"
    model_dict = model.state_dict()
    len_prefix = 0 if len(prefix) == 0 else len(prefix) + 1
    pretrained_dict_filter = {k[len_prefix:]: v for k, v in pretrained_dict.items() if k[len_prefix:] in model_dict and prefix in k}
    assert len(pretrained_dict_filter) > 0
    model_dict.update(pretrained_dict_filter)
    model.load_state_dict(model_dict)
    return len(pretrained_dict_filter), len(model_dict)


def write_obj(points, colors, out_filename):
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()


def get_batch_offsets(batch_idxs, bs):
    '''
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    '''
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def get_merged_proposal_labels(proposal_idx):
    '''
    :param proposal_idx: (N_, ), int
    :return: remap the proposal idx remapper from 0 to nproposal - 1
    '''
    unique_proposal_idx = torch.unique(proposal_idx)
    nproposal = unique_proposal_idx.size(0)
    for i in range(nproposal):
        proposal_idx[proposal_idx == unique_proposal_idx[i]] = i
    return proposal_idx, nproposal


def print_error(message, user_fault=False):
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    if user_fault:
      sys.exit(2)
    sys.exit(-1)


def dice_loss_multi_classes(input: torch.Tensor,
                            target: torch.Tensor,
                            epsilon: float = 1e-5,
                            weight: Optional[float] = None) -> torch.Tensor:
    r"""
    cite from dknet, which
    modify compute_per_channel_dice from https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py
    """
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # convert the feature channel(category channel) as first
    axis_order = (1, 0) + tuple(range(2, input.dim()))
    input = input.permute(axis_order)
    target = target.permute(axis_order)

    target = target.float()
    # Compute per channel Dice Coefficient
    per_channel_dice = (2 * torch.sum(input * target, dim=1) + epsilon) / \
                       (torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4 + epsilon)

    loss = 1. - per_channel_dice

    return loss
