""" Checkpoint Saver

Track top-n training checkpoints

Hacked together by / Copyright 2020 Ross Wightman
"""

import glob
import operator
import os
import logging
import shutil
import torch

from .model import unwrap_model, get_state_dict

_logger = logging.getLogger(__name__)


class CheckpointSaver:
    def __init__(
            self,
            model,
            optimizer,
            scaler,
            args=None,
            checkpoint_prefix='checkpoint',
            checkpoint_dir='',
            decreasing=False,
            max_history=10,
            unwrap_fn=unwrap_model):

        # objects to save state_dicts of
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.scaler = scaler

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None

        # config
        self.checkpoint_dir = checkpoint_dir
        self.save_prefix = checkpoint_prefix
        self.extension = '.pth.tar'
        self.decreasing = decreasing  # a lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt  # True if lhs better than rhs
        self.max_history = max_history
        self.unwrap_fn = unwrap_fn
        assert self.max_history >= 1

    def save(self, epoch, metric=None):
        save_path = os.path.join(self.checkpoint_dir, str(epoch) + self.extension)
        self._save(save_path, epoch, metric)

    def save_checkpoint(self, epoch, metric=None):
        assert epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
        last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
        self._save(tmp_save_path, epoch, metric)  # 先保存到临时文件

        # 重命名 tmp -> last
        if os.path.exists(last_save_path):
            os.unlink(last_save_path)
        os.rename(tmp_save_path, last_save_path)

        # 仅在需要保留当前 epoch 时，才创建 checkpoint-<epoch> 并记录
        should_save = (len(self.checkpoint_files) < self.max_history) or (metric is not None and self.cmp(metric, self.checkpoint_files[-1][1]))
        if should_save:
            filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            shutil.copy2(last_save_path, save_path)  # 复制 last 到 checkpoint 文件
            self.checkpoint_files.append((save_path, metric))
            # 排序并清理历史
            self.checkpoint_files = sorted(
                self.checkpoint_files, key=lambda x: x[1],
                reverse=not self.decreasing)
            if len(self.checkpoint_files) > self.max_history:
                self._cleanup_checkpoints(len(self.checkpoint_files) - self.max_history)

            # 记录最佳模型
            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                best_save_path = os.path.join(self.checkpoint_dir, 'model_best' + self.extension)
                if os.path.exists(best_save_path):
                    os.unlink(best_save_path)
                shutil.copy2(last_save_path, best_save_path)

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _save(self, save_path, epoch, metric=None):
        save_state = {
            'epoch': epoch,
            'arch': type(self.model).__name__.lower(),
            'state_dict': get_state_dict(self.model, self.unwrap_fn),
            'optimizer': self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            'version': 2,  # version < 2 increments epoch before save
        }
        if self.args is not None:
            save_state['args'] = self.args
        if metric is not None:
            save_state['metric'] = metric
        torch.save(save_state, save_path)

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index < 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                _logger.debug("Cleaning checkpoint: {}".format(d))
                os.remove(d[0])
            except Exception as e:
                _logger.error("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]