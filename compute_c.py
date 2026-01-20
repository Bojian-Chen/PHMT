import argparse
import copy
import csv
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from utils.set import set_model, set_dataset, set_optimizer, set_random_seed
from utils.Fisher import Fisher, Fisher_BN

from trainer.source_train import source_train
from trainer.shot import shot
from trainer.norm import norm
from trainer.PesudoLabel import PesudoLabel
from trainer.tent import tent
from trainer.gsfda import gsfda_source_train, gsfda
from trainer.UCSN import UCSN_source_train, UCSN
from trainer.CoTTA import CoTTA
from trainer.CoSDA import CoSDA
from trainer.ours import ours
from trainer.ours_sc import ours_sc
from trainer.RaTP import RaTP
from trainer.EATA import EATA
from trainer.AFSFFD import AFSFFD
from trainer.ours_simple import ours_simple
from trainer.rmt import rmt


ALL_MODES = [
    "norm",
    "PesudoLabel",
    "Tent",
    "shot",
    "CoTTA",
    "gsfda",
    "UCSN",
    "CoSDA",
    "ours",
    "ours_sc",
    "ours_simple",
    "RaTP",
    "EATA",
    "AFSFFD",
    "rmt",
]


@dataclass
class OverheadResult:
    params_bytes: int
    buffers_bytes: int
    activation_bytes_sum: int
    activation_bytes_max: int
    macs: int

    @property
    def flops(self) -> int:
        # FLOPs ~= 2 * MACs (mul + add)
        return self.macs * 2


@dataclass
class TrainOverheadResult:
    total_forward_macs: int
    train_forward_macs: int
    total_forward_flops: int
    train_forward_flops: int
    total_flops: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    latency_ms: float
    throughput: float
    timed_batches: int
    updated_params_per_session: list
    updated_params_total: int
    total_train_time_sec: float
    bn_params_total: int
    bn_buffers_total: int
    bn_buffers_per_session: list


def _kernel_numel(kernel_size):
    if isinstance(kernel_size, tuple):
        total = 1
        for k in kernel_size:
            total *= k
        return total
    return int(kernel_size)


def _count_conv(module: nn.Module, output: torch.Tensor) -> int:
    kernel_mul = _kernel_numel(module.kernel_size)
    cin_per_group = module.in_channels // module.groups
    return int(output.numel() * cin_per_group * kernel_mul)


def _count_linear(module: nn.Module, output: torch.Tensor) -> int:
    return int(output.numel() * module.in_features)


def _bytes_for_tensor(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _extract_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    return None


def _make_model(args):
    backbone, classifier = set_model(args)

    class ModelWrapper(nn.Module):
        def __init__(self, backbone, classifier, use_classifier):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier
            self.use_classifier = use_classifier

        def forward(self, x):
            if self._is_embedding_backbone():
                features = self.backbone(x, t=0, s=100, all_mask=False)
                if isinstance(features, (tuple, list)):
                    features = features[0]
            else:
                features = self.backbone(x)
            if self.use_classifier:
                return self.classifier(features)
            return features

        def _is_embedding_backbone(self):
            return self.backbone.__class__.__name__ == "backbone_embedding"

    return ModelWrapper(backbone, classifier, args.include_classifier)


def _track_overhead(model, inputs, include_bn=False, include_relu=False, include_pool=False):
    macs = 0
    activation_sum = 0
    activation_max = 0

    def hook_fn(module, _inputs, output):
        nonlocal macs, activation_sum, activation_max
        tensor_out = _extract_tensor(output)
        if tensor_out is None:
            return

        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            macs += _count_conv(module, tensor_out)
        elif isinstance(module, nn.Linear):
            macs += _count_linear(module, tensor_out)
        elif include_bn and isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            macs += int(tensor_out.numel())
        elif include_relu and isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
            macs += int(tensor_out.numel())
        elif include_pool and isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                                                 nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                                                 nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)):
            macs += int(tensor_out.numel())

        out_bytes = _bytes_for_tensor(tensor_out)
        activation_sum += out_bytes
        if out_bytes > activation_max:
            activation_max = out_bytes

    hooks = []
    for module in model.modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        _ = model(inputs)

    for h in hooks:
        h.remove()

    params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    return OverheadResult(
        params_bytes=params_bytes,
        buffers_bytes=buffers_bytes,
        activation_bytes_sum=activation_sum,
        activation_bytes_max=activation_max,
        macs=macs,
    )


class FlopsCounter:
    def __init__(self, include_bn=False, include_relu=False, include_pool=False, batch_tracker=None):
        self.include_bn = include_bn
        self.include_relu = include_relu
        self.include_pool = include_pool
        self.macs = 0
        self.batch_tracker = batch_tracker

    def add_hooks(self, model: nn.Module):
        hooks = []
        for module in model.modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(self._hook))
        return hooks

    def _hook(self, module, _inputs, output):
        tensor_out = _extract_tensor(output)
        if tensor_out is None:
            return

        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            macs = _count_conv(module, tensor_out)
        elif isinstance(module, nn.Linear):
            macs = _count_linear(module, tensor_out)
        elif self.include_bn and isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            macs = int(tensor_out.numel())
        elif self.include_relu and isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
            macs = int(tensor_out.numel())
        elif self.include_pool and isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                                                       nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                                                       nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)):
            macs = int(tensor_out.numel())
        else:
            macs = 0

        if macs > 0:
            self.macs += macs
            if self.batch_tracker is not None:
                self.batch_tracker.add_macs(macs)


class BatchTracker:
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        self.current_macs = 0
        self.current_has_step = False
        self.train_forward_macs = 0
        self.total_forward_macs = 0
        self.batch_start = None
        self.last_step = None
        self.pending_batch_size = None
        self.latencies = []
        self.samples = 0

    def _now(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def _infer_batch_size(self, batch):
        if isinstance(batch, (list, tuple)) and batch:
            first = batch[0]
            if torch.is_tensor(first):
                return int(first.shape[0])
        if torch.is_tensor(batch):
            return int(batch.shape[0])
        return 0

    def add_macs(self, macs):
        self.current_macs += macs
        self.total_forward_macs += macs

    def on_batch_start(self, batch):
        if self.batch_start is not None and self.current_has_step and self.last_step is not None:
            duration = self.last_step - self.batch_start
            if duration > 0 and self.pending_batch_size:
                self.latencies.append(duration)
                self.samples += self.pending_batch_size
            self.train_forward_macs += self.current_macs

        self.batch_start = self._now()
        self.last_step = None
        self.current_macs = 0
        self.current_has_step = False
        self.pending_batch_size = self._infer_batch_size(batch)

    def on_step(self):
        self.last_step = self._now()
        self.current_has_step = True

    def flush(self):
        if self.batch_start is not None and self.current_has_step and self.last_step is not None:
            duration = self.last_step - self.batch_start
            if duration > 0 and self.pending_batch_size:
                self.latencies.append(duration)
                self.samples += self.pending_batch_size
            self.train_forward_macs += self.current_macs
        self.batch_start = None
        self.last_step = None
        self.pending_batch_size = None
        self.current_macs = 0
        self.current_has_step = False

    def latency_ms(self):
        if not self.latencies:
            return 0.0
        return (sum(self.latencies) / len(self.latencies)) * 1000.0

    def throughput(self):
        if not self.latencies:
            return 0.0
        total_time = sum(self.latencies)
        if total_time <= 0:
            return 0.0
        return float(self.samples) / total_time


class TimedDataLoader:
    def __init__(self, loader, tracker):
        self.loader = loader
        self.tracker = tracker

    def __iter__(self):
        for batch in self.loader:
            self.tracker.on_batch_start(batch)
            yield batch

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, item):
        return getattr(self.loader, item)


def _format_bytes(num_bytes):
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def _parse_domain_seq(value):
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return np.array([int(p) for p in parts], dtype=int)


def _apply_mode_settings(args):
    if args.incremental_mode in ("PesudoLabel", "Tent", "EATA"):
        args.train_parames = "BN"

    if args.incremental_mode == "shot":
        args.classifer = "fcwn"
        args.LabelSmooth = True

    if args.incremental_mode == "ours_simple":
        args.classifer = "cos"
        args.PCL = True
        args.LabelSmooth = False
        args.contrastive_loss = False

    if args.incremental_mode == "ours_sc":
        args.classifer = "cos"
        args.PCL = False
        args.LabelSmooth = False
        args.contrastive_loss = False

    if args.incremental_mode == "gsfda":
        args.classifer = "fcwn"

    if args.incremental_mode == "UCSN":
        args.classifer = "fc"

    if args.incremental_mode == "CoSDA":
        args.classifer = "fcwn"

    if args.incremental_mode == "ours":
        args.classifer = "cos"
        args.PCL = False
        args.LabelSmooth = False
        args.contrastive_loss = False

    if args.incremental_mode == "RaTP":
        args.classifer = "cos"
        args.PCL = True
        args.RandMix = True


def _apply_backbone_settings(args):
    if args.backbone_name in ("cnn", "resnet18_1D"):
        args.data_dimension = "1D"
        args.data_mode = "Time"
    else:
        args.data_dimension = "2D"
        args.data_mode = "Frequence"


def _apply_dataset_settings(args):
    if args.dataset_name == "SK":
        args.train_list = "./SK_all_10classes.mat"
        args.test_list = "./SK_all_10classes.mat"
        if args.Domain_Seq is None:
            args.Domain_Seq = np.array([6, 1, 8, 15, 22, 17])
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 10
    elif args.dataset_name == "SK_new":
        args.train_list = "./SK_new_all_10classes.mat"
        args.test_list = "./SK_new_all_10classes.mat"
        if args.Domain_Seq is None:
            args.Domain_Seq = np.array([6, 1, 8, 15, 22, 17])
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 10
    elif args.dataset_name == "HUST":
        args.train_list = "./HUST_Bearings_10domains_9classes.mat"
        args.test_list = "./HUST_Bearings_10domains_9classes.mat"
        if args.Domain_Seq is None:
            args.Domain_Seq = np.array([5, 6, 7, 8])
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 9
    elif args.dataset_name == "HUST_gear":
        args.train_list = "./HUST_Gearbox_25domains_3classes.mat"
        args.test_list = "./HUST_Gearbox_25domains_3classes.mat"
        if args.Domain_Seq is None:
            args.Domain_Seq = np.array([0, 6, 12, 18, 24])
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 3
    elif args.dataset_name == "iFlytek":
        args.batch_size = 64
        args.train_list = "./iFlytek_all_5classes.mat"
        args.test_list = "./iFlytek_all_5classes.mat"
        if args.Domain_Seq is None:
            args.Domain_Seq = np.array([2, 3, 4, 5, 7])
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 5
    elif args.dataset_name == "WT":
        args.batch_size = 128
        args.train_list = "./WT_all_5classes.mat"
        args.test_list = "./WT_all_5classes.mat"
        if args.Domain_Seq is None:
            args.Domain_Seq = np.array([0, 1, 2, 3, 4])
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 5
    elif args.dataset_name == "PU_Real":
        args.train_list = "./PU_Real_4doamins_5classes.mat"
        args.test_list = "./PU_Real_4doamins_5classes.mat"
        if args.Domain_Seq is None:
            args.Domain_Seq = np.array([3, 2, 0, 1])
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 5
    elif args.dataset_name == "PU_Real_200":
        args.train_list = "./PU_Real_4doamins_5classes_200.mat"
        args.test_list = "./PU_Real_4doamins_5classes_200.mat"
        if args.Domain_Seq is None:
            args.Domain_Seq = np.array([3, 2, 0, 1])
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 5
        args.data_num_class = 200
    elif args.dataset_name == "PU_Art":
        args.train_list = "./PU_Art_4doamins_8classes.mat"
        args.test_list = "./PU_Art_4doamins_8classes.mat"
        if args.Domain_Seq is None:
            args.Domain_Seq = np.array([2, 0])
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 8
    elif args.dataset_name == "Robot":
        args.train_list = "./Robot_4classes_4domains.mat"
        args.test_list = "./Robot_4classes_4domains.mat"
        if args.Domain_Seq is None:
            args.Domain_Seq = np.array([0, 1, 2, 3])
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 4
    else:
        raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")


def _attach_hooks(counter, *models):
    hooks = []
    for model in models:
        hooks.extend(counter.add_hooks(model))
    return hooks


def _wrap_optimizer(optimizer, tracker):
    if optimizer is None:
        return None
    optimizer._stepped = False
    original_step = optimizer.step

    def step_with_tracking(*args, **kwargs):
        result = original_step(*args, **kwargs)
        optimizer._stepped = True
        tracker.on_step()
        return result

    optimizer.step = step_with_tracking
    return optimizer


def _count_updated_params(*optimizers):
    param_ids = set()
    total = 0
    for opt in optimizers:
        if opt is None or not getattr(opt, "_stepped", False):
            continue
        for group in opt.param_groups:
            for p in group.get("params", []):
                pid = id(p)
                if pid not in param_ids:
                    param_ids.add(pid)
                    total += p.numel()
    return total


def _collect_param_ids(param_map, *optimizers):
    for opt in optimizers:
        if opt is None or not getattr(opt, "_stepped", False):
            continue
        for group in opt.param_groups:
            for p in group.get("params", []):
                pid = id(p)
                if pid not in param_map:
                    param_map[pid] = p.numel()


def _count_bn_params(model):
    total = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.weight is not None:
                total += m.weight.numel()
            if m.bias is not None:
                total += m.bias.numel()
    return total


def _count_bn_buffers(model):
    total = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.running_mean is not None:
                total += m.running_mean.numel()
            if m.running_var is not None:
                total += m.running_var.numel()
    return total


def _train_with_overhead(args, counter, tracker):
    backbone, classifier = set_model(args)
    trainloader_list, testloader_list = set_dataset(args)

    trainloader_list = [TimedDataLoader(loader, tracker) for loader in trainloader_list]
    testloader_list = [TimedDataLoader(loader, tracker) for loader in testloader_list]

    hooks = _attach_hooks(counter, backbone, classifier)
    updated_params_per_session = []
    updated_param_map = {}
    bn_buffers_per_session = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_start = time.perf_counter()

    for session in range(args.nb_session):
        print(f"session: {session}")
        train_loader = trainloader_list[session]
        test_loader = testloader_list[session]

        backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler = set_optimizer(
            args, backbone, classifier, session
        )
        backbone_optimizer = _wrap_optimizer(backbone_optimizer, tracker)
        classifier_optimizer = _wrap_optimizer(classifier_optimizer, tracker)

        if session == 0:
            if args.incremental_mode == "gsfda":
                backbone, classifier = gsfda_source_train(
                    args, backbone, classifier, train_loader, test_loader,
                    backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler
                )
            elif args.incremental_mode == "UCSN":
                backbone, classifier = UCSN_source_train(
                    args, backbone, classifier, train_loader, test_loader,
                    backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler
                )
            else:
                backbone, classifier = source_train(
                    args, backbone, classifier, train_loader, test_loader,
                    backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler
                )

            source_backbone = copy.deepcopy(backbone)
            source_classifier = copy.deepcopy(classifier)

            if args.incremental_mode == "EATA":
                fisher_hooks = _attach_hooks(counter, source_backbone, source_classifier)
                fishers = Fisher_BN(source_backbone, source_classifier, train_loader)
                for h in fisher_hooks:
                    h.remove()
            elif args.incremental_mode == "AFSFFD":
                fisher_hooks = _attach_hooks(counter, source_backbone, source_classifier)
                fishers = Fisher(source_backbone, source_classifier, train_loader)
                for h in fisher_hooks:
                    h.remove()
            else:
                fishers = None

        else:
            if args.incremental_mode == "norm":
                backbone, classifier = norm(args, backbone, classifier, train_loader, test_loader)

            elif args.incremental_mode == "PesudoLabel":
                backbone, classifier = PesudoLabel(
                    args, backbone, classifier, train_loader, test_loader,
                    backbone_optimizer, backbone_scheduler
                )

            elif args.incremental_mode == "Tent":
                backbone, classifier = tent(
                    args, backbone, classifier, train_loader, test_loader,
                    backbone_optimizer, backbone_scheduler
                )

            elif args.incremental_mode == "EATA":
                backbone, classifier = EATA(
                    args, backbone, classifier, train_loader, test_loader,
                    backbone_optimizer, backbone_scheduler, fishers
                )

            elif args.incremental_mode == "AFSFFD":
                backbone, classifier, fishers = AFSFFD(
                    args, backbone, classifier, train_loader, test_loader,
                    backbone_optimizer, backbone_scheduler, fishers
                )

            elif args.incremental_mode == "shot":
                backbone, classifier = shot(
                    args, backbone, classifier, train_loader, test_loader,
                    backbone_optimizer, backbone_scheduler
                )

            elif args.incremental_mode == "gsfda":
                backbone, classifier = gsfda(
                    args, backbone, classifier, train_loader, test_loader,
                    backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler
                )

            elif args.incremental_mode == "UCSN":
                backbone, classifier = UCSN(
                    args, backbone, classifier, train_loader, test_loader,
                    backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler
                )

            elif args.incremental_mode == "CoTTA":
                teacher_backbone = copy.deepcopy(backbone)
                teacher_classifier = copy.deepcopy(classifier)
                teacher_hooks = _attach_hooks(counter, teacher_backbone, teacher_classifier)
                backbone, classifier = CoTTA(
                    args, teacher_backbone, teacher_classifier, backbone, classifier,
                    train_loader, test_loader, backbone_optimizer, classifier_optimizer,
                    backbone_scheduler, classifier_scheduler
                )
                for h in teacher_hooks:
                    h.remove()

            elif args.incremental_mode == "CoSDA":
                teacher_backbone = copy.deepcopy(backbone)
                teacher_classifier = copy.deepcopy(classifier)
                teacher_hooks = _attach_hooks(counter, teacher_backbone, teacher_classifier)
                backbone, classifier = CoSDA(
                    args, teacher_backbone, teacher_classifier, backbone, classifier,
                    train_loader, test_loader, backbone_optimizer, classifier_optimizer,
                    backbone_scheduler, classifier_scheduler
                )
                for h in teacher_hooks:
                    h.remove()

            elif args.incremental_mode == "ours":
                teacher_backbone = copy.deepcopy(backbone)
                teacher_classifier = copy.deepcopy(classifier)
                teacher_hooks = _attach_hooks(counter, teacher_backbone, teacher_classifier)
                backbone, classifier = ours(
                    args, teacher_backbone, teacher_classifier, backbone, classifier,
                    train_loader, test_loader, backbone_optimizer, classifier_optimizer,
                    backbone_scheduler, classifier_scheduler
                )
                for h in teacher_hooks:
                    h.remove()

            elif args.incremental_mode == "ours_sc":
                teacher_backbone = copy.deepcopy(backbone)
                teacher_classifier = copy.deepcopy(classifier)
                teacher_hooks = _attach_hooks(counter, teacher_backbone, teacher_classifier)
                backbone, classifier = ours_sc(
                    args, teacher_backbone, teacher_classifier, backbone, classifier,
                    train_loader, test_loader, backbone_optimizer, classifier_optimizer,
                    backbone_scheduler, classifier_scheduler
                )
                for h in teacher_hooks:
                    h.remove()

            elif args.incremental_mode == "ours_simple":
                teacher_backbone = copy.deepcopy(backbone)
                teacher_classifier = copy.deepcopy(classifier)
                teacher_hooks = _attach_hooks(counter, teacher_backbone, teacher_classifier)
                backbone, classifier = ours_simple(
                    args, teacher_backbone, teacher_classifier, backbone, classifier,
                    source_backbone, train_loader, test_loader, backbone_optimizer, backbone_scheduler
                )
                for h in teacher_hooks:
                    h.remove()

            elif args.incremental_mode == "RaTP":
                teacher_backbone = copy.deepcopy(backbone)
                teacher_classifier = copy.deepcopy(classifier)
                teacher_hooks = _attach_hooks(counter, teacher_backbone, teacher_classifier)
                backbone, classifier = RaTP(
                    args, teacher_backbone, teacher_classifier, backbone, classifier,
                    train_loader, test_loader, backbone_optimizer, classifier_optimizer,
                    backbone_scheduler, classifier_scheduler
                )
                for h in teacher_hooks:
                    h.remove()

            elif args.incremental_mode == "rmt":
                teacher_backbone = copy.deepcopy(backbone)
                teacher_classifier = copy.deepcopy(classifier)
                teacher_hooks = _attach_hooks(counter, teacher_backbone, teacher_classifier)
                backbone, classifier = rmt(
                    args, teacher_backbone, teacher_classifier, backbone, classifier,
                    source_classifier, train_loader, test_loader, backbone_optimizer,
                    classifier_optimizer, backbone_scheduler, classifier_scheduler
                )
                for h in teacher_hooks:
                    h.remove()

        tracker.flush()
        updated_params = _count_updated_params(backbone_optimizer, classifier_optimizer)
        _collect_param_ids(updated_param_map, backbone_optimizer, classifier_optimizer)
        bn_buffers_count = _count_bn_buffers(backbone) + _count_bn_buffers(classifier)
        bn_buffers_per_session.append(bn_buffers_count)
        updated_params_per_session.append(updated_params + bn_buffers_count)

        for h in hooks:
            h.remove()
        hooks = _attach_hooks(counter, backbone, classifier)

    for h in hooks:
        h.remove()

    total_forward_macs = tracker.total_forward_macs
    train_forward_macs = tracker.train_forward_macs
    total_forward_flops = total_forward_macs * 2
    train_forward_flops = train_forward_macs * 2
    total_flops = int(train_forward_flops * (1.0 + args.backward_multiplier) + (total_forward_flops - train_forward_flops))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_end = time.perf_counter()
    total_train_time_sec = train_end - train_start

    if torch.cuda.is_available():
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    else:
        peak_allocated = 0
        peak_reserved = 0

    bn_buffers_total = _count_bn_buffers(backbone) + _count_bn_buffers(classifier)
    updated_params_total = sum(updated_param_map.values()) + bn_buffers_total

    return TrainOverheadResult(
        total_forward_macs=total_forward_macs,
        train_forward_macs=train_forward_macs,
        total_forward_flops=total_forward_flops,
        train_forward_flops=train_forward_flops,
        total_flops=total_flops,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        latency_ms=tracker.latency_ms(),
        throughput=tracker.throughput(),
        timed_batches=len(tracker.latencies),
        updated_params_per_session=updated_params_per_session,
        updated_params_total=updated_params_total,
        total_train_time_sec=total_train_time_sec,
        bn_params_total=_count_bn_params(backbone) + _count_bn_params(classifier),
        bn_buffers_total=bn_buffers_total,
        bn_buffers_per_session=bn_buffers_per_session,
    )


def _run_train_overhead(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training overhead measurement in this codebase.")

    modes = args.modes
    if modes == "all":
        mode_list = ALL_MODES
    else:
        raw_modes = [m.strip() for m in modes.split(",") if m.strip()]
        lookup = {m.lower(): m for m in ALL_MODES}
        mode_list = []
        for m in raw_modes:
            key = m.lower()
            if key not in lookup:
                raise ValueError(f"Unsupported mode: {m}")
            mode_list.append(lookup[key])

    results = {}

    for mode in mode_list:
        mode_args = copy.deepcopy(args)
        mode_args.incremental_mode = mode
        _apply_mode_settings(mode_args)
        _apply_backbone_settings(mode_args)
        _apply_dataset_settings(mode_args)

        set_random_seed(mode_args.random_seed)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        tracker = BatchTracker(use_cuda=True)
        counter = FlopsCounter(
            include_bn=mode_args.count_bn,
            include_relu=mode_args.count_relu,
            include_pool=mode_args.count_pool,
            batch_tracker=tracker,
        )

        result = _train_with_overhead(mode_args, counter, tracker)
        results[mode] = result

        print(f"=== Mode: {mode} ===")
        print(f"Total forward MACs: {result.total_forward_macs:,}")
        print(f"Train forward MACs: {result.train_forward_macs:,}")
        print(f"Total forward FLOPs: {result.total_forward_flops:,}")
        print(f"Train forward FLOPs: {result.train_forward_flops:,}")
        print(f"Total FLOPs (forward + backward): {result.total_flops:,}")
        print(f"Peak allocated: {_format_bytes(result.peak_allocated_bytes)}")
        print(f"Peak reserved: {_format_bytes(result.peak_reserved_bytes)}")
        if result.timed_batches > 0:
            print(f"Latency (ms/batch): {result.latency_ms:.3f}")
            print(f"Throughput (samples/s): {result.throughput:.2f}")
            print(f"Timed batches: {result.timed_batches}")
        else:
            print("Latency (ms/batch): N/A")
            print("Throughput (samples/s): N/A")
            print("Timed batches: 0")
        print(f"Updated state per session (params + BN buffers): {result.updated_params_per_session}")
        print(f"Updated state (total unique params + BN buffers): {result.updated_params_total:,}")
        print(f"BN params (total): {result.bn_params_total:,}")
        print(f"BN buffers (total): {result.bn_buffers_total:,}")
        print(f"BN buffers per session: {result.bn_buffers_per_session}")
        print(f"Total train time (all sessions): {result.total_train_time_sec:.2f}s")
        print(f"Backward multiplier: {mode_args.backward_multiplier}")
        print("")

    if args.save_csv:
        _write_csv(args.save_csv, results)
        print(f"CSV saved: {args.save_csv}")

    return results


def _write_csv(path, results):
    headers = [
        "mode",
        "total_forward_macs",
        "train_forward_macs",
        "total_forward_flops",
        "train_forward_flops",
        "total_flops",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "latency_ms",
        "throughput_samples_s",
        "timed_batches",
        "updated_state_total",
        "updated_state_per_session",
        "bn_params_total",
        "bn_buffers_total",
        "bn_buffers_per_session",
        "total_train_time_sec",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for mode, result in results.items():
            writer.writerow([
                mode,
                result.total_forward_macs,
                result.train_forward_macs,
                result.total_forward_flops,
                result.train_forward_flops,
                result.total_flops,
                result.peak_allocated_bytes,
                result.peak_reserved_bytes,
                f"{result.latency_ms:.6f}",
                f"{result.throughput:.6f}",
                result.timed_batches,
                result.updated_params_total,
                ",".join(str(x) for x in result.updated_params_per_session),
                result.bn_params_total,
                result.bn_buffers_total,
                ",".join(str(x) for x in result.bn_buffers_per_session),
                f"{result.total_train_time_sec:.6f}",
            ])


def _run_static_overhead(args):
    args.include_classifier = not args.no_classifier
    model = _make_model(args).to(args.device)

    if args.backbone_name in ("cnn", "resnet18_1D"):
        inputs = torch.randn(args.batch_size, 1, args.input_length, device=args.device)
    else:
        inputs = torch.randn(args.batch_size, 1, args.input_size, args.input_size, device=args.device)

    result = _track_overhead(
        model,
        inputs,
        include_bn=args.count_bn,
        include_relu=args.count_relu,
        include_pool=args.count_pool,
    )

    print("=== Computational and Memory Overhead (Static) ===")
    print(f"MACs: {result.macs:,}")
    print(f"FLOPs (approx): {result.flops:,}")
    print(f"Params memory: {_format_bytes(result.params_bytes)}")
    print(f"Buffers memory: {_format_bytes(result.buffers_bytes)}")
    print(f"Activation memory (sum): {_format_bytes(result.activation_bytes_sum)}")
    print(f"Activation memory (max module output): {_format_bytes(result.activation_bytes_max)}")
    print("Note: FLOPs count includes Conv/Linear only by default.")


def main():
    parser = argparse.ArgumentParser(description="Compute computational and memory overhead.")
    parser.add_argument("--mode", default="static", choices=["static", "train"])

    parser.add_argument("--backbone_name", default="resnet14", type=str,
                        choices=["resnet14", "resnet32", "resnet18_1D", "cnn"])
    parser.add_argument("--classifer", default="cos", type=str,
                        choices=["fc", "cos", "eu", "fcwn"])
    parser.add_argument("--incremental_mode", default="ours_simple", type=str)
    parser.add_argument("--nb_cl", default=5, type=int, help="number of classes")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size for training or dummy input")
    parser.add_argument("--device", default="cpu", type=str, choices=["cpu", "cuda"])
    parser.add_argument("--no_classifier", action="store_true", help="exclude classifier from overhead")
    parser.add_argument("--count_bn", action="store_true", help="count BN FLOPs")
    parser.add_argument("--count_relu", action="store_true", help="count ReLU FLOPs")
    parser.add_argument("--count_pool", action="store_true", help="count Pool FLOPs")
    parser.add_argument("--input_length", default=1024, type=int, help="1D input length")
    parser.add_argument("--input_size", default=32, type=int, help="2D input H/W")

    parser.add_argument("--modes", default="all", type=str, help="comma list or 'all'")
    parser.add_argument("--dataset_name", default="SK", type=str)
    parser.add_argument("--domain_seq", default=None, type=str, help="comma list, e.g. 6,1,8,15,22,17")
    parser.add_argument("--data_num_class", default=100, type=int)
    parser.add_argument("--test_batch_size", default=100, type=int)
    parser.add_argument("--random_seed", default=2025, type=int)
    parser.add_argument("--train_parames", default="default", type=str)
    parser.add_argument("--preprocess", default="zscore", type=str)
    parser.add_argument("--eta_min", default=0.001, type=float)

    parser.add_argument("--contrastive_loss", action="store_true")
    parser.add_argument("--LabelSmooth", action="store_true")
    parser.add_argument("--PCL", action="store_true")
    parser.add_argument("--base_epochs", default=40, type=int)
    parser.add_argument("--base_lr", default=0.1, type=float)
    parser.add_argument("--RandMix", action="store_true")

    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr", default=0.1, type=float)

    parser.add_argument("--dataroot", default="./data/", type=str)
    parser.add_argument("--save_model", action="store_true")

    parser.add_argument("--TOPK", action="store_false")
    parser.add_argument("--PCA", action="store_false")
    parser.add_argument("--MI", action="store_false")
    parser.add_argument("--SR", action="store_false")
    parser.add_argument("--mixup", action="store_false")

    parser.add_argument("--backward_multiplier", default=2.0, type=float)
    parser.add_argument("--save_csv", default="overhead_results.csv", type=str)

    args = parser.parse_args()
    args.Domain_Seq = _parse_domain_seq(args.domain_seq)

    if args.mode == "static":
        _run_static_overhead(args)
    else:
        _run_train_overhead(args)


if __name__ == "__main__":
    main()
