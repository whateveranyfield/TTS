import json
import torch
import logging
import argparse
import numpy as np

from pathlib import Path
from os.path import join
from optimizers.sgd import SGD
from utils import config_argparse
from utils.types import str2bool
from utils.types import str_or_none
from utils.types import int_or_none
from utils.types import str2triple_str
from schedulers.noam_lr import NoamLR
from schedulers.warmup_lr import WarmupLR
from distutils.version import LooseVersion
from tts.fastspeech2 import FastSpeech2


optim_classes = dict(
    adam=torch.optim.Adam,
    adamw=torch.optim.AdamW,
    sgd=SGD,
    adadelta=torch.optim.Adadelta,
    adagrad=torch.optim.Adagrad,
    adamax=torch.optim.Adamax,
    asgd=torch.optim.ASGD,
    lbfgs=torch.optim.LBFGS,
    rmsprop=torch.optim.RMSprop,
    rprop=torch.optim.Rprop,
)
if LooseVersion(torch.__version__) >= LooseVersion("1.10.0"):
    # From 1.10.0, RAdam is officially supported
    optim_classes.update(
        radam=torch.optim.RAdam,
    )
try:
    import torch_optimizer

    optim_classes.update(
        accagd=torch_optimizer.AccSGD,
        adabound=torch_optimizer.AdaBound,
        adamod=torch_optimizer.AdaMod,
        diffgrad=torch_optimizer.DiffGrad,
        lamb=torch_optimizer.Lamb,
        novograd=torch_optimizer.NovoGrad,
        pid=torch_optimizer.PID,
        # torch_optimizer<=0.0.1a10 doesn't support
        # qhadam=torch_optimizer.QHAdam,
        qhm=torch_optimizer.QHM,
        sgdw=torch_optimizer.SGDW,
        yogi=torch_optimizer.Yogi,
    )
    if LooseVersion(torch_optimizer.__version__) < LooseVersion("0.2.0"):
        # From 0.2.0, RAdam is dropped
        optim_classes.update(
            radam=torch_optimizer.RAdam,
        )
    del torch_optimizer
except ImportError:
    pass
try:
    import apex

    optim_classes.update(
        fusedadam=apex.optimizers.FusedAdam,
        fusedlamb=apex.optimizers.FusedLAMB,
        fusednovograd=apex.optimizers.FusedNovoGrad,
        fusedsgd=apex.optimizers.FusedSGD,
    )
    del apex
except ImportError:
    pass
try:
    import fairscale
except ImportError:
    fairscale = None


scheduler_classes = dict(
    ReduceLROnPlateau=torch.optim.lr_scheduler.ReduceLROnPlateau,
    lambdalr=torch.optim.lr_scheduler.LambdaLR,
    steplr=torch.optim.lr_scheduler.StepLR,
    multisteplr=torch.optim.lr_scheduler.MultiStepLR,
    exponentiallr=torch.optim.lr_scheduler.ExponentialLR,
    CosineAnnealingLR=torch.optim.lr_scheduler.CosineAnnealingLR,
    noamlr=NoamLR,
    warmuplr=WarmupLR,
    cycliclr=torch.optim.lr_scheduler.CyclicLR,
    onecyclelr=torch.optim.lr_scheduler.OneCycleLR,
    CosineAnnealingWarmRestarts=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
)


def build_optimizers(conf, model):

    optim_class = optim_classes.get(conf["optim"])
    optim = optim_class(model.parameters(), **conf["optim_conf"])

    optimizers = [optim]
    return optimizers


def model_param_setting(model_config, preprocess_config):
    # params = dict()
    params = model_config["tts_conf"]
    symbol_path = preprocess_config["path"]["symbol_path"]
    # network structure related

    with open(symbol_path, 'r') as symbol_file:
        symbols = json.load(symbol_file)

    params["idim"] = len(symbols)
    params["odim"] = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]

    return params


def resume(
    checkpoint,
    model,
    reporter=None,
    optimizers=None,
    schedulers=None,
    device=None,
):
    states = torch.load(
        checkpoint,
        map_location=device,
    )

    if reporter is None or optimizers is None or schedulers is None:
        model.load_state_dict(states)
        return model
    else:
        model.load_state_dict(states["model"])

    reporter.load_state_dict(states["reporter"])
    for optimizer, state in zip(optimizers, states["optimizers"]):
        optimizer.load_state_dict(state)
    for scheduler, state in zip(schedulers, states["schedulers"]):
        if scheduler is not None:
            scheduler.load_state_dict(state)

    logging.info(f"The training was resumed using {checkpoint}")

    return model, optimizers, schedulers


def build_schedulers(model_config, optimizers):
    schedulers = []
    for i, optim in enumerate(optimizers, 1):
        name = model_config["scheduler"]
        conf = model_config["scheduler_conf"]
        if name is not None:
            cls_ = scheduler_classes.get(name)
            if cls_ is None:
                raise ValueError(
                    f"must be one of {list(scheduler_classes)}: {name}"
                )
            scheduler = cls_(optim, **conf)
        else:
            scheduler = None

        schedulers.append(scheduler)

    return schedulers


def get_parser():
    class ArgumentDefaultsRawTextHelpFormatter(
        argparse.RawTextHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
    ):
        pass

    parser = config_argparse.ArgumentParser(
        description="base parser",
        formatter_class=ArgumentDefaultsRawTextHelpFormatter,
    )

    # NOTE(kamo): Use '_' instead of '-' to avoid confusion.
    #  I think '-' looks really confusing if it's written in yaml.

    # NOTE(kamo): add_arguments(..., required=True) can't be used
    #  to provide --print_config mode. Instead of it, do as
    parser.set_defaults(required=["output_dir"])

    group = parser.add_argument_group("Common configuration")

    group.add_argument(
        "--print_config",
        action="store_true",
        help="Print the config file and exit",
    )
    group.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )
    group.add_argument(
        "--dry_run",
        type=str2bool,
        default=False,
        help="Perform process without training",
    )
    group.add_argument(
        "--iterator_type",
        type=str,
        choices=["sequence", "chunk", "task", "none"],
        default="sequence",
        help="Specify iterator type",
    )

    group.add_argument("--output_dir", type=str_or_none, default=None)
    group.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    group.add_argument("--seed", type=int, default=50, help="Random seed")
    group.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )
    group.add_argument(
        "--num_att_plot",
        type=int,
        default=3,
        help="The number images to plot the outputs from attention. "
             "This option makes sense only when attention-based model. "
             "We can also disable the attention plot by setting it 0",
    )

    group = parser.add_argument_group("distributed training related")
    group.add_argument(
        "--dist_backend",
        default="nccl",
        type=str,
        help="distributed backend",
    )
    group.add_argument(
        "--dist_init_method",
        type=str,
        default="env://",
        help='if init_method="env://", env values of "MASTER_PORT", "MASTER_ADDR", '
             '"WORLD_SIZE", and "RANK" are referred.',
    )
    group.add_argument(
        "--dist_world_size",
        default=None,
        type=int_or_none,
        help="number of nodes for distributed training",
    )
    group.add_argument(
        "--dist_rank",
        type=int_or_none,
        default=None,
        help="node rank for distributed training",
    )
    group.add_argument(
        # Not starting with "dist_" for compatibility to launch.py
        "--local_rank",
        type=int_or_none,
        default=None,
        help="local rank for distributed training. This option is used if "
             "--multiprocessing_distributed=false",
    )
    group.add_argument(
        "--dist_master_addr",
        default=None,
        type=str_or_none,
        help="The master address for distributed training. "
             "This value is used when dist_init_method == 'env://'",
    )
    group.add_argument(
        "--dist_master_port",
        default=None,
        type=int_or_none,
        help="The master port for distributed training"
             "This value is used when dist_init_method == 'env://'",
    )
    group.add_argument(
        "--dist_launcher",
        default=None,
        type=str_or_none,
        choices=["slurm", "mpi", None],
        help="The launcher type for distributed training",
    )
    group.add_argument(
        "--multiprocessing_distributed",
        default=False,
        type=str2bool,
        help="Use multi-processing distributed training to launch "
             "N processes per node, which has N GPUs. This is the "
             "fastest way to use PyTorch for either single node or "
             "multi node data parallel training",
    )
    group.add_argument(
        "--unused_parameters",
        type=str2bool,
        default=False,
        help="Whether to use the find_unused_parameters in "
             "torch.nn.parallel.DistributedDataParallel ",
    )
    group.add_argument(
        "--sharded_ddp",
        default=False,
        type=str2bool,
        help="Enable sharded training provided by fairscale",
    )

    group = parser.add_argument_group("cudnn mode related")
    group.add_argument(
        "--cudnn_enabled",
        type=str2bool,
        default=torch.backends.cudnn.enabled,
        help="Enable CUDNN",
    )
    group.add_argument(
        "--cudnn_benchmark",
        type=str2bool,
        default=torch.backends.cudnn.benchmark,
        help="Enable cudnn-benchmark mode",
    )
    group.add_argument(
        "--cudnn_deterministic",
        type=str2bool,
        default=True,
        help="Enable cudnn-deterministic mode",
    )

    group = parser.add_argument_group("collect stats mode related")
    group.add_argument(
        "--collect_stats",
        type=str2bool,
        default=False,
        help='Perform on "collect stats" mode',
    )
    group.add_argument(
        "--write_collected_feats",
        type=str2bool,
        default=False,
        help='Write the output features from the model when "collect stats" mode',
    )

    group = parser.add_argument_group("Trainer related")
    group.add_argument(
        "--patience",
        type=int_or_none,
        default=None,
        help="Number of epochs to wait without improvement "
             "before stopping the training",
    )
    group.add_argument(
        "--val_scheduler_criterion",
        type=str,
        nargs=2,
        default=("valid", "loss"),
        help="The criterion used for the value given to the lr scheduler. "
             'Give a pair referring the phase, "train" or "valid",'
             'and the criterion name. The mode specifying "min" or "max" can '
             "be changed by --scheduler_conf",
    )
    group.add_argument(
        "--early_stopping_criterion",
        type=str,
        nargs=3,
        default=("valid", "loss", "min"),
        help="The criterion used for judging of early stopping. "
             'Give a pair referring the phase, "train" or "valid",'
             'the criterion name and the mode, "min" or "max", e.g. "acc,max".',
    )
    group.add_argument(
        "--best_model_criterion",
        type=str2triple_str,
        nargs="+",
        default=[
            ("train", "loss", "min"),
            ("valid", "loss", "min"),
        ],
        help="The criterion used for judging of the best model. "
             'Give a pair referring the phase, "train" or "valid",'
             'the criterion name, and the mode, "min" or "max", e.g. "acc,max".',
    )
    group.add_argument(
        "--keep_nbest_models",
        type=int,
        nargs="+",
        default=[10],
        help="Remove previous snapshots excluding the n-best scored epochs",
    )
    group.add_argument(
        "--nbest_averaging_interval",
        type=int,
        default=0,
        help="The epoch interval to apply model averaging and save nbest models",
    )
    group.add_argument(
        "--grad_clip",
        type=float,
        default=5.0,
        help="Gradient norm threshold to clip",
    )
    group.add_argument(
        "--grad_clip_type",
        type=float,
        default=2.0,
        help="The type of the used p-norm for gradient clip. Can be inf",
    )
    group.add_argument(
        "--grad_noise",
        type=str2bool,
        default=False,
        help="The flag to switch to use noise injection to "
             "gradients during training",
    )
    group.add_argument(
        "--accum_grad",
        type=int,
        default=1,
        help="The number of gradient accumulation",
    )
    group.add_argument(
        "--no_forward_run",
        type=str2bool,
        default=False,
        help="Just only iterating data loading without "
             "model forwarding and training",
    )
    group.add_argument(
        "--resume",
        type=str2bool,
        default=False,
        help="Enable resuming if checkpoint is existing",
    )
    group.add_argument(
        "--train_dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type for training.",
    )
    group.add_argument(
        "--use_amp",
        type=str2bool,
        default=False,
        help="Enable Automatic Mixed Precision. This feature requires pytorch>=1.6",
    )
    group.add_argument(
        "--log_interval",
        type=int_or_none,
        default=None,
        help="Show the logs every the number iterations in each epochs at the "
             "training phase. If None is given, it is decided according the number "
             "of training samples automatically .",
    )
    group.add_argument(
        "--use_matplotlib",
        type=str2bool,
        default=True,
        help="Enable matplotlib logging",
    )
    group.add_argument(
        "--use_tensorboard",
        type=str2bool,
        default=True,
        help="Enable tensorboard logging",
    )
    group.add_argument(
        "--use_wandb",
        type=str2bool,
        default=False,
        help="Enable wandb logging",
    )
    group.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Specify wandb project",
    )
    group.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="Specify wandb id",
    )
    group.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Specify wandb entity",
    )
    group.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Specify wandb run name",
    )
    group.add_argument(
        "--wandb_model_log_interval",
        type=int,
        default=-1,
        help="Set the model log period",
    )
    group.add_argument(
        "--detect_anomaly",
        type=str2bool,
        default=False,
        help="Set torch.autograd.set_detect_anomaly",
    )

    group = parser.add_argument_group("Pretraining model related")
    group.add_argument("--pretrain_path", help="This option is obsoleted")
    group.add_argument(
        "--init_param",
        type=str,
        default=[],
        nargs="*",
        help="Specify the file path used for initialization of parameters. "
             "The format is '<file_path>:<src_key>:<dst_key>:<exclude_keys>', "
             "where file_path is the model file path, "
             "src_key specifies the key of model states to be used in the model file, "
             "dst_key specifies the attribute of the model to be initialized, "
             "and exclude_keys excludes keys of model states for the initialization."
             "e.g.\n"
             "  # Load all parameters"
             "  --init_param some/where/model.pth\n"
             "  # Load only decoder parameters"
             "  --init_param some/where/model.pth:decoder:decoder\n"
             "  # Load only decoder parameters excluding decoder.embed"
             "  --init_param some/where/model.pth:decoder:decoder:decoder.embed\n"
             "  --init_param some/where/model.pth:decoder:decoder:decoder.embed\n",
    )
    group.add_argument(
        "--ignore_init_mismatch",
        type=str2bool,
        default=False,
        help="Ignore size mismatch when loading pre-trained model",
    )
    group.add_argument(
        "--freeze_param",
        type=str,
        default=[],
        nargs="*",
        help="Freeze parameters",
    )

    group = parser.add_argument_group("BatchSampler related")
    group.add_argument(
        "--num_iters_per_epoch",
        type=int_or_none,
        default=None,
        help="Restrict the number of iterations for training per epoch",
    )
    group.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="The mini-batch size used for training. Used if batch_type='unsorted',"
             " 'sorted', or 'folded'.",
    )
    group.add_argument(
        "--valid_batch_size",
        type=int_or_none,
        default=None,
        help="If not given, the value of --batch_size is used",
    )
    group.add_argument(
        "--batch_bins",
        type=int,
        default=1000000,
        help="The number of batch bins. Used if batch_type='length' or 'numel'",
    )
    group.add_argument(
        "--valid_batch_bins",
        type=int_or_none,
        default=None,
        help="If not given, the value of --batch_bins is used",
    )

    group.add_argument("--train_shape_file", type=str, action="append", default=[])
    group.add_argument("--valid_shape_file", type=str, action="append", default=[])

    return parser


def get_model(configs, args=None, reporter=None, device=None, train=False):
    if len(configs) > 2:
        (preprocess_config, model_config, _) = configs
    else:
        (preprocess_config, model_config) = configs

    params = model_param_setting(model_config, preprocess_config)
    model = FastSpeech2(**params).to(device)

    if args is None:
        return model

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        optimizers = build_optimizers(model_config, model)
        schedulers = build_schedulers(model_config, optimizers)

        model, optimizers, schedulers = resume(
            checkpoint=join(args.result_path, "checkpoint.pth"),
            model=model,
            optimizers=optimizers,
            schedulers=schedulers,
            reporter=reporter,
            device=device,
        )

        return model, optimizers, schedulers

    if train:
        optimizers = build_optimizers(model_config, model)
        schedulers = build_schedulers(model_config, optimizers)

        model.train()
        return model, optimizers, schedulers

    model.eval()
    model.requires_grad_ = False

    return model


def get_param_num(model):
    num_params = sum(param.numel() for param in model.parameters())
    return num_params