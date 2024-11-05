""" Main training script """

import argparse
import glob
import os
import random
import warnings

# from peft import get_peft_config, get_peft_model, PrefixTuningConfig, TaskType, PeftType
import numpy as np
import torch
import torch.nn
import wandb
from huggingface_hub import hf_hub_download
from open_flamingo import Flamingo, create_model_and_transforms
from pipeline.eval.eval_exp import eval_model_exp
from pipeline.eval.eval_img_gen import eval_model_img_gen
from pipeline.eval.eval_img_sel import eval_model_img_sel
from pipeline.eval.eval_rec import eval_model_rec
from pipeline.eval.eval_search import eval_model_search
from pipeline.train.data import get_data_rec
from pipeline.train.distributed import init_distributed_device, world_info_from_env
from pipeline.train.my_module import HMDataset, Trainer
from torch.utils.data import DataLoader, RandomSampler

warnings.filterwarnings("ignore")
os.environ["NCCL_P2P_LEVEL"] = "NVL"
# os.environ["NCCL_P2P_DISABLE"] = "1"

import sys
import time

from accelerate import Accelerator
from pipeline.mm_utils.arguments import add_data_args
from pipeline.train.train_utils import (
    AverageMeter,
    get_autocast,
    get_cast_dtype,
    get_checkpoint,
)
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=1,
        help="how often to add a cross-attention layer after each transformer layer",
    )
    parser.add_argument(
        "--external_save_dir",
        type=str,
        default=None,
        help="set to save model to external path",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="mm_3b",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--use_media_placement_augmentation", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log loss every n steps"
    )
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="path to huggingface model or model identifier from local path or huggingface.co",
        default="3b",
    )
    parser.add_argument(
        "--load_from_original_checkpoint",
        type=str,
        help="path to openflamingo provided checkpoint, in .pt format",
        default=None,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite_checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument(
        "--mmrec_path",
        type=str,
        help="path to mmrec dataset.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="rec",
    )
    parser.add_argument("--use_semantic", default=False, action="store_true")
    parser.add_argument("--use_reweight", default=False, action="store_true")
    parser.add_argument(
        "--subset",
        type=str,
        help="subset of Amazon.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--loss_multiplier_multi_instruct", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--warmup_steps_ratio", default=None, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--do_eval", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )
    # data args
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train_num_samples", type=int, default=None)
    parser.add_argument("--dataset_resampled", action="store_true")
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument("--mask_lm_head", action="store_true")
    parser.add_argument("--save_hf_model", default=False, action="store_true")
    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument("--single_task", default=False, action="store_true")
    parser.add_argument(
        "--train_method", type=str, default="multi_task", help="multi_task | continue"
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )

    parser = add_data_args(parser)
    args = parser.parse_args()

    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()
    if args.world_size > 1:
        device_id = init_distributed_device(args)
    else:
        device_id = 0

    random_seed(args.seed)
    if args.pretrained_model_name_or_path == "3b":
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1,
        )
        checkpoint_path = hf_hub_download(
            "openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt"
        )
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
    elif args.pretrained_model_name_or_path == "3b-instruct":
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
            cross_attn_every_n_layers=1,
        )
        checkpoint_path = hf_hub_download(
            "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", "checkpoint.pt"
        )
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
    elif args.pretrained_model_name_or_path == "4b":
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="togethercomputer/RedPajama-INCITE-Base-3B-v1",
            tokenizer_path="togethercomputer/RedPajama-INCITE-Base-3B-v1",
            cross_attn_every_n_layers=2,
        )
        checkpoint_path = hf_hub_download(
            "openflamingo/OpenFlamingo-4B-vitl-rpj3b", "checkpoint.pt"
        )
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
    elif args.pretrained_model_name_or_path == "4b-instruct":
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
            tokenizer_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
            cross_attn_every_n_layers=2,
        )
        checkpoint_path = hf_hub_download(
            "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct", "checkpoint.pt"
        )
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
    elif args.pretrained_model_name_or_path == "9b":
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-7b",
            tokenizer_path="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=4,
        )
        checkpoint_path = hf_hub_download(
            "openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt"
        )
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
    # 测试一下哪个比较难，只输入text，用semantic id和普通id；输入both text+id，id是atomic和semantic
    # 还有只输入id
    # 1
    # anas-awadalla/mpt-1b-redpajama-200b
    # openflamingo/OpenFlamingo-3B-vitl-mpt1b
    # 1
    # anas-awadalla/mpt-1b-redpajama-200b-dolly
    # openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct
    # 2
    # togethercomputer/RedPajama-INCITE-Instruct-3B-v1
    # openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct

    # add <answer> token to tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": ["<answer>"]})
    tokenizer.add_tokens(["rate_1", "rate_2", "rate_3", "rate_4", "rate_5"])
    tokenizer.add_tokens(["s_0", "s_1", "s_2", "s_3", "s_4"])

    num_item = 60233
    if not args.use_semantic:
        item_tokens = [f"item_{i}" for i in range(num_item)]
        tokenizer.add_tokens(item_tokens)
    else:
        item_tokens = [f"item_{i}" for i in range(512)]
        tokenizer.add_tokens(item_tokens)
        item_tokens = [f"item_last_{i}" for i in range(32)]
        tokenizer.add_tokens(item_tokens)

    img_tokens = [f"img_{i}," for i in range(1024)]
    tokenizer.add_tokens(img_tokens)

    model.lang_encoder.resize_token_embeddings(len(tokenizer))
    args.tokenizer = tokenizer
    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")
    device_id = args.rank % torch.cuda.device_count()

    ## dataloader로 변경할 수 있도록 dataset을 짜야함!! ##
    # multi_instruct_loader = get_data_rec(
    #     args, tokenizer, "mmrec", split="train", task=args.task
    # )
    # multi_instruct_eval_loader = get_data_rec(
    #     args, tokenizer, "mmrec", split="eval", task=args.task
    # )
    # multi_instruct_test_loader = get_data_rec(
    #     args, tokenizer, "mmrec", split="test", task=args.task
    # )

    print("LOAD DATA")
    idx2meta = torch.load("./data/idx2meta.pt")
    idx2item = torch.load("./data/idx2item.pt")
    test_data = torch.load(
        "../../Fa-rec/data/datasets--SLKpnu--sequential/snapshots/724e08a869ebcf7197032760d45d3bd74ad4b5cf/small/test_data.pt"
    )

    train_dataset = HMDataset(
        args, test_data, num_item, idx2meta, idx2item, type="Train"
    )
    valid_dataset = HMDataset(
        args, test_data, num_item, idx2meta, idx2item, type="Valid"
    )
    test_dataset = HMDataset(args, test_data, num_item, idx2meta, idx2item, type="Test")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=RandomSampler(train_dataset),
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=1,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return (
                "gated_cross_attn_layer" in x
                and "ff_gate" not in x
                and "attn_gate" not in x
                and "norm" not in x
                and "bias" not in x
            )

        for n, p in model.named_parameters():
            # if p.requires_grad:
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        return [
            {"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]

    args.train_num_samples = (
        train_dataloader.num_samples
        if args.train_num_samples is None
        else args.train_num_samples
    )

    total_training_steps = len(train_dataloader) * args.num_epochs

    resume_from_epoch = 0
    # check if a checkpoint exists for this run
    args.external_save_dir = (
        os.path.join(args.external_save_dir, args.run_name)
        if args.external_save_dir
        else args.run_name
    )
    if (
        os.path.exists(f"{args.external_save_dir}")
        and args.resume_from_checkpoint is True
    ):
        checkpoint_list = glob.glob(f"{args.external_save_dir}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.external_save_dir}.")
        else:
            resume_from_checkpoint_path = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            print(
                f"Found checkpoint {resume_from_checkpoint_path} for run {args.external_save_dir}."
            )

        if args.rank == 0:
            print(f"Loading checkpoint from {resume_from_checkpoint_path}")
        checkpoint = torch.load(resume_from_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        resume_from_epoch = checkpoint["epoch"] + 1

    optimizer = torch.optim.AdamW(get_grouped_params(model), lr=args.learning_rate)

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    args.warmup_steps = (
        total_training_steps * args.warmup_steps_ratio
        if args.warmup_steps_ratio is not None
        else args.warmup_steps
    )

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps,
            num_training_steps=total_training_steps // args.gradient_accumulation_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )
    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    # if args.single_task:
    #     (
    #         model,
    #         optimizer,
    #         lr_scheduler,
    #         train_dataloader,
    #         valid_dataloader,
    #         test_dataloader,
    #     ) = accelerator.prepare(
    #         model,
    #         optimizer,
    #         lr_scheduler,
    #         train_dataloader,
    #         valid_dataloader,
    #         test_dataloader,
    #     )

    # multi_instruct_test_loader_img_gen
    model.train()
    device_id = "cuda_0"
    if args.single_task:
        multi_instruct_test_loader = test_dataloader
        # eval_func = eval(f"eval_model_{args.task}")
        if args.task == "rec":
            eval_func = eval_model_rec
        elif args.task == "exp":
            eval_func = eval_model_exp
        elif args.task == "img_sel":
            eval_func = eval_model_img_sel
        elif args.task == "search":
            eval_func = eval_model_search
        elif args.task == "img_gen":
            eval_func = eval_model_img_gen
        else:
            raise KeyError("Not Supported Task Type")

    interval = 1 if args.task != "img_gen" else 1
    for epoch in range(resume_from_epoch, args.num_epochs):
        Trainer(
            args=args,
            model=model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            multi_instruct_loader=train_dataloader,
            # accelerator=accelerator,
            device_id=device_id,
            wandb=wandb,
        )
        if args.rank == 0:
            if not os.path.exists(args.external_save_dir):
                os.makedirs(args.external_save_dir)

        # accelerator.wait_for_everyone()

        if (epoch + 1) % interval == 0:
            if args.do_eval:
                eval_func(
                    args=args,
                    model=model,
                    epoch=epoch,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    multi_instruct_loader=valid_dataloader,
                    # accelerator=accelerator,
                    device_id=device_id,
                    wandb=wandb,
                )
                # accelerator.wait_for_everyone()
            if args.do_test:
                if args.single_task:
                    # if epoch>5:
                    eval_func(
                        args=args,
                        model=model,
                        epoch=epoch,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        multi_instruct_loader=test_dataloader,
                        # accelerator=accelerator,
                        device_id=device_id,
                        wandb=wandb,
                    )

                # accelerator.wait_for_everyone()
            if args.rank == 0:
                if not os.path.exists(args.external_save_dir):
                    os.makedirs(args.external_save_dir)

    #             unwrapped_model = accelerator.unwrap_model(model)
    #             accelerator.save(
    #                 get_checkpoint(model=unwrapped_model),
    #                 f"{args.external_save_dir}/weights_epoch_{epoch}.pt",
    #             )

    # accelerator.wait_for_everyone()
    if args.rank == 0:
        if not os.path.exists(args.external_save_dir):
            os.makedirs(args.external_save_dir)

        # unwrapped_model = accelerator.unwrap_model(model)
        # accelerator.save(
        #     get_checkpoint(model=unwrapped_model),
        #     f"{args.external_save_dir}/final_weights.pt",
        # )
        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{args.external_save_dir}/final_weights.pt")


if __name__ == "__main__":
    main()
