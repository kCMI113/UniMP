import time

import numpy as np
import torch
from PIL import Image
from pipeline.train.train_utils import (
    AverageMeter,
    get_autocast,
    get_cast_dtype,
    get_checkpoint,
)
from torch.utils.data import Dataset
from tqdm import tqdm

from ..mm_utils.collate_rec import collate_fn

IMG_PATH = "/home/kcm/UniMP/data/ori_images"


class HMDataset(Dataset):
    def __init__(
        self,
        args,
        user_seq,
        num_item: int,
        idx2meta: dict,
        idx2item: dict,
        supported_data_types=["seq_rec"],
        task="rec",
        type: str = "Train",
    ):
        # super().__init__()
        self.type = type
        self.args = args
        self.tokenizer = args.tokenizer
        self.tasks = task
        self.use_semantic = args.use_semantic
        self.max_src_length = args.max_src_length
        self.max_tgt_length = args.max_tgt_length
        self.history_len = 8
        self.num_item = num_item
        self.all_items = set(range(num_item))
        self.idx2meta = idx2meta
        self.idx2item = idx2item
        self.user_seq = user_seq

        self.seed = args.pretrain_seed
        self.code_dict_size = args.code_dict_size
        self.patch_image_size = args.patch_image_size
        self.code_image_size = args.code_image_size
        self.supported_data_types = supported_data_types

        scales = [(args.patch_image_size, args.patch_image_size)]

        self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id])
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])
        self.rank = args.rank
        self.test_len = -20

    def collate(self, samples):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        samples_v1 = []  # containing image-text pairs
        for sample_tuple in samples:
            samples_v1.append(sample_tuple)

        res_v1 = collate_fn(
            samples_v1,
            pad_idx=self.tokenizer.pad_token_id,
            eos_idx=self.tokenizer.eos_token_id,
        )
        return res_v1

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        user = np.array(self.user_seq[index])  # item index range : (1,n_items)
        tokens = user[: self.type_idx]
        input_seq = ""
        img_seq = []

        start = (
            np.random.choice(list(range(0, len(tokens) - self.history_len)), 1)[0]
            if self.type == "Train"
            else self.test_len
        )
        end = start + self.history_len if self.type == "Train" else -1

        for idx in tokens[start:end]:
            image_item = Image.open(
                f"{IMG_PATH}/{self.idx2item[idx][:3]}/{self.idx2item[idx]}.jpg"
            ).convert("RGB")
            img_seq.append(self.patch_resize_transform(image_item))

            meta_item = self.idx2meta[idx]
            input_seq += f"<image> {meta_item} <answer> item_{idx} <|endofchunk|> "

        input_seq = (
            input_seq + f"What is the next item recommended to the user? <answer>"
        )

        if self.type == "Train":
            input_seq += f" item_{tokens[end]}"
        else:
            semantic_id = f"item_{tokens[end]}"

        patch_image = torch.stack(img_seq, dim=0)
        src_text = self.tokenizer(
            input_seq, return_tensors="pt", add_special_tokens=False, truncation=True
        )

        if self.type == "Train":
            src_item = src_text["input_ids"].squeeze(0)
            src_item_mask = src_text["attention_mask"].squeeze(0)
            src_item = torch.cat([self.bos_item, src_item, self.eos_item])
            src_item_mask = torch.cat([self.bos_mask, src_item_mask, self.eos_mask])

            example = {
                "net_input": {
                    "input_ids": src_item,
                    "attention_masks": src_item_mask,
                    "patch_images": patch_image,
                    "weights": torch.tensor(2.0),
                }
            }
        else:
            input_len = len(input_seq.split(" "))
            src_item = src_text["input_ids"].squeeze(0)
            src_item_mask = src_text["attention_mask"].squeeze(0)

            example = {
                "net_input": {
                    "input_ids": src_item,
                    "attention_masks": src_item_mask,
                    "patch_images": patch_image,
                    "input_len": input_len,
                },
                "net_output": {
                    "output_ids": semantic_id,
                },
            }

        return example


def Trainer(
    args,
    model,
    epoch,
    multi_instruct_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    # accelerator,
    wandb,
):
    # num_batches_per_epoch = multi_instruct_loader.num_batches
    num_batches_per_epoch = len(multi_instruct_loader)
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]
    answer_token_id = tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
    if args.task == "img_gen":
        img_tokens = ["img_789", "img_591", "img_977"]  # ??? gen이라 신경꺼도 되나?
        delete_list = []
        for token in img_tokens:
            img_token = tokenizer(token, add_special_tokens=False)["input_ids"][-1]
            delete_list.append(img_token)

    model.train()
    # alpha = 0.25
    gamma = args.gamma
    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    for num_steps, (batch_multi_instruct) in tqdm(
        enumerate(multi_instruct_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)

        global_step = num_steps + epoch * num_batches_per_epoch
        #### MULTI_INSTRUCT FORWARD PASS ####

        # images = (
        #     batch_multi_instruct["net_input"]["patch_images"]
        #     .to(device_id, dtype=cast_dtype, non_blocking=True)
        #     .unsqueeze(1)
        #     .unsqueeze(1)
        # )

        # images = (
        #     batch_multi_instruct["net_input"]["patch_images"].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2)
        # )
        # input_ids = batch_multi_instruct["net_input"]["input_ids"].to(
        #     device_id, dtype=cast_dtype, non_blocking=True
        # )
        # attention_mask = batch_multi_instruct["net_input"]["attention_masks"].to(
        #     device_id, dtype=cast_dtype, non_blocking=True
        # )
        images = batch_multi_instruct["net_input"]["patch_images"].unsqueeze(2)

        input_ids = batch_multi_instruct["net_input"]["input_ids"]
        attention_mask = batch_multi_instruct["net_input"]["attention_masks"]
        weights = batch_multi_instruct["net_input"]["weights"]

        labels = input_ids.clone()
        # reweight = (1-alpha)*torch.ones_like(labels, dtype=torch.float)
        # only keep the loss for eos and the answer between <answer> and <endofchunk>
        for i in range(labels.shape[0]):
            answer_flag = 0
            for j in range(labels.shape[1]):
                if not answer_flag:
                    if labels[i, j] == answer_token_id:
                        answer_flag = 1
                    labels[i, j] = -100
                else:
                    if labels[i, j] == endofchunk_token_id:
                        answer_flag = 0
                        labels[i, j] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        # for i in range(labels.shape[0]):
        #     # remove loss for any token before <answer> token
        #     label_idx = 0
        #     while (
        #         label_idx < labels.shape[1] and labels[i][label_idx] != answer_token_id
        #     ):
        #         labels[i][label_idx] = -100
        #         label_idx += 1
        labels[labels == answer_token_id] = -100
        labels[labels == media_token_id] = -100
        # if args.task=="img_gen":
        #     for delete_token in delete_list:
        #         reweight[labels == delete_token] = alpha
        #     reweight = reweight[:, 1:].contiguous()

        # labels.to(device_id, dtype=cast_dtype, non_blocking=True)
        # with accelerator.accumulate(model):
        with autocast():
            output = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss_multi_instruct_b = output[0]

            # divided_loss_multi_instruct = loss_multi_instruct

            #### BACKWARD PASS ####
            # loss_multi_instruct = loss_multi_instruct_b*weights[0]

            lm_logits = output["logits"]
            labels = labels.to(lm_logits.device)
            # batch_size x n_tokens
            n1, n2 = labels.shape[0], labels.shape[1] - 1
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            # resize
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            labels = labels.view(-1)
            # loss is zero for label index = 100
            lm_loss = loss_fct(shift_logits, labels).view(n1, n2)
            # task weight
            loss_multi_instruct = torch.unsqueeze(weights, 1) * lm_loss
            loss_multi_instruct = loss_multi_instruct.view(-1)
            if args.use_reweight:
                # focal term
                p = torch.nn.functional.softmax(shift_logits, dim=-1)
                all_rows = torch.arange(len(shift_logits))
                pt = p[all_rows, labels]
                focal_term = (1 - pt) ** gamma
                # print(loss_multi_instruct.shape, focal_term.shape, reweight.shape)
                loss_multi_instruct = loss_multi_instruct * focal_term
            loss_multi_instruct = torch.sum(loss_multi_instruct) / torch.sum(
                labels != -100
            )

            optimizer.backward(loss_multi_instruct)

            cast_dtype = get_cast_dtype(args.precision)

            #### MASK GRADIENTS FOR EMBEDDINGS ####
            # Note (anas): Do not apply weight decay to embeddings as it will break this function.
            def mask_embedding(m):
                if m.weight.requires_grad:
                    zero_mask = torch.zeros_like(m.weight.grad)
                    zero_mask[answer_token_id] = torch.ones_like(
                        zero_mask[answer_token_id]
                    )
                    m.weight.grad = m.weight.grad * zero_mask

            if args.mask_lm_head:
                model.module.lang_encoder.model.embed_tokens.apply(mask_embedding)
                model.module.lang_encoder.lm_head.apply(mask_embedding)
            # def mask_embedding(m):
            #     if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
            #         zero_mask = torch.zeros_like(m.weight)
            #         zero_mask[media_token_id] = torch.ones_like(
            #             zero_mask[media_token_id]
            #         )
            #         zero_mask[endofchunk_token_id] = torch.ones_like(
            #             zero_mask[endofchunk_token_id]
            #         )
            #         zero_mask[answer_token_id] = torch.ones_like(
            #             zero_mask[answer_token_id]
            #         )
            #         m.weight.grad = m.weight.grad * zero_mask

            # model.apply(mask_embedding)

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # if accelerator.sync_gradients:
            #     accelerator.clip_grad_norm_(model.parameters(), 1.0)

            # step optimizer and log
            # if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            #     num_steps == num_batches_per_epoch - 1
            # ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # step time and reset end outside of rank 0
        step_time_m.update(time.time() - end)
        end = time.time()

        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                multi_instruct_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    * args.world_size
                    / step_time_m.val
                )
                multi_instruct_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps * args.batch_size / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "multi_instruct_samples_per_second": multi_instruct_samples_per_second,
                        "multi_instruct_samples_per_second_per_gpu": multi_instruct_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_multi_instruct": loss_multi_instruct_b.item(),
                        "global_step": global_step // args.gradient_accumulation_steps,
                    },
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss Multi-Instruct: {loss_multi_instruct_b.item():.3f}"
            )
