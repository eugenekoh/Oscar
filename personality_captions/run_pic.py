# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import argparse
import ast
import base64
import json
import os.path as op
import random
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from oscar.modeling.modeling_bert import BertForPersonalityImageCaptioning
from oscar.utils.caption_evaluate import (evaluate_on_coco_caption,
                                          evaluate_on_nocaps)
from oscar.utils.logger import setup_logger
from oscar.utils.misc import (mkdir, set_seed,
                              load_from_yaml_file, find_file_path_in_yaml)
from oscar.utils.tsv_file import TSVFile
from oscar.utils.tsv_file_ops import tsv_writer
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from transformers.pytorch_transformers import BertTokenizer, BertConfig


class CaptionTSVDataset(Dataset):
    def __init__(self, yaml_file, tokenizer=None, add_od_labels=True,
                 max_img_seq_length=50, max_seq_length=70, max_seq_a_length=40,
                 is_train=True, mask_prob=0.15, max_masked_tokens=3, **kwargs):
        """Constructor.
        Args:
            yaml file with all required data (image feature, caption, labels, etc)
            tokenizer: tokenizer for text processing.
            add_od_labels: whether to add labels from yaml file to BERT. 
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
            kwargs: other arguments.
        """
        self.yaml_file = yaml_file
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = op.dirname(yaml_file)
        self.label_file = find_file_path_in_yaml(self.cfg['label'], self.root)
        self.feat_file = find_file_path_in_yaml(self.cfg['feature'], self.root)
        self.caption_file = find_file_path_in_yaml(self.cfg.get('caption'), self.root)
        self.personality_file = find_file_path_in_yaml(self.cfg.get('personality'), self.root)

        assert op.isfile(self.feat_file)
        if add_od_labels: assert op.isfile(self.label_file)
        if is_train: assert op.isfile(self.caption_file) and tokenizer is not None

        self.label_tsv = None if not self.label_file else TSVFile(self.label_file)
        self.feat_tsv = TSVFile(self.feat_file)
        if self.caption_file and op.isfile(self.caption_file):
            with open(self.caption_file, 'r') as f:
                self.captions = json.load(f)

        self.tokenizer = tokenizer
        self.tensorizer = CaptionTensorizer(self.tokenizer, max_img_seq_length,
                                            max_seq_length, max_seq_a_length, mask_prob, max_masked_tokens,
                                            is_train=is_train)
        self.add_od_labels = add_od_labels
        self.is_train = is_train
        self.kwargs = kwargs
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.key2captions = self.prepare_image_key_to_captions()
        self.personality2index = self.prepare_personality_to_index()

    def get_valid_tsv(self):
        # based on the order of file size
        if self.label_tsv:
            return self.label_tsv
        if self.feat_tsv:
            return self.feat_tsv

    def prepare_personality_to_index(self):
        with open(self.personality_file) as f:
            personalities = list(x.strip() for x in f.readlines())
            assert len(personalities) == 215

        return {x: i for i, x in enumerate(personalities)}

    def prepare_image_keys(self):
        tsv = self.get_valid_tsv()
        return [tsv.seek(i)[0] for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.seek(i)[0]: i for i in range(tsv.num_rows())}

    def prepare_image_key_to_captions(self):
        if self.is_train:
            key2captions = {key: [] for key in self.image_keys}
            for cap in self.captions:
                key2captions[cap['image_hash']].append(cap['comment'])
            return key2captions

    def get_image_index(self, idx):
        img_cap_pair = self.captions[idx]
        img_key = img_cap_pair['image_hash']
        return self.key2index[img_key]

    def get_image_key(self, idx):
        img_idx = self.get_image_index(idx)
        return self.image_keys[img_idx]

    def get_image_features(self, img_idx):
        feat_info = ast.literal_eval(self.feat_tsv.seek(img_idx)[1])
        num_boxes = feat_info['num_boxes']
        features = np.frombuffer(base64.b64decode(feat_info['features']), np.float32
                                 ).reshape((num_boxes, -1))
        return torch.Tensor(features)

    def get_caption(self, idx):
        if self.is_train:
            img_cap_pair = self.captions[idx]
            return img_cap_pair['comment']
        return ""

    def get_personality(self, idx):
        img_cap_pair = self.captions[idx]
        return self.personality2index[img_cap_pair['personality']]

    def get_od_labels(self, img_idx):
        od_labels = None
        if self.add_od_labels:
            label_info = ast.literal_eval(self.label_tsv.seek(img_idx)[1])
            od_labels = " ".join([l['class'] for l in label_info])
        return od_labels

    def get_caption_file_in_coco_format(self):
        cap_file = op.splitext(self.caption_file)[0] + '_coco_format.json'
        return cap_file

    def get_captions_by_key(self, key):
        assert self.is_train, "cannot get captions for inference"
        return self.key2captions[key]

    def __getitem__(self, idx):
        img_idx = self.get_image_index(idx)
        img_key = self.image_keys[img_idx]
        features = self.get_image_features(img_idx)
        caption = self.get_caption(idx)
        od_labels = self.get_od_labels(img_idx)
        personality = self.get_personality(idx)
        example = self.tensorizer.tensorize_example(caption, features, text_b=od_labels, personality=personality)
        return img_key, example

    def __len__(self):
        if self.is_train:
            return len(self.captions)
        return self.get_valid_tsv().num_rows()


class CaptionTensorizer(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=70,
                 max_seq_a_length=40, mask_prob=0.15, max_masked_tokens=3,
                 is_train=True):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.mask_prob = mask_prob
        self.max_masked_tokens = max_masked_tokens
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len,
                                                     self.max_seq_len), dtype=torch.long))

    def tensorize_example(self, text_a, img_feat, text_b=None,
                          cls_token_segment_id=0, pad_token_segment_id=0,
                          sequence_a_segment_id=0, sequence_b_segment_id=1,
                          personality=None):
        if self.is_train:
            tokens_a = self.tokenizer.tokenize(text_a)
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)
        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            padding_a_len = self.max_seq_a_len - seq_a_len
            tokens += [self.tokenizer.pad_token] * padding_a_len
            segment_ids += ([pad_token_segment_id] * padding_a_len)

            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        if self.is_train:
            masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            # randomly mask words for prediction, ignore [CLS]
            candidate_masked_idx = list(range(1, seq_a_len))  # only mask text_a
            random.shuffle(candidate_masked_idx)
            num_masked = min(max(round(self.mask_prob * seq_a_len), 1), self.max_masked_tokens)
            num_masked = int(num_masked)
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= 0.8:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    from random import randint
                    i = randint(0, len(self.tokenizer.vocab))
                    self.tokenizer._convert_id_to_token(i)
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

            masked_pos[masked_idx] = 1
            # pad masked tokens to the same length
            if num_masked < self.max_masked_tokens:
                masked_token = masked_token + ([self.tokenizer.pad_token] *
                                               (self.max_masked_tokens - num_masked))
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        else:
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)

        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0: self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len,
                                          img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # prepare attention mask:
        # note that there is no attention from caption to image
        # because otherwise it will violate the triangle attention 
        # for caption as caption will have full attention on image. 
        max_len = self.max_seq_len + self.max_img_seq_len
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_a_len
        l_start, l_end = self.max_seq_a_len, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
        # triangle mask for caption to caption
        attention_mask[c_start: c_end, c_start: c_end].copy_(self._triangle_mask[0: seq_a_len, 0: seq_a_len])
        # full attention for L-L, R-R
        attention_mask[l_start: l_end, l_start: l_end] = 1
        attention_mask[r_start: r_end, r_start: r_end] = 1
        # full attention for C-L, C-R
        attention_mask[c_start: c_end, l_start: l_end] = 1
        attention_mask[c_start: c_end, r_start: r_end] = 1
        # full attention for L-R:
        attention_mask[l_start: l_end, r_start: r_end] = 1
        attention_mask[r_start: r_end, l_start: l_end] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        personality = torch.tensor(personality, dtype=torch.long)

        # add personality index
        if self.is_train:
            masked_ids = torch.tensor(masked_ids, dtype=torch.long)
            return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, masked_ids, personality)
        return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, personality)


def build_dataset(yaml_file, tokenizer, args, is_train=True):
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        assert op.isfile(yaml_file)

    if is_train:
        return CaptionTSVDataset(yaml_file, tokenizer=tokenizer,
                                 add_od_labels=args.add_od_labels, max_img_seq_length=args.max_img_seq_length,
                                 max_seq_length=args.max_seq_length, max_seq_a_length=args.max_seq_a_length,
                                 is_train=True, mask_prob=args.mask_prob, max_masked_tokens=args.max_masked_tokens)

    dataset_class = CaptionTSVDataset
    return dataset_class(yaml_file, tokenizer=tokenizer,
                         add_od_labels=args.add_od_labels, max_img_seq_length=args.max_img_seq_length,
                         max_seq_length=args.max_seq_length, max_seq_a_length=args.max_gen_length,
                         is_train=False)


def save_checkpoint(model, tokenizer, args, epoch, global_step):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, global_step))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    save_num = 0
    while (save_num < 10):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return checkpoint_dir


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data  # argmax
    scores = logits == labels
    return scores


def train(args, train_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    writer = SummaryWriter(logdir=args.tensorboard_log_dir)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=args.num_workers)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // \
                                                   args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                  * args.num_train_epochs

    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if \
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(
            optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc = 0, 0.0, 0.0
    model.zero_grad()
    for epoch in range(int(args.num_train_epochs)):
        for step, (img_keys, batch) in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)

            model.train()
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                      'token_type_ids': batch[2], 'img_feats': batch[3],
                      'masked_pos': batch[4], 'masked_ids': batch[5],
                      'personality_ids': batch[6],
                      }
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            masked_ids = inputs['masked_ids']
            masked_ids = masked_ids[masked_ids != 0]
            batch_score = compute_score_with_logits(logits, masked_ids)
            batch_acc = torch.sum(batch_score.float()) / torch.sum(inputs['masked_pos'])

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            global_loss += loss.item()
            global_acc += batch_acc
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
                optimizer.step()
                model.zero_grad()

                tensorboard_step = global_step + args.global_step_offset
                if global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), " \
                                "score: {:.4f} ({:.4f})".format(epoch, tensorboard_step,
                                                                optimizer.param_groups[0]["lr"], loss,
                                                                global_loss / global_step,
                                                                batch_acc, global_acc / global_step)
                                )
                    writer.add_scalar("Loss/Loss", loss, tensorboard_step)
                    writer.add_scalar("Loss/Global_Loss", global_loss / global_step, tensorboard_step)
                    writer.add_scalar("Accuracy/Accuracy", batch_acc, tensorboard_step)
                    writer.add_scalar("Accuracy/Global_Accuracy", global_acc / global_step, tensorboard_step)

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    _ = save_checkpoint(model, tokenizer, args, epoch, tensorboard_step)
                    # evaluation
                    if args.evaluate_during_training:
                        logger.info("Perform validation at step: %d" % tensorboard_step)
                        val_loss, val_acc = validate(args, model, tokenizer)

                        # update tensorboard
                        writer.add_scalar("Loss/Val_Loss", val_loss, tensorboard_step)
                        writer.add_scalar("Accuracy/Val_Accuracy", val_acc, tensorboard_step)

    return global_step, global_loss / global_step


def validate(args, model, tokenizer):
    val_dataset = build_dataset(op.join(args.data_dir, args.val_yaml), tokenizer, args, is_train=True)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = DataLoader(val_dataset, sampler=val_sampler,
                                  batch_size=int(args.train_batch_size/2), num_workers=args.num_workers)
    global_loss = global_acc = 0
    global_step = 0
    for step, (img_keys, batch) in enumerate(tqdm(train_dataloader)):
        batch = tuple(t.to(args.device) for t in batch)

        model.eval()
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                  'token_type_ids': batch[2], 'img_feats': batch[3],
                  'masked_pos': batch[4], 'masked_ids': batch[5],
                  'personality_ids': batch[6],
                  }
        outputs = model(**inputs)
        loss, logits = outputs[:2]
        masked_ids = inputs['masked_ids']
        masked_ids = masked_ids[masked_ids != 0]
        batch_score = compute_score_with_logits(logits, masked_ids)
        batch_acc = torch.sum(batch_score.float()) / torch.sum(inputs['masked_pos'])

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        global_step += 1
        global_loss += loss.item()
        global_acc += batch_acc
    return global_loss / global_step, global_acc / global_step


def get_predict_file(output_dir, yaml_file, args):
    cc = ['pred']
    # make sure it works with/without / in end of the path.
    data = op.basename(op.join(args.data_dir, '')[:-1])
    split = op.basename(yaml_file)
    assert split.endswith('.yaml')
    split = split[:-5]
    cc.append(data)
    cc.append(split)
    cc.append('beam{}'.format(args.num_beams))
    cc.append('max{}'.format(args.max_gen_length))
    if args.add_od_labels:
        cc.append('odlabels')
    if args.num_keep_best != 1:
        cc.append('best{}'.format(args.num_keep_best))
    if args.output_hidden_states:
        cc.append('hidden')
    return op.join(output_dir, '{}.tsv'.format('.'.join(cc)))


def get_evaluate_file(predict_file):
    assert predict_file.endswith('.tsv')
    fpath = op.splitext(predict_file)[0]
    return fpath + '.eval.json'


def get_evaluate_method(predict_file):
    if 'nocaps' in op.basename(predict_file):
        return 'nocaps'
    else:
        return 'coco'


def evaluate(args, val_dataset, model, tokenizer, output_dir):
    assert op.isdir(output_dir)
    predict_file = get_predict_file(output_dir, val_dataset.yaml_file, args)
    if op.isfile(predict_file):
        logger.info('Skip predict. {} already exists'.format(predict_file))
    else:
        test(args, val_dataset, model, tokenizer, predict_file)

    evaluate_file = get_evaluate_file(predict_file)
    if op.isfile(evaluate_file):
        logger.info('Skip evaluation. {} already exists'.format(evaluate_file))
        return evaluate_file

    eval_method = get_evaluate_method(predict_file)
    if eval_method == 'coco':
        gt_file = val_dataset.get_caption_file_in_coco_format()
        result = evaluate_on_coco_caption(predict_file, gt_file, outfile=evaluate_file)
    else:
        split = 'val' if 'val' in op.basename(val_dataset.yaml_file) else 'test'
        result = evaluate_on_nocaps(split, predict_file,
                                    data_dir=args.data_dir, evaluate_file=evaluate_file)
    logger.info("evaluation result: {}".format(str(result)))
    return evaluate_file


def test(args, test_dataset, model, tokenizer, predict_file):
    args.test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    test_sampler = SequentialSampler(test_dataset)
    cache_file = predict_file

    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                 batch_size=args.test_batch_size, num_workers=args.num_workers)

    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token,
                                         tokenizer.sep_token, tokenizer.pad_token, tokenizer.mask_token, '.']
                                        )
    model.eval()

    def gen_rows():
        time_meter = 0
        # restore existing results for long running inference tasks
        exist_key2pred = {}
        tmp_file = cache_file + '.tmp.copy'
        if op.isfile(tmp_file):
            with open(tmp_file, 'r') as fp:
                for line in fp:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        exist_key2pred[parts[0]] = parts[1]

        with torch.no_grad():
            for step, (img_keys, batch) in enumerate(tqdm(test_dataloader)):
                is_exist = True
                for k in img_keys:
                    if k not in exist_key2pred:
                        is_exist = False
                        break
                if is_exist:
                    for k in img_keys:
                        yield k, exist_key2pred[k]
                    continue
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'is_decode': True,
                          'input_ids': batch[0], 'attention_mask': batch[1],
                          'token_type_ids': batch[2], 'img_feats': batch[3],
                          'masked_pos': batch[4],
                          'personality_ids': batch[5],
                          'do_sample': False,
                          'bos_token_id': cls_token_id,
                          'pad_token_id': pad_token_id,
                          'eos_token_ids': [sep_token_id, pad_token_id],
                          'mask_token_id': mask_token_id,
                          # for adding od labels
                          'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,

                          # hyperparameters of beam search
                          'max_length': args.max_gen_length,
                          'num_beams': args.num_beams,
                          "temperature": args.temperature,
                          "top_k": args.top_k,
                          "top_p": args.top_p,
                          "repetition_penalty": args.repetition_penalty,
                          "length_penalty": args.length_penalty,
                          "num_return_sequences": args.num_return_sequences,
                          "num_keep_best": args.num_keep_best,
                          }
                tic = time.time()
                # captions, logprobs
                outputs = model(**inputs)
                time_meter += time.time() - tic
                all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                all_confs = torch.exp(outputs[1])

                for img_key, caps, confs in zip(img_keys, all_caps, all_confs):
                    res = []
                    for cap, conf in zip(caps, confs):
                        cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                        res.append({'caption': cap, 'conf': conf.item()})
                    if isinstance(img_key, torch.Tensor):
                        img_key = img_key.item()
                    yield img_key, json.dumps(res)

        logger.info("Inference model computing time: {} seconds per batch".format(time_meter / (step + 1)))

    tsv_writer(gen_rows(), cache_file)
    return predict_file


def restore_training_settings(args):
    assert not args.do_train
    assert args.do_test or args.do_eval
    # restore training settings, check hasattr for backward compatibility
    train_args = torch.load(op.join(args.eval_model_dir, 'training_args.bin'))
    if hasattr(train_args, 'max_seq_a_length'):
        max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
        max_seq_length = args.max_gen_length + max_od_labels_len
        args.max_seq_length = max_seq_length
        logger.warning('Override max_seq_length to {} = max_gen_length:{} + od_labels_len:{}'.format(
            max_seq_length, args.max_gen_length, max_od_labels_len))

    override_params = ['max_seq_a_length', 'do_lower_case', 'add_od_labels',
                       'max_img_seq_length', 'img_feature_dim',
                       'img_feature_type']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                                                                              test_v, train_v))
                setattr(args, param, train_v)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='datasets/coco_caption', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--train_yaml", default='train.yaml', type=str, required=False,
                        help="yaml file for training.")
    parser.add_argument("--test_yaml", default='test.yaml', type=str, required=False,
                        help="yaml file for testing.")
    parser.add_argument("--val_yaml", default='val.yaml', type=str, required=False,
                        help="yaml file used for validation during training.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or model type.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--loss_type", default='sfmx', type=str,
                        help="Loss function types: support kl, x2, sfmx")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=70, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded.")
    parser.add_argument("--max_seq_a_length", default=40, type=int,
                        help="The maximum sequence length for caption.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Probability to mask input sentence during training.")
    parser.add_argument("--max_masked_tokens", type=int, default=3,
                        help="The max number of masked tokens per sentence.")
    parser.add_argument("--add_od_labels", default=False, action='store_true',
                        help="Whether to add object detection labels or not")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=50, type=int,
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int,
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int,
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear or")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=40, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1,
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")

    # for tensorboard
    parser.add_argument('--global_step_offset', type=int, default=0, help="global step offset for logging to directory")
    parser.add_argument('--tensorboard_log_dir', type=str, default='runs/pic', help="directory to store runs info")

    # for generation
    parser.add_argument("--eval_model_dir", type=str, default='',
                        help="Model directory for evaluation.")
    parser.add_argument('--max_gen_length', type=int, default=40,
                        help="max length of generated sentences")
    parser.add_argument('--output_hidden_states', action='store_true',
                        help="Turn on for fast decoding")
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help="repeating times per image")
    parser.add_argument('--num_beams', type=int, default=5, help="beam search width")
    parser.add_argument('--num_keep_best', type=int, default=1,
                        help="number of hypotheses to keep in beam search")
    parser.add_argument('--temperature', type=float, default=1,
                        help="temperature in softmax for sampling")
    parser.add_argument('--top_k', type=int, default=0,
                        help="filter distribution for sampling")
    parser.add_argument('--top_p', type=float, default=1,
                        help="filter distribution for sampling")
    parser.add_argument('--repetition_penalty', type=int, default=1,
                        help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)")
    parser.add_argument('--length_penalty', type=int, default=1,
                        help="beam search length penalty")
    args = parser.parse_args()

    global logger

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    output_dir = args.output_dir
    mkdir(output_dir)

    logger = setup_logger("vlpretrain", output_dir, 0)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    set_seed(args.seed, args.n_gpu)

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = BertConfig, BertForPersonalityImageCaptioning, BertTokenizer
    if args.do_train:
        assert args.model_name_or_path is not None
        config = config_class.from_pretrained(args.config_name if args.config_name else \
                                                  args.model_name_or_path, num_labels=args.num_labels,
                                              finetuning_task='image_captioning')
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
                                                        else args.model_name_or_path, do_lower_case=args.do_lower_case)
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        config.output_hidden_states = args.output_hidden_states
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    if args.do_train:
        train_dataset = build_dataset(op.join(args.data_dir, args.train_yaml), tokenizer, args)
        global_step, avg_loss = train(args, train_dataset, model, tokenizer)
        logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)

    # inference and evaluation
    if args.do_test or args.do_eval:
        args = restore_training_settings(args)
        test_dataset = build_dataset(op.join(args.data_dir, args.test_yaml),
                                     tokenizer, args, is_train=False)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if not args.do_eval:
            # generate captions only
            predict_file = get_predict_file(checkpoint, test_dataset.yaml_file, args)
            test(args, test_dataset, model, tokenizer, predict_file)
            logger.info("Prediction results saved to: {}".format(predict_file))
        else:
            # add evaluation
            evaluate_file = evaluate(args, test_dataset, model, tokenizer,
                                     checkpoint)
            logger.info("Evaluation results saved to: {}".format(evaluate_file))


if __name__ == "__main__":
    main()
