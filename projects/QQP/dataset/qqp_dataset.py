import oneflow as flow
from libai.data.data_samplers import MegatronPretrainingSampler, MegatronPretrainingRandomSampler
from libai.data.build import cyclic_iter
from libai.utils import distributed as dist
from .data_utils import clean_text
from .data import GLUEAbstractDataset
from libai.tokenizer import get_tokenizer
import logging

logger = logging.getLogger("libai."+__name__)

def build_pretraining_data_loader(cfg, dataset, consumed_samples, dataloader_type="single", drop_last=True):
    """Build dataloader given an input dataset."""

    if dataset is None:
        return None

    # print(f"rank: {flow.env.get_rank()} and data parallel rank: {dist.get_data_parallel_rank()}")
    # Megatron sampler
    if dataloader_type == "single":
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=cfg.train.micro_batch_size,
            data_parallel_rank=dist.get_data_parallel_rank(),
            data_parallel_size=dist.get_data_parallel_size(),
            drop_last=drop_last
        )
    elif dataloader_type == "cyclic":
        batch_sampler = MegatronPretrainingRandomSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=cfg.train.micro_batch_size,
            data_parallel_rank=dist.get_data_parallel_rank(),
            data_parallel_size=dist.get_data_parallel_size(),
        )
    else:
        raise Exception(
            "{} dataloader type is not supported.".format(dataloader_type)
        )

    # Torch dataloader.
    return flow.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=cfg.data.num_workers,
    )


def train_valid_test_datasets_provider(cfg):
    tokenizer = get_tokenizer()

    train_dataset = QQPDataset('training', cfg.train.train_data,
                               tokenizer, cfg.data.seq_length)
    valid_dataset = QQPDataset('validation', cfg.train.valid_data,
                               tokenizer, cfg.data.seq_length)
    return train_dataset, valid_dataset, None

def build_train_valid_test_data_iterators(cfg):
    """通过外部定义的build_train_valid_test_datasets_provider函数，
        1. 先计算train_val_test_num_samples；
        2. 然后传入该函数中生成数据集；
        3. 由数据集生成iterator"""

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    logger.info("> building train, validation, and test datasets ...")

    # Backward compatibility, assume fixed batch size.
    if cfg.train.start_iter > 0 and cfg.train.consumed_train_samples == 0:
        assert (
            cfg.train.train_samples is None
        ), "only backward compatibility support for iteration-based training"
        cfg.train.consumed_train_samples = (
            cfg.train.start_iter * cfg.train.global_batch_size
        )
    if cfg.train.start_iter > 0 and cfg.train.consumed_valid_samples == 0:
        if cfg.train.train_samples is None:
            cfg.train.consumed_valid_samples = (
                (cfg.train.start_iter // cfg.train.eval_period)
                * cfg.train.eval_iter
                * cfg.train.global_batch_size
            )

    # args中参数：
    #   global_batch_size：如果没设置，则为args.micro_batch_size * args.data_parallel_size，代表全局的 batch_size，
    #   主要作用是为了计算总的样本数或者是已经训练的样本数，每张卡上的 batch_size 是通过 micro_batch_size
    #   args.iteration是在初始化模型时setup_model_and_optimizer设置为当前已经训练的step，若为新的则为0
    #   args.train_samples为args的初始设置，如果没有则采用iter based训练，需要用args.train_iters

    # Data loader only on rank 0 of each model parallel group.
    # if mpu.get_tensor_model_parallel_rank() == 0:

    # Number of train/valid/test samples.
    if cfg.train.train_samples:
        train_samples = cfg.train.train_samples
    else:
        train_samples = cfg.train.train_iter * cfg.train.global_batch_size
    eval_iter = (
        cfg.train.train_iter // cfg.train.eval_period + 1
    ) * cfg.train.eval_iter
    test_iter = cfg.train.eval_iter
    train_val_test_num_samples = [
        train_samples,
        eval_iter * cfg.train.global_batch_size,
        test_iter * cfg.train.global_batch_size,
    ]
    logger.info(" > datasets target sizes (minimum size):")
    logger.info("    train:      {}".format(train_val_test_num_samples[0]))
    logger.info("    validation: {}".format(train_val_test_num_samples[1]))
    logger.info("    test:       {}".format(train_val_test_num_samples[2]))

    # Build the datasets.
    train_ds, valid_ds, test_ds = train_valid_test_datasets_provider(cfg)

    # Build dataloders.
    train_dataloader = build_pretraining_data_loader(
        cfg, train_ds, cfg.train.consumed_train_samples, dataloader_type = "cyclic"
    )
    valid_dataloader = build_pretraining_data_loader(
        cfg, valid_ds, cfg.train.consumed_valid_samples, dataloader_type = "single", drop_last=False
    )
    test_dataloader = build_pretraining_data_loader(cfg, test_ds, 0, dataloader_type = "single", drop_last=False)

    # Flags to know if we need to do training/validation/testing.
    do_train = train_dataloader is not None and cfg.train.train_iter > 0
    do_valid = valid_dataloader is not None and cfg.train.eval_iter > 0
    do_test = test_dataloader is not None and cfg.train.eval_iter > 0
    # Need to broadcast num_tokens and num_type_tokens.
    flags = flow.tensor(
        [int(do_train), int(do_valid), int(do_test)], dtype=flow.long, device="cuda"
    )
    # flags = torch.cuda.LongTensor(
    #     [int(do_train), int(do_valid), int(do_test)])
    # else:
    #     flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    # torch.distributed.broadcast(flags,
    #                             mpu.get_tensor_model_parallel_src_rank(),  # 获取当前全局rank对应的tp组的第一个local rank
    #                             group=mpu.get_tensor_model_parallel_group())
    cfg.train.do_train = flags[0].item()
    cfg.train.do_valid = flags[1].item()
    cfg.train.do_test = flags[2].item()

    # Build iterators.
    dl_type = cfg.data.dataloader_type
    assert dl_type in ["single", "cyclic"]

    # if train_dataloader is not None:
    #     train_data_iterator = (
    #         iter(train_dataloader)
    #         if dl_type == "single"
    #         else iter(cyclic_iter(train_dataloader))
    #     )
    # else:
    #     train_data_iterator = None

    # if valid_dataloader is not None:
    #     valid_data_iterator = (
    #         iter(valid_dataloader)
    #         if dl_type == "single"
    #         else iter(cyclic_iter(valid_dataloader))
    #     )
    # else:
    #     valid_data_iterator = None

    # if test_dataloader is not None:
    #     test_data_iterator = (
    #         iter(test_dataloader)
    #         if dl_type == "single"
    #         else iter(cyclic_iter(test_dataloader))
    #     )
    # else:
    #     test_data_iterator = None

    return train_dataloader, valid_dataloader, test_dataloader


LABELS = [0, 1]


class QQPDataset(GLUEAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length,
                 test_label=0):
        self.test_label = test_label
        super().__init__('QQP', name, datapaths,
                         tokenizer, max_seq_length)

    def process_samples_from_single_path(self, filename):
        """"Implement abstract method."""
        logger.info(' > Processing {} ...'.format(filename))

        samples = []
        total = 0
        first = True
        is_test = False
        with open(filename, 'r') as f:
            for line in f:
                row = line.strip().split('\t')
                if first:
                    first = False
                    if len(row) == 3:
                        is_test = True
                        logger.info('   reading {}, {}, and {} columns and '
                                     'setting labels to {}'.format(
                                         row[0].strip(), row[1].strip(),
                                         row[2].strip(), self.test_label))
                    else:
                        assert len(row) == 6
                        logger.info('    reading {}, {}, {}, and {} columns'
                                     ' ...'.format(
                                         row[0].strip(), row[3].strip(),
                                         row[4].strip(), row[5].strip()))
                    continue

                if is_test:
                    assert len(row) == 3, 'expected length 3: {}'.format(row)
                    uid = int(row[0].strip())
                    text_a = clean_text(row[1].strip())
                    text_b = clean_text(row[2].strip())
                    label = self.test_label
                    assert len(text_a) > 0
                    assert len(text_b) > 0
                else:
                    if len(row) == 6:
                        uid = int(row[0].strip())
                        text_a = clean_text(row[3].strip())
                        text_b = clean_text(row[4].strip())
                        label = int(row[5].strip())
                    else:
                        logger.info('***WARNING*** index error, '
                                     'skipping: {}'.format(row))
                        continue
                    if len(text_a) == 0:
                        logger.info('***WARNING*** zero length a, '
                                     'skipping: {}'.format(row))
                        continue
                    if len(text_b) == 0:
                        logger.info('***WARNING*** zero length b, '
                                     'skipping: {}'.format(row))
                        continue
                assert label in LABELS
                assert uid >= 0

                sample = {'uid': uid,
                          'text_a': text_a,
                          'text_b': text_b,
                          'label': label}
                total += 1
                samples.append(sample)

                if total % 50000 == 0:
                    logger.info('  > processed {} so far ...'.format(total))

        logger.info(' >> processed {} samples.'.format(len(samples)))
        return samples
