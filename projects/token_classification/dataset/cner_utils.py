import logging
import os
import pdb

from .utils import DataProcessor, EncodePattern, InputExample, InputFeatures

logger = logging.getLogger(__name__)


def cner_convert_examples_to_features(
    examples,
    tokenizer,
    max_length,
    task=None,
    pattern=EncodePattern.bert_pattern,
    label_list=None,
    output_mode=None,
):
    if task is not None:
        processor = clue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info(f"Using label list {label_list} for task {task}")
        if output_mode is None:
            output_mode = clue_output_modes[task]
            logger.info(f"Using output mode {output_mode} for task {task}")


    label_map = {label: i for i, label in enumerate(label_list)}
    pdb.set_trace()

    start_token = [] if tokenizer.start_token is None else [tokenizer.start_token]
    end_token = [] if tokenizer.end_token is None else [tokenizer.end_token]
    pad_id = tokenizer.pad_token_id

    if pattern == EncodePattern.bert_pattern:
        added_special_tokens = [2, 3]
    elif pattern == EncodePattern.roberta_pattern:
        added_special_tokens = [2, 4]
    else:
        raise KeyError("pattern is not a valid EncodePattern")

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        if isinstance(example.text_a,list):
            example.text_a = " ".join(example.text_a)
        tokens_a = tokenizer.tokenize(example.text_a)
        label = [label_map[x] for x in example.label]
 

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_length - added_special_tokens[1])
        else:
            if len(tokens_a) > max_length - added_special_tokens[0]:
                tokens_a = tokens_a[: (max_length - added_special_tokens[0])]

        if pattern is EncodePattern.bert_pattern:
            tokens = start_token + tokens_a + end_token
            label = [label_map['O']] + label+ [label_map['O']]
            token_type_ids = [0] * len(tokens)
            if tokens_b:
                tokens += tokens_b + end_token
                token_type_ids += [1] * (len(tokens) - len(token_type_ids))
        elif pattern is EncodePattern.roberta_pattern:
            tokens = start_token + tokens_a + end_token
            token_type_ids = [0] * len(tokens)
            if tokens_b:
                tokens += end_token + tokens_b + end_token
                token_type_ids += [1] * (len(tokens) - len(token_type_ids))
        else:
            raise KeyError("pattern is not a valid EncodePattern")

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        label = label + ([0] * padding_length)
        # pdb.set_trace()

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label)) [label_map[x] for x in example.labels]

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=label,
            )
        )
       

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class CnerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        """See base class."""
        return ["X",'B-CONT','B-EDU','B-LOC','B-NAME','B-ORG','B-PRO','B-RACE','B-TITLE',
                'I-CONT','I-EDU','I-LOC','I-NAME','I-ORG','I-PRO','I-RACE','I-TITLE',
                'O','S-NAME','S-ORG','S-RACE',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a,text_b=None, label=labels))
        return examples


clue_tasks_num_labels = {
    "cner": 23
}

cner_processors = {
    "cner":CnerProcessor
}

cner_output_modes = {
    "cner":"classification"
}
