# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

from .utils import DataProcessor, EncodePattern, InputExample, InputFeatures

logger = logging.getLogger(__name__)


def clue_convert_examples_to_features(
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
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_length - added_special_tokens[1])
        else:
            if len(tokens_a) > max_length - added_special_tokens[0]:
                tokens_a = tokens_a[: (max_length - added_special_tokens[0])]

        if pattern is EncodePattern.bert_pattern:
            tokens = start_token + tokens_a + end_token
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

        label = None
        if example.label is not None:
            if output_mode == "classification":
                label = label_map[example.label]
            elif output_mode == "regression":
                label = float(example.label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

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


class TnewsProcessor(DataProcessor):
    """Processor for the TNEWS data set (CLUE version).
    Single sentence classification task.
    The task is to predict which category the title belongs to.
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        labels = []
        for i in range(17):
            if i == 5 or i == 11:
                continue
            labels.append(str(100 + i))
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line["sentence"]
            label = None if set_type == "test" else str(line["label"])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class IflytekProcessor(DataProcessor):
    """Processor for the IFLYTEK data set (CLUE version).
    Single sentence classification task.
    The task is to predict the categories according to discription.
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        labels = []
        for i in range(119):
            labels.append(str(i))
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line["sentence"]
            label = None if set_type == "test" else str(line["label"])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class AfqmcProcessor(DataProcessor):
    """Processor for the AFQMC data set (CLUE version).
    Sentence pair classification task.
    This task is to predict whether two sentences are semantically similar.
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line["sentence1"]
            text_b = line["sentence2"]
            label = None if set_type == "test" else str(line["label"])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class OcnliProcessor(DataProcessor):
    """Processor for the OCNLI data set (CLUE version).
    Sentence pair classification task.
    Given a premise sentence and a hypothesis sentence,
    the task is to predict whether the premise entails the hypothesis (entailment),
    contradicts the hypothesis (contradiction), or neither (neutral).
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line["sentence1"]
            text_b = line["sentence2"]
            label = None if set_type == "test" else str(line["label"])
            if label.strip() == "-":
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class CmnliProcessor(DataProcessor):
    """Processor for the CMNLI data set (CLUE version).
    Sentence pair classification task.
    Given a premise sentence and a hypothesis sentence,
    the task is to predict whether the premise entails the hypothesis (entailment),
    contradicts the hypothesis (contradiction), or neither (neutral).
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line["sentence1"]
            text_b = line["sentence2"]
            label = None if set_type == "test" else str(line["label"])
            if label.strip() == "-":
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class CslProcessor(DataProcessor):
    """Processor for the CSL data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = " ".join(line["keyword"])
            text_b = line["abst"]
            label = None if set_type == "test" else str(line["label"])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WscProcessor(DataProcessor):
    """Processor for the WSC data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["true", "false"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line["text"]
            text_a_list = list(text_a)
            target = line["target"]
            query = target["span1_text"]
            query_idx = target["span1_index"]
            pronoun = target["span2_text"]
            pronoun_idx = target["span2_index"]
            assert (
                text_a[pronoun_idx : (pronoun_idx + len(pronoun))] == pronoun
            ), "pronoun: {}".format(pronoun)
            assert text_a[query_idx : (query_idx + len(query))] == query, "query: {}".format(query)
            if pronoun_idx > query_idx:
                text_a_list.insert(query_idx, "_")
                text_a_list.insert(query_idx + len(query) + 1, "_")
                text_a_list.insert(pronoun_idx + 2, "[")
                text_a_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
            else:
                text_a_list.insert(pronoun_idx, "[")
                text_a_list.insert(pronoun_idx + len(pronoun) + 1, "]")
                text_a_list.insert(query_idx + 2, "_")
                text_a_list.insert(query_idx + len(query) + 2 + 1, "_")
            text_a = "".join(text_a_list)
            label = None if set_type == "test" else str(line["label"])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class CopaProcessor(DataProcessor):
    """Processor for the COPA data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            i = 2 * i
            guid1 = f"{set_type}-{i}"
            guid2 = "%s-%s" % (set_type, i + 1)
            premise = line["premise"]
            choice0 = line["choice0"]
            label = None if set_type == "test" else str(1 if line["label"] == 0 else 0)
            choice1 = line["choice1"]
            label2 = None if set_type == "test" else str(1 if line["label"] == 0 else 0)
            if line["question"] == "effect":
                text_a = premise
                text_b = choice0
                text_a2 = premise
                text_b2 = choice1
            elif line["question"] == "cause":
                text_a = choice0
                text_b = premise
                text_a2 = choice1
                text_b2 = premise
            else:
                raise ValueError(f'unknowed {line["question"]} type')
            examples.append(InputExample(guid=guid1, text_a=text_a, text_b=text_b, label=label))
            examples.append(InputExample(guid=guid2, text_a=text_a2, text_b=text_b2, label=label2))
        return examples

    def _create_examples_version2(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            if line["question"] == "cause":
                text_a = line["premise"] + "这是什么原因造成的？" + line["choice0"]
                text_b = line["premise"] + "这是什么原因造成的？" + line["choice1"]
            else:
                text_a = line["premise"] + "这造成了什么影响？" + line["choice0"]
                text_b = line["premise"] + "这造成了什么影响？" + line["choice1"]
            label = None if set_type == "test" else str(1 if line["label"] == 0 else 0)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


clue_tasks_num_labels = {
    "iflytek": 119,
    "cmnli": 3,
    "ocnli": 3,
    "afqmc": 2,
    "csl": 2,
    "wsc": 2,
    "copa": 2,
    "tnews": 15,
}

clue_processors = {
    "tnews": TnewsProcessor,
    "iflytek": IflytekProcessor,
    "cmnli": CmnliProcessor,
    "ocnli": OcnliProcessor,
    "afqmc": AfqmcProcessor,
    "csl": CslProcessor,
    "wsc": WscProcessor,
    "copa": CopaProcessor,
}

clue_output_modes = {
    "tnews": "classification",
    "iflytek": "classification",
    "cmnli": "classification",
    "ocnli": "classification",
    "afqmc": "classification",
    "csl": "classification",
    "wsc": "classification",
    "copa": "classification",
}
