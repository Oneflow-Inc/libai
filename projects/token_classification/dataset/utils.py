import csv
import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union


class EncodePattern(Enum):
    """encode pattern
    bert pattern:
        single sentence: [CLS] A [SEP]
        pair of sentences: [CLS] A [SEP] B [SEP]
    roberta/bart pattern:
        single sentence: <s> A </s>
        pair of sentences: <s> A </s> </s> B </s>
    """

    bert_pattern = "S*E*E"
    roberta_pattern = "S*EE*E"


@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in `[0, 1]`: Usually `1` for tokens that are NOT MASKED,
            `0` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    labels: Optional[Union[int, float]] = None


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of [`InputExample`] for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of [`InputExample`] for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of [`InputExample`] for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json list file."""
        with open(input_file, "r") as f:
            reader = f.readlines()
            lines = []
            for line in reader:
                lines.append(json.loads(line.strip()))
            return lines
            
    @classmethod
    def _read_text(self,input_file):
        
        lines = []
        with open(input_file,'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words":words,"labels":labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words":words,"labels":labels})
        return lines
