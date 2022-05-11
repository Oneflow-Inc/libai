import argparse

from libai.utils.file_utils import get_data_from_cache

VOCAB_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/gpt_dataset/gpt2-vocab.json"  # noqa
MERGE_FILE_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/gpt_dataset/gpt2-merges.txt"  # noqa
BIN_DATA_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.bin"  # noqa
IDX_DATA_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.idx"  # noqa

VOCAB_MD5 = "dffec25a898b1f5e569bec4dffd7e5c0"
MERGE_FILE_MD5 = "75a37753dd7a28a2c5df80c28bf06e4e"
BIN_DATA_MD5 = "b842467bd5ea7e52f7a612ea6b4faecc"
IDX_DATA_MD5 = "cf5963b8543f0a7a867361eb980f0372"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", default="./gpt_dataset", type=str, help="The output path to store data"
    )
    args = parser.parse_args()
    cache_dir = args.output

    get_data_from_cache(VOCAB_URL, cache_dir, md5=VOCAB_MD5)
    get_data_from_cache(MERGE_FILE_URL, cache_dir, md5=MERGE_FILE_MD5)
    get_data_from_cache(BIN_DATA_URL, cache_dir, md5=BIN_DATA_MD5)
    get_data_from_cache(IDX_DATA_URL, cache_dir, md5=IDX_DATA_MD5)


if __name__ == "__main__":
    main()
