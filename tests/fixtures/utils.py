from libai.utils.download import download

fixtrue_urls = {
    "sample_text.txt": "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/fixtures/sample_text.txt",  # noqa
    "spiece.model": "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/fixtures/spiece.model",  # noqa
    "test_sentencepiece.model": "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/fixtures/test_sentencepiece.model",  # noqa
}

BASE_DIR = "tests/fixtures"


def get_fixtures(fixture_dir):
    fixture_name = fixture_dir.split("/")[-1]
    if fixture_name not in fixtrue_urls:
        raise RuntimeError("{} not available in LiBai tests fixtrues!".format(fixture_name))
    return download(fixtrue_urls[fixture_name], BASE_DIR)
