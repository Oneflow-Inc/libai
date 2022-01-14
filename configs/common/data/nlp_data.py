from libai.config import LazyCall
from libai.data.build import build_nlp_test_loader, build_nlp_train_val_test_loader
from tests.data.datasets.demo_dataset import DemoNlpDataset


dataloader = dict(
    # Pad the vocab size to be divisible by this value
    # This is added for computational efficiency reasons.
    train = LazyCall(build_nlp_train_val_test_loader)(
        dataset = [
            LazyCall(DemoNlpDataset)(
                data_root="train1",
            ),
            LazyCall(DemoNlpDataset)(
                data_root="train2",
            ),
        ],
        splits=[[949., 50., 1.], [900., 99., 1.]],
        weights=[0.5, 0.5],
        batch_size=32,
        num_workers=4,
    ),
    test = [
        LazyCall(build_nlp_test_loader)(
            dataset=LazyCall(DemoNlpDataset)(
                data_root="test1",
            ),
            batch_size=32,
        ),
        LazyCall(build_nlp_test_loader)(
            dataset=LazyCall(DemoNlpDataset)(
                data_root="test2",
            ),
            batch_size=32,
        )
    ],
)