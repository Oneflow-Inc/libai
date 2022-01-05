data = dict(
    # Path to dataset
    data_path = "./dataset/imagenet",
    # Input image size
    img_size = 224,
    # Interpolation to resize image (random, bilinear, bicubic)
    interpolation = "bicubic",
    # Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
    pin_memory = True,
    # Number of data loading threads
    num_workers=8,
    # Augmentation settings
    augmentation = dict(
        # Color jitter factor
        color_jitter = 0.4,
        # Use AutoAugment policy. "v0" or "original"
        auto_augment = "rand-m9-mstd0.5-inc1",
        # Random erase prob
        reprob = 0.25,
        # Random erase mode
        remode = "pixel",
        # Random erase count
        recount = 1,
        # Mixup alpha, mixup enabled if > 0
        mixup = 0.8,
        # Cutmix alpha, cutmix enabled if > 0
        cutmix = 1.0,
        # Cutmix min/max ratio, overrides alpha and enables cutmix if set
        cutmix_minmax = None,
        # Probability of performing mixup or cutmix when either/both is enabled
        mixup_prob = 1.0,
        # Probability of switching to cutmix when both mixup and cutmix enabled
        mixup_switch_prob = 0.5,
        # How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
        mixup_mode = "batch"
    ),
    test = dict(
        # Whether to use center crop when testing
        crop = True,
        # Whether to use SequentialSampler as validation sampler
        sequential = False,
    )
)