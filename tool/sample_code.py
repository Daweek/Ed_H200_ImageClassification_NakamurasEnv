
# get adamw#
from timm.optim import create_optimizer_v2 

optinizer =create_optimizer_v2(
    model,
    opt="adamw"
    lr: Optional[float] = 5.0e-4,
    weight_decay: float = 0.05,
    eps= 1.0e-6)


# finetuningの transfomerの定義# 
from timm.data import create_transform

transform = create_transform(
    224,
    is_training=False,
    use_prefetcher=False,
    no_aug=False,
    scale=(0.08,1.0),
    ratio=(0.75,1.3333),
    hflip=0.5,
    vflip=0.,
    color_jitter=0.4,
    auto_augment="and-m9-mstd0.5-inc1",
    interpolation='bicubic',
    mean=( 0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    re_prob=0.25,
    re_mode='pixel',
    re_count=1
    )


# 事前学数 trasformeの定義
transform = create_transform(
    224,
    is_training=False,
    use_prefetcher=False,
    no_aug=True,
    scale=(1.0,1.0),
    ratio=(1.0,1.0),
    hflip=0.0,
    vflip=0.,
    color_jitter=0.0,
    interpolation='bicubic',
    mean=( 0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    re_prob=0,
    re_mode='pixel',
    re_count=1
    )