from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
from pathlib import Path
from monai import transforms

def get_dataset(dataset_path: Path, config):
    datalist = load_decathlon_datalist(dataset_path)
    keys= ["image", "label"]

    train_transforms = transforms.Compose([
        transforms.LoadImaged(keys=keys, ensure_channel_first=True),
        transforms.Orientationd(keys=keys, axcodes="RAS"),
        transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True
            ),
        transforms.SpatialPadd(keys=keys, spatial_size=[224, 224, 128]),
        transforms.RandSpatialCropSamplesd(
        keys=keys,
        roi_size=[224, 224, 128],
        num_samples=4,
        random_center=True,
        random_size=False,
        ),
        transforms.ToTensord(keys=keys),
    ])
    train_ds = Dataset(data=datalist, transform=train_transforms)
    return train_ds, train_transforms

def get_test_dataset(dataset_path: Path):
    datalist = load_decathlon_datalist(dataset_path, data_list_key="test")
    keys= ["image", "label"]

    test_transforms = transforms.Compose([
        transforms.LoadImaged(keys=keys, ensure_channel_first=True),
        transforms.Orientationd(keys=keys, axcodes="RAS"),
        transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True
            ),
        transforms.SpatialPadd(keys=keys, spatial_size=[224, 224, 128]),
        transforms.RandSpatialCropSamplesd(
        keys=keys,
        roi_size=[224, 224, 128],
        num_samples=4,
        random_center=True,
        random_size=False,
        ),
        transforms.ToTensord(keys=keys),
    ])
    ds = Dataset(data=datalist, transform=test_transforms)
    return ds, test_transforms