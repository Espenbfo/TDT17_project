from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
from pathlib import Path
from monai import transforms
from sklearn.model_selection import train_test_split

def get_dataset(dataset_path: Path, config):
    datalist = load_decathlon_datalist(dataset_path)
    keys= ["image", "label"]

    train_data, val_data = train_test_split(datalist, train_size=0.8, random_state=20)

    train_transforms = transforms.Compose([
        transforms.LoadImaged(keys=keys, ensure_channel_first=True),
        transforms.Orientationd(keys=keys, axcodes="RAS"),
        transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True
            ),
        transforms.SpatialPadd(keys=keys, spatial_size=[96, 96, 96]),
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=16,
            image_key="image",
            image_threshold=0,
        ),
        transforms.ToTensord(keys=keys),    
    ])

    val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=keys, ensure_channel_first=True),
        transforms.Orientationd(keys=keys, axcodes="RAS"),
        transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True
            ),
        
        transforms.ToTensord(keys=keys),    
    ]
)

    train_ds = Dataset(data=train_data, transform=train_transforms)
    val_ds = Dataset(data=val_data, transform=val_transforms)
    return train_ds, val_ds

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