from monai.networks.nets import UNet
from monai.networks.layers import Norm



# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
def get_new_model():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    return model

