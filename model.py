from monai.networks.nets import UNet
from monai.networks.layers import Norm, Act
from monai.inferers import sliding_window_inference


# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
def get_new_model():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        act=Act.GELU
    )
    return model

def run_validation(model, batch, roi_size=(96, 96, 96), sliding_batch=16):  
    return sliding_window_inference(batch, roi_size, sliding_batch, model)