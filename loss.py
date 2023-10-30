from monai.losses import DiceLoss

def get_loss_func():
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    return loss_function