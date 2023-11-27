from monai.losses import DiceLoss, GeneralizedDiceLoss

def get_loss_func(loss_smoothing=1e-5):
    loss_function = DiceLoss(include_background=True, to_onehot_y=True, softmax=True, smooth_dr=loss_smoothing, smooth_nr=loss_smoothing)
    return loss_function