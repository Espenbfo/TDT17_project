from pathlib import Path
import torch
import numpy as np
from monai.data import DataLoader
import tqdm
from monai.metrics import DiceMetric
import json

from data import get_dataset, get_test_dataset
from loss import get_loss_func
from model import get_new_model, run_validation

LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-6
BATCH_SIZE=4
EPOCHS=100
MAX_LABEL_SMOOTHING = 1e3 # Epoch 1
MIN_LABEL_SMOOTHING = 1e-5
LABEL_SMOOTHING_EPOCHS = 50
EVAL_INTERVAL = 5
SAVE_INTERVAL = 25
METRIC_FILE_PATH = "metrics.json"

def label_transform(y):
    return y
def label_reverse_transform(x):
    return x

def save_metrics(train_loss, val_loss, val_metrics):
    dic = {"train_loss": train_loss, "val_loss": val_loss, "val_metrics": val_metrics}
    with open(METRIC_FILE_PATH, "w") as f:
        json.dump(dic, f)

def label_smoothing_schedule(current_epoch, max_epoch):
    return 1e-5
    current_epoch = min(current_epoch, max_epoch)
    return MIN_LABEL_SMOOTHING + (MAX_LABEL_SMOOTHING-MIN_LABEL_SMOOTHING)*(1+np.cos(current_epoch/max_epoch*np.pi))/2
def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Running on {device}")
    dataset_folder = Path("/home/espenbfo/datasets/segmentation/Task07_Pancreas/dataset.json")
    train_dataset, val_dataset = get_dataset(dataset_folder, None)

    model = get_new_model().to(device)


    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=MIN_LEARNING_RATE)
    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=1, shuffle=False)

    dice_metric = DiceMetric(include_background=False, reduction="mean", num_classes=3)
    loss_epochs = []
    val_loss_epochs = []
    val_metric_epochs = []
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch + 1 } of {EPOCHS}")
        print("Train:")
        model.train()
        total_loss = 0
        smoothing = label_smoothing_schedule(epoch, LABEL_SMOOTHING_EPOCHS)
        print(f"Using smoothing {smoothing:.2g}")

        loss_func = get_loss_func(smoothing)
        for index, batch in (pbar := tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            labels= label_transform(labels)
            pred = model(inputs)
            loss = loss_func(pred, labels)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()
            total_loss += loss.detach().cpu()
            pbar.set_postfix({"loss": f"{total_loss/(index+1):.3f}"})
            
        loss_epochs.append((epoch, float(total_loss/len(train_dataloader))))
        torch.save(model.state_dict(), "intermediate_weights.pt")

        scheduler.step()

        if (1+epoch)%EVAL_INTERVAL==0:
            print("Eval:")
            model.eval()
            with torch.no_grad():
                total_test_loss = 0
                total_test_metric = 0
                for index, batch in (pbar := tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader))):
                    inputs, labels = batch["image"].to(device), batch["label"].to(device)
                    pred = run_validation(model, inputs)
                    labels=label_transform(labels)
                    loss = loss_func(pred, labels)
                    total_test_loss += loss.detach().cpu()
                    dice_metric(y_pred=torch.round(torch.softmax(pred, dim=1)), y=labels)
                    pbar.set_postfix({"loss": f"{total_test_loss/(index+1):.3f}"})
                val_loss_epochs.append((1, float(total_test_loss/len(val_dataloader))))
                dice_metric_value = dice_metric.aggregate().item()
                dice_metric.reset()
                print(f"Dice metric value {dice_metric_value}")
                val_metric_epochs.append((1, float(dice_metric_value)))
            print("")
            save_metrics(loss_epochs, val_loss_epochs, val_metric_epochs)

        if (1+epoch)%SAVE_INTERVAL==0:
            torch.save(model.state_dict(), f"epoch_{epoch}_weights.pt")

    torch.save(model.state_dict(), "final_weights.pt")

if __name__ == "__main__":
    main()