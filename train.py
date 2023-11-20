from pathlib import Path
import torch
from monai.data import DataLoader
import tqdm
from monai.metrics import DiceMetric

from data import get_dataset, get_test_dataset
from loss import get_loss_func
from model import get_new_model

LEARNING_RATE = 1e-4
BATCH_SIZE=4
EPOCHS=10

def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Running on {device}")
    dataset_folder = Path("/home/espenbfo/datasets/segmentation/Task07_Pancreas/dataset.json")
    dataset, train_transforms = get_dataset(dataset_folder, None)
    test_dataset, test_transforms = get_test_dataset(dataset_folder)

    model = get_new_model().to(device)

    loss_func = get_loss_func()

    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    dataloader = DataLoader(dataset, num_workers=4, batch_size=BATCH_SIZE)
    dataloader_test = DataLoader(test_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    loss_epochs = []
    test_loss_epochs = []
    test_metric_epochs = []
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch + 1 } of {EPOCHS}")
        print("Train:")
        model.train()
        total_loss = 0
        for index, batch in (pbar := tqdm.tqdm(enumerate(dataloader), total=len(dataloader))):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            pred = model(inputs)

            loss = loss_func(pred, labels)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()
            total_loss += loss.detach().cpu()
            pbar.set_postfix({"loss": f"{total_loss/(index+1):.3f}"})
        loss_epochs.append(total_loss/len(dataloader))

        print("Eval:")
        model.eval()
        with torch.no_grad():
            total_test_loss = 0
            total_test_metric = 0
            for index, batch in (pbar := tqdm.tqdm(enumerate(dataloader), total=len(dataloader))):
                inputs, labels = batch["image"].to(device), batch["label"].to(device)
                pred = model(inputs)

                loss = loss_func(pred, labels)
                total_test_loss += loss.detach().cpu()
                total_test_metric += dice_metric(y_pred=pred, y=labels).aggregate().item.detach().cpu()
                pbar.set_postfix({"loss": f"{total_test_loss/(index+1):.3f}", "dice metric": f"{total_test_metric/(index+1):.3f}"})
            test_loss_epochs.append(total_test_loss)
            test_metric_epochs.append(total_test_metric)
        print("")
    torch.save(model.state_dict, "final_weights.pt")
if __name__ == "__main__":
    main()