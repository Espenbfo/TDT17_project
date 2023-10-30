from pathlib import Path
import torch
from monai.data import DataLoader
import tqdm
from monai.metrics import DiceMetric

from data import get_dataset
from loss import get_loss_func
from model import get_new_model

LEARNING_RATE = 1e-4
BATCH_SIZE=1

def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    dataset_folder = Path("E:\datasets\Task06_Lung\dataset.json")
    dataset, train_transforms = get_dataset(dataset_folder, None)

    model = get_new_model().to(device)

    loss_func = get_loss_func()

    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    dataloader = DataLoader(dataset, num_workers=4, batch_size=BATCH_SIZE)

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    for batch in tqdm.tqdm(dataloader):
        inputs, labels = batch["image"].to(device), batch["label"].to(device)

        pred = model(inputs)

        loss = loss_func(pred, labels)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        print(loss.detach().cpu())


if __name__ == "__main__":
    main()