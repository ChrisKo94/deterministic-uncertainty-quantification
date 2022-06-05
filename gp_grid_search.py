import random
import numpy as np

import torch
import torch.utils.data
from torch.nn import functional as F

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss

from ignite.contrib.handlers.tqdm_logger import ProgressBar

from utils.datasets import FastFashionMNIST, get_FashionMNIST
from utils.datasets import all_datasets
from utils.resnet_duq import ResNet_DUQ
from torchvision import transforms
from torchvision.models import resnet18


def train_model(l_gradient_penalty, length_scale, final_model):

    ds = all_datasets["Eurosat"]()

    input_size, num_classes, train_dataset, val_dataset, test_dataset = ds

    '''
    dataset = FastFashionMNIST("data/", train=True, download=True)
    test_dataset = FastFashionMNIST("data/", train=False, download=True)

    idx = list(range(60000))
    random.shuffle(idx)

    if final_model:
        train_dataset = dataset
        val_dataset = test_dataset
    else:
        train_dataset = torch.utils.data.Subset(dataset, indices=idx[:55000])
        val_dataset = torch.utils.data.Subset(dataset, indices=idx[55000:])
        
    num_classes = 10
    '''

    model_output_size = 512
    gamma = 0.999
    epochs = 50
    milestones = [15, 25, 35]
    feature_extractor = resnet18()

    feature_extractor.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    feature_extractor.maxpool = torch.nn.Identity()
    feature_extractor.fc = torch.nn.Identity()

    centroid_size = model_output_size

    model = ResNet_DUQ(
        feature_extractor,
        num_classes,
        centroid_size,
        model_output_size,
        length_scale,
        gamma
    )

    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4
    )

    def output_transform_bce(output):
        y_pred, y, _, _ = output
        return y_pred, y

    def output_transform_acc(output):
        y_pred, y, _, _ = output
        return y_pred, torch.argmax(y, dim=1)

    def output_transform_gp(output):
        y_pred, y, x, y_pred_sum = output
        return x, y_pred_sum

    def calc_gradient_penalty(x, y_pred_sum):
        gradients = torch.autograd.grad(
            outputs=y_pred_sum,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred_sum),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.flatten(start_dim=1)

        # L2 norm
        grad_norm = gradients.norm(2, dim=1)

        # Two sided penalty
        gradient_penalty = ((grad_norm - 1) ** 2).mean().float()

        return gradient_penalty

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch
        y = F.one_hot(y, num_classes=10).float()

        x, y = x.cuda(), y.cuda()

        x.requires_grad_(True)

        y_pred = model(x)

        loss = F.binary_cross_entropy(y_pred, y, reduction='mean')
        loss += l_gradient_penalty * calc_gradient_penalty(x, y_pred.sum(1))

        x.requires_grad_(False)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            model.update_embeddings(x, y)

        return loss.item()

    def eval_step(engine, batch):
        model.eval()

        x, y = batch
        y = F.one_hot(y, num_classes=10).float()

        x, y = x.cuda(), y.cuda()

        x.requires_grad_(True)

        y_pred = model(x)

        return y_pred, y, x, y_pred.sum(1)

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Accuracy(output_transform=output_transform_acc)
    metric.attach(evaluator, "accuracy")

    metric = Loss(F.binary_cross_entropy, output_transform=output_transform_bce)
    metric.attach(evaluator, "bce")

    metric = Loss(calc_gradient_penalty, output_transform=output_transform_gp)
    metric.attach(evaluator, "gradient_penalty")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20], gamma=0.2
    )

    dl_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True
    )

    dl_val = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=0
    )

    dl_test = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=0
    )

    pbar = ProgressBar()
    pbar.attach(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):

        scheduler.step()

        #if trainer.state.epoch % 5 == 0:
        evaluator.run(dl_val)

        metrics = evaluator.state.metrics

        val_accuracy = evaluator.state.metrics["accuracy"]

        print(
            f"Validation Results - Epoch: {trainer.state.epoch} "
            f"Acc: {metrics['accuracy']:.4f} "
            f"Val Acc: {val_accuracy:.4f} "
            f"BCE: {metrics['bce']:.2f} "
            f"GP: {metrics['gradient_penalty']:.6f} "
        )
        print(f"Sigma: {model.sigma}")

    trainer.run(dl_train, max_epochs=50)

    evaluator.run(dl_val)
    val_accuracy = evaluator.state.metrics["accuracy"]

    evaluator.run(dl_test)
    test_accuracy = evaluator.state.metrics["accuracy"]

    return model, val_accuracy, test_accuracy


if __name__ == "__main__":
    #_, _, _, fashionmnist_test_dataset = get_FashionMNIST()

    # Finding length scale - decided based on validation accuracy
    l_gradient_penalties = [0,0.1,0.3,0.5,0.7,0.9,1.0] #, 0.1, 0.3, 0.5, 1.0]
    length_scales = [0.1, 0.2, 0.5, 0.8, 1, 1.5] # [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    # Finding gradient penalty - decided based on AUROC on NotMNIST
    # l_gradient_penalties = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    # length_scales = [0.1]

    repetition = 1  # Increase for multiple repetitions
    final_model = False  # set true for final model to train on full train set

    results = {}

    for l_gradient_penalty in l_gradient_penalties:
        for length_scale in length_scales:

            print(" ### NEW MODEL ### ")
            model, val_accuracy, test_accuracy = train_model(
                l_gradient_penalty, length_scale, final_model
            )

            results[f"lgp{l_gradient_penalty}_ls{length_scale}"] = [
                val_accuracy,
                test_accuracy,
            ]
            print(results[f"lgp{l_gradient_penalty}_ls{length_scale}"])

    print(results)



