import time
import importlib
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
AverageMeter = importlib.import_module(f"src.utils.metrics").AverageMeter
ProgressMeter = importlib.import_module(f"src.utils.metrics").ProgressMeter
accuracy = importlib.import_module(f"src.utils.metrics").accuracy


def pgd_loss(
    model,
    x_natural,
    y,
    optimizer,
    step_size,
    epsilon,
    perturb_steps,
    clip_min,
    clip_max,
    device,
    rand_start_mode='gaussian',
    criterion=nn.CrossEntropyLoss(),
):




    model.train()
    logits_ = model(x_natural)
    loss = criterion(logits_, y)
    return loss


def train(model, dataloader, optimizer, args, epoch, device, logger, AttackPolicy):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, losses, top1],
        logger,
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    model.train()
    for i, data in enumerate(dataloader):
        images = data[0].to(device)
        labels = data[1].to(device)
        outputs = model(images)
        # calculate robust loss
        loss = pgd_loss(
            model=model,
            x_natural=images,
            y=labels,
            optimizer=optimizer,
            step_size=args.step_size,
            epsilon= args.epsilon,
            perturb_steps=args.num_steps,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            device=device,
        )

        # measure accuracy and record loss
        acc1, _ = accuracy(outputs, labels, topk=(1, ))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        # _, preds = torch.max(outputs.data, 1)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    progress.display(i)

    return model, losses.avg, 0, 0, top1.avg
