import time
import importlib
import torch
import torch.nn as nn
from torch.autograd import Variable
from src.utils.weight_noise import udr_pgd_1
from src.utils.utils import mynorm
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
    lamda,
    tau,
    rand_start_mode='gaussian',
    criterion=nn.CrossEntropyLoss(),

):

    model.eval()

    # if rand_start_mode == 'gaussian':
    #     x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).detach().to(device)
    # elif rand_start_mode == 'uniformv1':
    #     x_adv = x_natural.detach() + epsilon * torch.rand(x_natural.shape).detach().to(device)
    # elif rand_start_mode == 'uniformv2':
    #     x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).to(device)
    # else:
    #     raise NameError
    #
    # for _ in range(perturb_steps):
    #     x_adv.requires_grad_()
    #
    #     with torch.enable_grad():
    #         loss = criterion(model(x_adv), y)
    #     grad = torch.autograd.grad(loss, x_adv)[0]
    #     x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
    #     x_adv = torch.min(
    #         torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
    #     )
    #     x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)
    # model.train()
    x_adv, delta = udr_pgd_1(x_natural,x_natural, y, model, perturb_steps, 0.01, tau, lamda=lamda, device=device, epsilon=epsilon)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    x_adv = Variable(torch.clamp(x_adv, min=clip_min, max=clip_max), requires_grad=False)
    logits_ = model(x_adv)
    loss = criterion(logits_, y)
    return loss,delta


def train(model, dataloader, optimizer, args, epoch, device, logger, AttackPolicy,lamda,tau):
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
    delta_list=[]
    for i, data in enumerate(dataloader):
        images = data[0][0].to(device)
        labels = data[1].to(device)
        outputs = model(images)
        # calculate robust loss
        loss,delta = pgd_loss(
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
            lamda=lamda,
            tau=tau
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

    return model, losses.avg, 0, 0, top1.avg,delta
