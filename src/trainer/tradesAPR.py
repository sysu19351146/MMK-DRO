import time
import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random
import torchvision
AverageMeter = importlib.import_module(f"src.utils.metrics").AverageMeter
ProgressMeter = importlib.import_module(f"src.utils.metrics").ProgressMeter
accuracy = importlib.import_module(f"src.utils.metrics").accuracy


def mix_data(x, use_cuda=True, prob=0.6):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    p = random.uniform(0, 1)

    if p > prob:
        return x

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    fft_1 = torch.fft.fftn(x, dim=(1,2,3))
    abs_1, angle_1 = torch.abs(fft_1), torch.angle(fft_1)

    fft_2 = torch.fft.fftn(x[index, :], dim=(1,2,3))
    abs_2, angle_2 = torch.abs(fft_2), torch.angle(fft_2)

    fft_1 = abs_2*torch.exp((1j) * angle_1)

    mixed_x = torch.fft.ifftn(fft_1, dim=(1,2,3)).float()

    return mixed_x

normalize = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

def trades_loss(
    model,
    x_natural,
    y,
    optimizer,
    step_size,
    epsilon,
    perturb_steps,
    beta,
    clip_min,
    clip_max,
    device,
    distance="l_inf",
    natural_criterion=nn.CrossEntropyLoss(),
    cutmix_prob=0.5
):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = (
        x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    )
    if distance == "l_inf":
        for _ in (range(perturb_steps)):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(
                    F.log_softmax(model(x_adv), dim=1),
                    F.softmax(model(x_natural), dim=1),
                )

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
            )
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif distance == "l_2":
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(
                    F.log_softmax(model(adv), dim=1), F.softmax(model(x_natural), dim=1)
                )
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(
                    delta.grad[grad_norms == 0]
                )
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(clip_min, clip_max).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    optimizer.zero_grad()

    inputs_mix = mix_data(x_natural)
    inputs_mix = Variable(inputs_mix)
    batch_size = x_natural.size(0)

    inputs = torch.cat([x_natural, inputs_mix], 0)
    output = model(inputs)
    loss_natural = natural_criterion(output[:batch_size], y)+natural_criterion(output[batch_size:], y)


    loss_robust = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(model(x_adv), dim=1), F.softmax(model(x_natural), dim=1)
    )
    loss = (loss_natural)  + beta * loss_robust
    return loss, loss_natural, loss_robust*beta


def train(model, dataloader, optimizer, args, epoch, device, logger, AttackPolicy):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    nat_losses = AverageMeter("Natural Loss", ":.4f")
    rob_losses = AverageMeter("Robust Loss", ":.4f")
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
        loss, nat_loss, rob_loss = trades_loss(
            model=model,
            x_natural=images,
            y=labels,
            optimizer=optimizer,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
            beta=args.beta,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            device=device,
            distance=args.distance,
        )

        acc1, _ = accuracy(outputs, labels, topk=(1, ))
        losses.update(loss.item(), images.size(0))
        nat_losses.update(nat_loss.item(), images.size(0))
        rob_losses.update(rob_loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        batch_time.update(time.time() - end)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return model, losses.avg, nat_losses.avg, rob_losses.avg, top1.avg
