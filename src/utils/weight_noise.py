import torch
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
EPS = 1E-20


def diff_in_attn(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if ('weight' in old_k) and ('attn' in old_k):
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict



def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class TradesAWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(TradesAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_aug, inputs_clean, targets, beta):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        loss_natural = F.cross_entropy(self.proxy(inputs_clean), targets)
        loss_robust = F.kl_div(F.log_softmax(self.proxy(inputs_aug), dim=1),
                               F.softmax(self.proxy(inputs_clean), dim=1),
                               reduction='batchmean')
        loss = - 1.0 * ((loss_natural ) + beta * loss_robust)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb

        diff = diff_in_weights(self.model, self.proxy)

        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


class CAP(object):
    def __init__(self, model, proxy, proxy_optim, gamma, num_classes):
        super(CAP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma
        self.num_classes = num_classes

    def calc_cap(self, inputs_aug, inputs_clean, targets, beta):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()



        loss_natural = F.cross_entropy(self.proxy(inputs_clean), targets)
        loss_robust = F.kl_div(F.log_softmax(self.proxy(inputs_aug), dim=1),
                               F.softmax(self.proxy(inputs_clean), dim=1),
                               reduction='batchmean')
        loss =  1.0 * ((loss_natural ) + beta * loss_robust)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_attn(self.model, self.proxy)

        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)

upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)
def udr_pgd(X, y,  model, attack_iters, alpha, tau, restarts=1, norm="l_inf", epsilon=0.03,
             early_stop=False, mixup=False, lamda=None, device=None):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            # output = model(normalize(X + delta))
            output = model(X + delta)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                # loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:

                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]

            # Gradient ascent step (ref to Algorithm 1 - step 2bi in our paper)
            d = d + alpha * torch.sign(g)  # equal x_adv = x_adv + alpha * torch.sign(g)

            # Projection step (ref to Algorithm 1 - step 2bii in our paper)
            # tau = alpha # Simply choose tau = alpha

            abs_d = torch.abs(d)
            abs_d = abs_d.detach()

            d = d - lamda.detach() * alpha / tau * (d - torch.sign(d) * epsilon) * (abs_d > epsilon)
            # d = d - lamda.detach() * alpha / tau * (d - torch.sign(d) * epsilon) * (abs_d > epsilon)-lamda.detach() * alpha  * (d - torch.sign(d) * epsilon) * (abs_d <= epsilon)

            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            # all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(X + delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta + X, max_delta

criterion_kl = nn.KLDivLoss(size_average=False)
def udr_pgd_1(X,x_aug, y,  model, attack_iters, alpha, tau, restarts=1, norm="l_inf", epsilon=0.03,
             early_stop=False, mixup=False, lamda=None, device=None,args=None):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            # output = model(normalize(X + delta))
            output = model(X + delta)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                # loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                # y_onehot = F.one_hot(y, args.num_classes).float()
                # logits_aug = F.softmax(model(x_aug), dim=1)
                #
                # y_smooth = epsilon * logits_aug + (1 - epsilon) * y_onehot
                #
                # logits_natural = F.softmax(model(X), dim=1)
                # logits_adv = F.softmax(model(X + delta), dim=1)
                #
                # loss_natural = torch.sum((logits_natural - y_smooth) ** 2, dim=-1)
                # loss_robust = torch.sum((logits_adv - logits_natural) ** 2, dim=-1)
                # loss_robust = F.relu(loss_robust)  # clip loss value
                outputs = model(X)
                loss_nat = F.cross_entropy(outputs, y)
                loss_robust = (1.0 /X.shape[0]) * criterion_kl(F.log_softmax(model(X + delta), dim=1),
                                                                F.softmax(model(X), dim=1))
                loss = loss_nat + 7.0 * loss_robust

                # loss_main.backward()
                #
                # loss = loss_natural.mean() + args.beta * loss_robust.mean()
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]

            # Gradient ascent step (ref to Algorithm 1 - step 2bi in our paper)
            d = d + alpha * torch.sign(g)  # equal x_adv = x_adv + alpha * torch.sign(g)

            # Projection step (ref to Algorithm 1 - step 2bii in our paper)
            # tau = alpha # Simply choose tau = alpha

            abs_d = torch.abs(d)
            abs_d = abs_d.detach()

            d = d - lamda.detach() * alpha / tau * (d - torch.sign(d) * epsilon) * (abs_d > epsilon)
            # d = d - lamda.detach() * alpha / tau * (d - torch.sign(d) * epsilon) * (abs_d > epsilon)-lamda.detach() * alpha  * (d - torch.sign(d) * epsilon) * (abs_d <= epsilon)

            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            # all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(X + delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta + X, max_delta