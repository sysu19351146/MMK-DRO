from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import time
import copy
from PIL import Image
import cv2
import torchvision.transforms as transforms
import os
def visual_show(args,model,test_dataset):
    name_list=['origin','MMK-DRO','AWP','TRADES','MMA','FAST','FAT','SCORE','PGD']
    net_paths=[
        "/share_data/cap_udr_mnist/visformer_t/18-cap-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar",
              "/share_data/cap_udr_mnist/visformer_t/4-awp-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar",
              "/share_data/cap_udr_mnist/visformer_t/5-trades-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar",
              "/share_data/cap_udr_mnist/visformer_t/6-mma-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar",
              #"/share_data/cap_udr_mnist/visformer_t/8-AVmixup-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar",
              "/share_data/cap_udr_mnist/visformer_t/12-fast-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar",
              "/share_data/cap_udr_mnist/visformer_t/13-fat_trades-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar",
              "/share_data/cap_udr_mnist/visformer_t/12-trades_score-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar",
              "/share_data/cap_udr_mnist/visformer_t/16-pgd-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar"]

    # name_list = ['origin', 'cap', 'awp', 'trades', 'mma', 'AVmixup', 'socre', 'pgd']
    # net_paths = ["/share_data/cap_udr_mnist/visformer_t/28-cap-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar",
    #              "/share_data/cap_udr_mnist/visformer_t/19-awp-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar",
    #              "/share_data/cap_udr_mnist/visformer_t/21-trades-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar",
    #              "/share_data/cap_udr_mnist/visformer_t/23-mma-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar",
    #              "/share_data/cap_udr_mnist/visformer_t/22-AVmixup-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar",
    #              "/share_data/cap_udr_mnist/visformer_t/24-trades_score-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar",
    #              "/share_data/cap_udr_mnist/visformer_t/20-pgd-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar"]

    begin=time.time()
    proxy_model = copy.deepcopy(model)
    net_path= "/share_data/cap_udr_mnist/visformer_t/16-pgd-pgd-3.0-adam-0.001-cosineAnn/best_adv_model.pth.tar"
    ckpt = torch.load(
        net_path,
        map_location='cpu')

    model_dict = proxy_model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt["state_dict"].items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    proxy_model.load_state_dict(model_dict, strict=False)
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    args.device = torch.device(f"cuda:{gpu_list[0]}"
                               if torch.cuda.is_available() and args.cuda else "cpu")
    proxy_model = proxy_model.to(args.device)

    torch.cuda.set_device(args.device)
    proxy_model.eval()
    for i in range(len(net_paths)):
    #for i in range(1):
        net_path = net_paths[i]
        ckpt = torch.load(
            net_path,
            map_location='cpu')

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ckpt["state_dict"].items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict )
        model.load_state_dict(model_dict, strict=False)
        gpu_list = [int(i) for i in args.gpu.strip().split(",")]
        args.device = torch.device(f"cuda:{gpu_list[0]}"
                                   if torch.cuda.is_available() and args.cuda else "cpu")
        model = model.to(args.device)

        torch.cuda.set_device(args.device)
        model.eval()
        print(name_list[i+1])
        test_for_minist_PGD(args,model, test_dataset, i+2,name_list)
        #test_for_minist_black(args, model, test_dataset, i + 2, name_list,proxy_model)
        #test_for_minist(args, model, test_dataset, i + 2, name_list)
    print(time.time() - begin)

    plt.savefig('visual.png', dpi=1200)
    plt.show()



def test_for_minist_PGD(args,model, dataset, index,name_list):
    eps = 0.1
    alpha = 0.01
    iters = 10
    data_index = 144
    for data_ in dataset:
        inputs, label = data_
        inputs, label = inputs.float().cuda(), label.cuda()

        inputs_last = inputs[data_index, :].unsqueeze(0)
        while 1:

            inputs_temp = inputs[data_index, :].unsqueeze(0)
            inputs_temp, err, pred = pgd_whitebox_train(args,model, inputs_temp, label[data_index].unsqueeze(0), eps,step_size=eps/10)

            if err.item() == 1:
                print(eps, pred.item())
                break
            else:
                eps += 0.04
                inputs_last = inputs_temp
            if eps > 1:
                print(-1)
                break

        if index == 2:
            ax = plt.subplot(3, 3, 1)
            plt.imshow(((inputs[data_index, :].view(32, 32).cpu().detach().numpy() + 1) / 2.0 * 255.0).astype(np.int),
                       cmap=plt.cm.gray)
            ax.set_title(name_list[0])
            plt.axis('off')
        ax1 = plt.subplot(3, 3, index)
        plt.imshow(((inputs_last.view(32, 32).cpu().detach().numpy() + 1) / 2.0 * 255.0).astype(np.int),
                   cmap=plt.cm.gray)
        ax1.set_title(name_list[index - 1])

        plt.axis('off')
        break


def test_for_minist_black(args,model, dataset, index,name_list,proxy_model):
    eps = 0.1
    alpha = 0.01
    iters = 10
    data_index = 144
    for data_ in dataset:
        inputs, label = data_
        inputs, label = inputs.float().cuda(), label.cuda()

        inputs_last = inputs[data_index, :].unsqueeze(0)
        while 1:

            inputs_temp = inputs[data_index, :].unsqueeze(0)
            inputs_temp, err, pred = pgd_whitebox_train(args,proxy_model, inputs_temp, label[data_index].unsqueeze(0), eps,step_size=eps/5)
            pred = model(inputs_temp).data.max(1)[1]
            err_pgd = (pred != label[data_index].unsqueeze(0).data).float().sum()
            if err_pgd.item() == 1:
                print(eps, pred.item())
                break
            else:
                eps += 0.02
                inputs_last = inputs_temp
            if eps > 1:
                print(-1)
                break

        if index == 2:
            ax = plt.subplot(3, 4, 1)
            plt.imshow(((inputs[data_index, :].view(32, 32).cpu().detach().numpy() + 1) / 2.0 * 255.0).astype(np.int),
                       cmap=plt.cm.gray)
            ax.set_title(name_list[0])
            plt.axis('off')
        ax1 = plt.subplot(3, 4, index)
        plt.imshow(((inputs_last.view(32, 32).cpu().detach().numpy() + 1) / 2.0 * 255.0).astype(np.int),
                   cmap=plt.cm.gray)
        ax1.set_title(name_list[index - 1])

        plt.axis('off')
        break
pre_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                ])
def test_for_minist(args,model, dataset, index,name_list):
    eps = 0.5
    alpha = 0.01
    iters = 10
    data_index = 1
    # for data_ in dataset:
    #     inputs, label = data_
    #     inputs, label = inputs.float().cuda(), label.cuda()
    #
    #     inputs_last = inputs[data_index, :].unsqueeze(0)
    #     i=1
    #     for j in range(inputs.shape[0]):
    #         if label[j]==8 and i<12:
    #             print(j)
    #             ax1 = plt.subplot(3, 4, i)
    #             plt.imshow(((inputs[j].view(32, 32).cpu().detach().numpy() + 1) / 2.0 * 255.0).astype(np.int),
    #                        cmap=plt.cm.gray)
    #             ax1.set_title(j)
    #             i+=1
    #     break
    x=Image.open(os.path.join('outputs/image','Covid (411).png')).convert('RGB')
    aug_x=Image.open(os.path.join('outputs/image','Covid (411).png')).convert('RGB')
    t=0.003
    kernel=10
    x, aug_x = pre_transform(x), pre_transform(aug_x)
    post_transform = transforms.Compose(
        [
            transforms.ToTensor()
        ])
    x=post_transform(x)
    #pn_aug = PN_Aug(x, aug_x, t, kernel)
    #aug_x = pn_aug.get_aug()
    #plt.imshow(aug_x)
    #inputs, label = data_
    #inputs, label = inputs.float().cuda(), label.cuda()
    x=x.cuda()
    inputs_last = x.unsqueeze(0)


    inputs_temp = x.unsqueeze(0)
    inputs_temp, err, pred = pgd_whitebox_train(args, model, inputs_temp, torch.tensor([1]).cuda(), 0.03,
                                                step_size=0.00784)

    plt.subplot(1, 1, 1)
    plt.imshow((((inputs_temp-x).squeeze(0).transpose(0, 2).transpose(0, 1).cpu().detach().numpy())/0.1 * 255.0).astype(np.int))



# def pgd_whitebox(model,
#                   X,
#                   y,
#                   epsilon,
#                   num_steps=20,
#                   step_size=0.003):
#     out = model(X)
#     X_pgd = Variable(X.data, requires_grad=True)
#
#     random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
#     X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
#
#     for _ in range(num_steps):
#         opt = torch.optim.SGD([X_pgd], lr=1e-3)
#         opt.zero_grad()
#
#         with torch.enable_grad():
#             loss = nn.CrossEntropyLoss()(model(X_pgd), y)
#         loss.backward()
#         eta = step_size * X_pgd.grad.data.sign()
#         X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
#         eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
#         X_pgd = Variable(X.data + eta, requires_grad=True)
#         X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
#     err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
#     return err_pgd

def pgd_whitebox_train(args,model,
                  X,
                  y,
                  epsilon,
                  num_steps=10,
                  step_size=0.003):
    out = model(X)
    X_pgd = Variable(X.data, requires_grad=True)

    #random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    #X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
    #y_=torch.tensor([3]).cuda()
    for _ in range(num_steps):
        opt = torch.optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            #loss = nn.CrossEntropyLoss()(model(X_pgd), y_.long())
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    pred=model(X_pgd).data.max(1)[1]
    err_pgd = (pred != y.data).float().sum()
    return X_pgd,err_pgd,pred


class PN_Aug:
    def __init__(self, img1, img2, t=0.003, kernel=5):
        self.img2 = img2  # contrast sample
        self.img1 = img1  # original sample
        self.t = t
        self.kernel = kernel

    @staticmethod
    def guideFilter(I, p, kernel, t, s):
        I = np.asarray(I) / 255.0
        p = np.asarray(p) / 255.0
        winSize = [kernel, kernel]
        h, w = I.shape[:2]

        size = (int(round(w * s)), int(round(h * s)))

        small_I = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
        small_p = cv2.resize(p, size, interpolation=cv2.INTER_CUBIC)

        X = winSize[0]
        small_winSize = (int(round(X * s)), int(round(X * s)))

        mean_small_I = cv2.blur(small_I, small_winSize)

        mean_small_p = cv2.blur(small_p, small_winSize)

        mean_small_II = cv2.blur(small_I * small_I, small_winSize)

        mean_small_Ip = cv2.blur(small_I * small_p, small_winSize)

        var_small_I = mean_small_II - mean_small_I * mean_small_I
        cov_small_Ip = mean_small_Ip - mean_small_I * mean_small_p

        small_a = cov_small_Ip / (var_small_I + t)
        small_b = mean_small_p - small_a * mean_small_I

        mean_small_a = cv2.blur(small_a, small_winSize)
        mean_small_b = cv2.blur(small_b, small_winSize)

        size1 = (w, h)
        mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR)

        q = mean_a * I + mean_b
        gf = q * 255
        gf[gf > 255] = 255
        gf = np.round(gf)
        gf = gf.astype(np.uint8)
        return gf

    def masked(self, img1, img2, ):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        mask = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)[1]
        mask_fg = cv2.bitwise_not(mask)
        mask_bg = mask

        img1_bg = cv2.bitwise_and(img1, img1, mask=mask_bg)
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask_fg)

        img2_fg = self.guideFilter(img1, img2_fg, self.kernel, t=self.t, s=0.5)
        dst = cv2.add(img1_bg, img2_fg)

        final_img = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
        return final_img

    def get_aug(self):
        image_dst = self.masked(np.asarray(self.img1.copy()), np.asarray(self.img1.copy()))
        return image_dst