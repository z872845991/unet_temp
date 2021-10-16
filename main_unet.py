import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from dataset.Fetus import FetusDataset
# from model.seunet import Unet
from model.unet import Unet
# from archs import NestedUNet
# from ince_unet import Unet
# from eca_unet import Unet
# from model.archs import NestedUNet
# from model.Eca_att_unet import Att_Unet
# from model.attention_u_net import Att_Unet
# from model.dp_unet import Unet
# from model.ternausnet import UNet11,UNet16
# from model.r2unet import R2U_Net

# from model.res_unet import ResNet34Unet
# from model.aug_att_uent import AugAtt_Unet
# from model.self_att_unet import Att_Unet
# from model.channel_unet import myChannelUnet
# from model.cenet import CE_Net_
# from model.nolocal.unet_nonlocal_2D import unet_nonlocal_2D
from tools.metrics import dice_coef,iou_score, get_accuracy, get_precision, get_specificity, get_recall, get_F1
from tools.utils import AverageMeter
import datetime

# import visdom
"""原始代码,train的同时训练 """

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

x_transforms = transforms.Compose([
    transforms.Resize((512,512)),
    # transforms.CenterCrop(512),
    transforms.ToTensor()
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    # transforms.CenterCrop(512),
    transforms.ToTensor()
   ])

#参数解析


#训练模型
def train(args):

    model = Unet(3,1)
    model = model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    fetus_dataset = FetusDataset("/home/p920/cf/data/train/", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(fetus_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=False)
    train_model(model, criterion, optimizer, dataloaders,args.epoches)


def train_model(model, criterion, optimizer, dataload, num_epochs):
    # 这个是用来找到miou最好的一次epoch
    bigiou=0

    model.train()
    for epoch in range(num_epochs):
        # print(epoch)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        avgmeter1 = AverageMeter()
        avgmeter2 = AverageMeter()
        avgmeter3 = AverageMeter()
        avgmeter4 = AverageMeter()
        avgmeter5 = AverageMeter()
        avgmeter6 = AverageMeter()
        avgmeter7 = AverageMeter()

        step = 0

        for x, y, _ in dataload:
            step += 1
            inputs = x.cuda()
            labels = y.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            iou = iou_score(outputs, labels)
            dice = dice_coef(outputs, labels)
            ACC = get_accuracy(outputs, labels)
            PPV = get_precision(outputs, labels)
            TNR = get_specificity(outputs, labels)
            TPR = get_recall(outputs, labels)
            F1 = get_F1(outputs, labels)

            avgmeter1.update(iou, args.batch_size)
            avgmeter2.update(dice, args.batch_size)
            avgmeter3.update(ACC, args.batch_size)
            avgmeter4.update(PPV, args.batch_size)
            avgmeter5.update(TNR, args.batch_size)
            avgmeter6.update(TPR, args.batch_size)
            avgmeter7.update(F1, args.batch_size)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()


        with open('../../result/train/train_7m_unet.txt', 'a+') as file:
            file.write("epoch %d loss:%0.3f miou:%.3f maxiou:%.3f miniou:%.3f  mdice:%.3f maxdice:%.3f mindice:%.3f" % (
            epoch, epoch_loss / step, avgmeter1.avg, avgmeter1.max, avgmeter1.min, avgmeter2.avg, avgmeter2.max,
            avgmeter2.min) + '\n')

        test_model(epoch, model, bigiou)

        # torch.save(model.state_dict(), '/home/p920/cf/checkpoints/weights_r2unet_%d.pth' % epoch)

def test_model(epoch, model, bigiou):
    threshold = 0.5

    model.eval()
    test_dataset = FetusDataset("/home/p920/cf/data/test/", mode='train',transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(test_dataset, batch_size=1)
    step = 0

    te_avgmeter1 = AverageMeter()
    te_avgmeter2 = AverageMeter()
    te_avgmeter3 = AverageMeter()
    te_avgmeter4 = AverageMeter()
    te_avgmeter5 = AverageMeter()
    te_avgmeter6 = AverageMeter()
    te_avgmeter7 = AverageMeter()


    for x, y, z in dataloaders:
        step += 1
        inputs = x.cuda()
        labels = y.cuda()
        outputs = model(inputs)


        iou1 = iou_score(outputs, labels)
        dice1 = dice_coef(outputs, labels)
        ACC1 = get_accuracy(outputs, labels)
        PPV1 = get_precision(outputs, labels)
        TNR1 = get_specificity(outputs, labels)
        TPR1 = get_recall(outputs, labels)
        F11 = get_F1(outputs, labels)

        te_avgmeter1.update(iou1)
        te_avgmeter2.update(dice1)
        te_avgmeter3.update(ACC1)
        te_avgmeter4.update(PPV1)
        te_avgmeter5.update(TNR1)
        te_avgmeter6.update(TPR1)
        te_avgmeter7.update(F11)


        if epoch == 80 and iou1 < threshold:
            with open('../../result/hard/hard_seg_7m_unet.txt', 'a+') as file:
                for s in z:
                    file.write(s + '\n')


    with open('../../result/test/test_7m_unet.txt', 'a+') as file:
        file.write(" ACC:%.4f  PPV:%.4f  TNR:%.4f  TPR:%.4f  F1:%.4f  miou:%.4f maxiou:%.4f miniou:%.4f  mdice:%.4f maxdice:%.4f mindice:%.4f iou1:%.4f iou2:%.4f iou3:%.4f iou4:%.4f iou5:%.4f iou6:%.4f iou7:%.4f iou8:%.4f dice1:%.4f dice2:%.4f dice3:%.4f dice4:%.4f dice5:%.4f dice6:%.4f dice7:%.4f dice8:%.4f" % (
            te_avgmeter3.avg, te_avgmeter4.avg, te_avgmeter5.avg, te_avgmeter6.avg, te_avgmeter7.avg, te_avgmeter1.avg, te_avgmeter1.max, te_avgmeter1.min, te_avgmeter2.avg, te_avgmeter2.max,te_avgmeter2.min, te_avgmeter1.first, te_avgmeter1.second, te_avgmeter1.third, te_avgmeter1.forth, te_avgmeter1.fifth, te_avgmeter1.sixth, te_avgmeter1.seventh, te_avgmeter1.eighth, te_avgmeter2.first, te_avgmeter2.second, te_avgmeter2.third, te_avgmeter2.forth, te_avgmeter2.fifth, te_avgmeter2.sixth, te_avgmeter2.seventh, te_avgmeter2.eighth) + '\n')


    if  te_avgmeter1.avg > bigiou:
        bigiou = te_avgmeter1.avg
        torch.save(model.state_dict(), '/home/p920/cf/checkpoints/weights_7m_unet_%d.pth' % epoch)



if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--epoches", type=int, default=81)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    # viz = visdom.Visdom(env='cer_seg')
    start = datetime.datetime.now()
    train(args)
    # test()
    end = datetime.datetime.now()
    print('unet')
    print(end-start)
    # with open('./result/test_unet.txt', 'a+') as file:
    #     file.write(end-start + '\n')
