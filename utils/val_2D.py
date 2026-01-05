import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred,gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred,gt)
        return np.array([dice, jc, hd95, asd])
    else:
        return np.zeros(4)


def test_single_volume_pred(image, label, net, classes, img_embed=None, patch_size=[256, 256], AMC=False):
    # image, label = image.squeeze(0).cpu().detach(
    # ).numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    total_metric = np.zeros((classes-1, 4))

    x, y = image.shape[0], image.shape[1]
    # image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)
    # input = torch.from_numpy(image).unsqueeze(
    #     0).unsqueeze(0).float().cuda()
    input = image.unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        if AMC:
            output = net(input)
            y = 0.25*(output[0]+output[1]+output[2]+output[3])
            output = torch.softmax(y, dim=1)
            out = torch.argmax(torch.softmax(
                y, dim=1), dim=1).squeeze(0)
        else:
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            output = torch.softmax(
                net(input), dim=1)
        prediction = out.cpu().detach().numpy()
        # prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            
    # metric_list = []
    for i in range(1, classes):
        # metric_list.append(calculate_metric_percase(
        #     prediction == i, label == i))
        total_metric[i-1, :] = cal_metric(label == i, prediction == i)
        
    # return metric_list
    return total_metric, prediction, output

def test_single_volume(image, label, net, classes, patch_size=[256, 256], AMC=False):
    # image, label = image.squeeze(0).cpu().detach(
    # ).numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    total_metric = np.zeros((classes-1, 4))

    x, y = image.shape[0], image.shape[1]
    # image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)
    # input = torch.from_numpy(image).unsqueeze(
    #     0).unsqueeze(0).float().cuda()
    input = image.unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        if AMC:
            output = net(input)
            y = 0.25*(output[0]+output[1]+output[2]+output[3])
            out = torch.argmax(torch.softmax(
                y, dim=1), dim=1).squeeze(0)
        else:
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
        # prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            
    # metric_list = []
    for i in range(1, classes):
        # metric_list.append(calculate_metric_percase(
        #     prediction == i, label == i))
        total_metric[i-1, :] = cal_metric(label == i, prediction == i)
        
    # return metric_list
    return total_metric 

def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
