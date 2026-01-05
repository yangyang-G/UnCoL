import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def get_next_click3D_torch_certain(prev_seg, gt_semantic_seg, entropy_map):

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))  # False negatives
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)  # False positives

    for i in range(gt_semantic_seg.shape[0]):

        # Get the coordinates of foreground and background points
        fg_points = torch.argwhere(true_masks[i])  # Foreground points
        bg_points = torch.argwhere(torch.logical_not(true_masks[i]))  # Background points

        # Initialize lists for entropy values
        fg_entropy_values = []
        bg_entropy_values = []

        # Compute entropy values for foreground points (lowest entropy = most certain)
        if len(fg_points) > 0:
            fg_entropy_values = entropy_map[i, fg_points[:, 0], fg_points[:, 1], fg_points[:, 2]]  # Assuming 3D (x, y, z)
        
        # Compute entropy values for background points
        if len(bg_points) > 0:
            bg_entropy_values = entropy_map[i, bg_points[:, 0], bg_points[:, 1], bg_points[:, 2]]

        selected_points = []
        selected_labels = []

        # First, select the top-1 point from the foreground with the lowest entropy
        if len(fg_entropy_values) > 0:
            min_fg_entropy_idx = torch.argmin(fg_entropy_values)  # Index of the point with the lowest entropy
            selected_point = fg_points[min_fg_entropy_idx]

            selected_points.append(selected_point[1:].clone().detach().reshape(1,-1).to(pred_masks.device))
            selected_labels.append(1)  # Foreground points are positive

        # If no foreground points, select the top-1 point from the background
        elif len(bg_entropy_values) > 0:
            min_bg_entropy_idx = torch.argmin(bg_entropy_values)  # Index of the point with the lowest entropy
            selected_point = bg_points[min_bg_entropy_idx]
            selected_points.append(selected_point[1:].clone().detach().reshape(1,-1).to(pred_masks.device))
            selected_labels.append(0)  # Background points are negative

        # If no points are selected, fall back to random selection from the background
        if len(selected_points) == 0:
            point = torch.Tensor([np.random.randint(sz) for sz in fn_masks[i].size()]).to(torch.int64)
            selected_points.append(point[1:].clone().detach().reshape(1,-1).to(pred_masks.device))
            selected_labels.append(0)  # Negative point

        # Convert to the correct format for batch
        selected_points = torch.stack(selected_points, dim=0).to(prev_seg.device)
        selected_labels = torch.tensor(selected_labels).reshape(-1, 1).to(prev_seg.device)

        # Store the batch points and labels
        batch_points.append(selected_points)
        batch_labels.append(selected_labels)

    return batch_points, batch_labels

def get_next_click3D_torch(prev_seg, gt_semantic_seg):

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    for i in range(gt_semantic_seg.shape[0]):#, desc="generate points":

        fn_points = torch.argwhere(fn_masks[i])
        fp_points = torch.argwhere(fp_masks[i])
        point = None
        if len(fn_points) > 0 and len(fp_points) > 0:
            if np.random.random() > 0.5:
                point = fn_points[np.random.randint(len(fn_points))]
                is_positive = True
            else:
                point = fp_points[np.random.randint(len(fp_points))]
                is_positive = False
        elif len(fn_points) > 0:
            point = fn_points[np.random.randint(len(fn_points))]
            is_positive = True
        elif len(fp_points) > 0:
            point = fp_points[np.random.randint(len(fp_points))]
            is_positive = False
        # if no mask is given, random click a negative point
        if point is None: 
            point = torch.Tensor([np.random.randint(sz) for sz in fn_masks[i].size()]).to(torch.int64)
            is_positive = False
        bp = point[1:].clone().detach().reshape(1,1,-1).to(pred_masks.device) 
        bl = torch.tensor([int(is_positive),]).reshape(1,1).to(pred_masks.device) 

        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels # , (sum(dice_list)/len(dice_list)).item()    


import edt
def get_next_click3D_torch_ritm(prev_seg, gt_semantic_seg):
    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    fn_mask_single = F.pad(fn_masks, (1,1,1,1,1,1), 'constant', value=0).to(torch.uint8)[0,0]
    fp_mask_single = F.pad(fp_masks, (1,1,1,1,1,1), 'constant', value=0).to(torch.uint8)[0,0]
    fn_mask_dt = torch.tensor(edt.edt(fn_mask_single.cpu().numpy(), black_border=True, parallel=4))[1:-1, 1:-1, 1:-1]
    fp_mask_dt = torch.tensor(edt.edt(fp_mask_single.cpu().numpy(), black_border=True, parallel=4))[1:-1, 1:-1, 1:-1]
    fn_max_dist = torch.max(fn_mask_dt)
    fp_max_dist = torch.max(fp_mask_dt)
    is_positive = fn_max_dist > fp_max_dist # the biggest area is selected to be interaction point
    dt = fn_mask_dt if is_positive else fp_mask_dt
    to_point_mask = dt > (max(fn_max_dist, fp_max_dist) / 2.0) # use a erosion area
    to_point_mask = to_point_mask[None, None]
    # import pdb; pdb.set_trace()

    for i in range(gt_semantic_seg.shape[0]):
        points = torch.argwhere(to_point_mask[i])
        point = points[np.random.randint(len(points))]
        if fn_masks[i, 0, point[1], point[2], point[3]]:
            is_positive = True
        else:
            is_positive = False

        bp = point[1:].clone().detach().reshape(1,1,3) 
        bl = torch.tensor([int(is_positive),]).reshape(1,1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels # , (sum(dice_list)/len(dice_list)).item()    



def get_next_click3D_torch_2(prev_seg, gt_semantic_seg):

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    to_point_mask = torch.logical_or(fn_masks, fp_masks)
    
    for i in range(gt_semantic_seg.shape[0]):

        points = torch.argwhere(to_point_mask[i])
        point = points[np.random.randint(len(points))]
        # import pdb; pdb.set_trace()
        if fn_masks[i, 0, point[1], point[2], point[3]]:
            is_positive = True
        else:
            is_positive = False

        bp = point[1:].clone().detach().reshape(1,1,3) 
        bl = torch.tensor([int(is_positive),]).reshape(1,1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels # , (sum(dice_list)/len(dice_list)).item()    


def get_next_click3D_torch_with_dice(prev_seg, gt_semantic_seg):

    def compute_dice(mask_pred, mask_gt):
        mask_threshold = 0.5

        mask_pred = (mask_pred > mask_threshold)
        # mask_gt = mask_gt.astype(bool)
        mask_gt = (mask_gt > 0)
        
        volume_sum = mask_gt.sum() + mask_pred.sum()
        if volume_sum == 0:
            return np.NaN
        volume_intersect = (mask_gt & mask_pred).sum()
        return 2*volume_intersect / volume_sum

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    dice_list = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)


    for i in range(gt_semantic_seg.shape[0]):

        fn_points = torch.argwhere(fn_masks[i])
        fp_points = torch.argwhere(fp_masks[i])
        if len(fn_points) > 0 and len(fp_points) > 0:
            if np.random.random() > 0.5:
                point = fn_points[np.random.randint(len(fn_points))]
                is_positive = True
            else:
                point = fp_points[np.random.randint(len(fp_points))]
                is_positive = False
        elif len(fn_points) > 0:
            point = fn_points[np.random.randint(len(fn_points))]
            is_positive = True
        elif len(fp_points) > 0:
            point = fp_points[np.random.randint(len(fp_points))]
            is_positive = False
        # bp = torch.tensor(point[1:]).reshape(1,1,3) 
        bp = point[1:].clone().detach().reshape(1,1,3) 
        bl = torch.tensor([int(is_positive),]).reshape(1,1)
        batch_points.append(bp)
        batch_labels.append(bl)
        dice_list.append(compute_dice(pred_masks[i], true_masks[i]))

    return batch_points, batch_labels, (sum(dice_list)/len(dice_list)).item()    


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_point(point, label, ax):
    if label == 0:
        ax.add_patch(plt.Circle((point[1], point[0]), 1, color='red'))
    else:
        ax.add_patch(plt.Circle((point[1], point[0]), 1, color='green'))
    # plt.scatter(point[0], point[1], label=label)


def gt_lowest_unc_loc(mask, entropy_map):
    masked_entropy_map = entropy_map[mask]
    min_val, min_flat_idx = masked_entropy_map.min(dim=0)
    min_val = min_val.item() 
    mask_indices = mask.nonzero(as_tuple=True)

    return torch.Tensor([mask_indices[1][min_flat_idx].item() , 
            mask_indices[2][min_flat_idx].item() , 
            mask_indices[3][min_flat_idx].item() ]).long()


def get_next_click3D_torch_unc(prev_seg, gt_semantic_seg):

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    # fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    # fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    # to_point_mask = torch.logical_or(fn_masks, fp_masks)
    mask = (pred_masks != true_masks).cuda() # [1, 1, 128, 128, 128]

    ent_map = cal_ent(prev_seg) # [1, 1, 128, 128, 128]

    for i in range(gt_semantic_seg.shape[0]):
        point = gt_lowest_unc_loc(mask[i], ent_map[i])
        if true_masks[i, 0, point[0], point[1], point[2]]:
            is_positive = True
        else:
            is_positive = False        

        bp = point.clone().detach().reshape(1,1,3) 
        bl = torch.tensor([int(is_positive),]).reshape(1,1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels # , (sum(dice_list)/len(dice_list)).item()    

def get_next_click3D_torch_unc_unlab(prev_sam_seg, vnet_seg):

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []

    pred_masks = (prev_sam_seg > mask_threshold)
    true_masks = (vnet_seg > 0)
    
    mask_false = (pred_masks != true_masks).cuda() # [1, 1, 128, 128, 128]

    ent_sam = cal_ent(prev_sam_seg) # [1, 1, 128, 128, 128]
    ent_vnet = cal_ent(vnet_seg) # [1, 1, 128, 128, 128]
    ent_map = torch.softmax(torch.cat([ent_sam, ent_vnet], dim=1), dim=1)
    ent_map = (-1.0*torch.sum(ent_map*torch.log(ent_map+1e-16), dim=1)/np.log(2)).unsqueeze(1)
    mask_ent = (ent_sam > ent_vnet).cuda()

    mask = torch.logical_and(mask_false, mask_ent)
    for i in range(vnet_seg.shape[0]):
        point = gt_lowest_unc_loc(mask[i], ent_map[i])
        if true_masks[i, 0, point[0], point[1], point[2]]:
            is_positive = True
        else:
            is_positive = False        

        bp = point.clone().detach().reshape(1,1,3) 
        bl = torch.tensor([int(is_positive),]).reshape(1,1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels # , (sum(dice_list)/len(dice_list)).item()

def cal_ent(pred_fg):
    pred_fg = torch.sigmoid(pred_fg)
    pred_bg = 1-pred_fg # (B, 1, W, H, D)
    pred = torch.softmax(torch.cat([pred_bg, pred_fg], dim=1), dim=1) # (B, 2, W, H, D)
    pred_ent = -1.0*torch.sum(pred*torch.log(pred+1e-16), dim=1)/np.log(2) # (B, 1, W, H, D)
    return pred_ent.unsqueeze(1)

if __name__ == "__main__":
    gt2D = torch.randn((2,1,256, 256)).cuda()
    prev_masks = torch.zeros_like(gt2D).to(gt2D.device)
    batch_points, batch_labels = get_next_click3D_torch(prev_masks.to(gt2D.device), gt2D)
    print(batch_points)