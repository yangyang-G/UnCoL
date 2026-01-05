import torch
import torch.nn.functional as F
import torchio as tio
import os


from datasets.brats19 import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from datasets.AortaDissection import AortaDissection
from datasets.pancreas import Pancreas

from SAM_Med3D.utils.click_method import (get_next_click3D_torch_ritm, get_next_click3D_torch,
                                            get_next_click3D_torch_2, get_next_click3D_torch_certain)
from utils import losses, ramps

click_methods = {
    'default': get_next_click3D_torch_ritm,
    'ritm': get_next_click3D_torch_ritm,
    'random': get_next_click3D_torch,
    'uncertain': get_next_click3D_torch_certain,
}


def compute_sigmoid_entropy(inputs, sigmoid=False):
    # Apply Sigmoid to each channel
    if sigmoid:
        sigmoid_probs = torch.sigmoid(inputs)  # [B, C, H, W, D]
    else:
        sigmoid_probs = inputs

    # Compute entropy: H = -p * log(p) - (1-p) * log(1-p)
    entropy = -sigmoid_probs * torch.log(sigmoid_probs + 1e-8) - (1 - sigmoid_probs) * torch.log(1 - sigmoid_probs + 1e-8) # [B, C, H, W, D]

    # Average entropy on each class
    return torch.mean(entropy, dim=1) # [B, H, W, D]  

def compute_softmax_entropy(inputs, softmax=False):
    # Apply Softmax across channels (dim=1 for C dimension)
    if softmax:
        softmax_probs = F.softmax(inputs, dim=1)  # [B, C, H, W, D]
    else:
        softmax_probs = inputs

    # Compute entropy: H = -sum(p * log(p)) along the channel dimension
    entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-8), dim=1)  # [B, H, W, D]

    return entropy  # [B, H, W, D]

def compute_batch_rank_maps(entropy_maps):
    """
    Compute rank maps for a batch of entropy maps.

    Args:
        entropy_maps (torch.Tensor): Batch of entropy maps of shape [B, H, W, D].
    
    Returns:
        torch.Tensor: Batch of rank maps with the same shape [B, H, W, D].
    """
    B = entropy_maps.shape[0]  # Batch size, Height, Width, Depth

    # Flatten each entropy map for batch-wise operations
    flattened_maps = entropy_maps.view(B, -1)  # Shape: [B, H * W * D]

    # Perform sorting for each map in the batch
    sorted_indices = torch.argsort(flattened_maps, dim=1, descending=False)  # Shape: [B, H * W * D]

    # Initialize rank maps
    rank_maps = torch.empty_like(flattened_maps, dtype=torch.long, device=entropy_maps.device)

    # Generate 1-based ranks for each map in the batch
    for i in range(B):
        rank_maps[i, sorted_indices[i]] = torch.arange(1, flattened_maps.size(1) + 1, dtype=torch.long, device=entropy_maps.device)

    # Reshape back to the original shape
    rank_maps = rank_maps.view_as(entropy_maps)  # Shape: [B, H, W, D]
    return rank_maps

def generate_mask(rank_gtm, rank_stm, pred_gtm, pred_stm, tau):
    """
    Generate the mask M^S based on rank difference and prediction disagreement.

    Args:
        rank_gtm (torch.Tensor): Rank map from GTM, shape (N, H, W, D).
        rank_stm (torch.Tensor): Rank map from STM, shape (N, H, W, D).
        pred_gtm (torch.Tensor): Predicted labels from GTM, shape (N, H, W, D).
        pred_stm (torch.Tensor): Predicted labels from STM, shape (N, H, W, D).
        tau (float): Threshold for rank difference.

    Returns:
        torch.Tensor: Mask M^S, shape (N, 1, H, W, D).
    """
    N, H, W, D = rank_gtm.size()
    tau = float(round(tau*H*W*D))
    # Compute rank difference and check if it is less than tau
    rank_diff_condition_stm = (rank_gtm - rank_stm) > tau
    rank_diff_condition_gtm = (rank_stm - rank_gtm) > tau

    # Check prediction disagreement
    pred_disagreement_condition = (pred_gtm != pred_stm)

    # Combine conditions (logical AND)
    mask_s = rank_diff_condition_stm & pred_disagreement_condition
    mask_g = rank_diff_condition_gtm & pred_disagreement_condition

    # Convert to float tensor for further processing
    return mask_s.unsqueeze(1).float(), mask_g.unsqueeze(1).float()

def compute_consistency_loss(
    phi_psm,
    pred_stm,
    pred_gtm,
    mask_s,
    mask_g,
    delta=1.0
):
    """
    Compute the consistency loss with Cross-Entropy (CE) loss and masks.

    Args:
        phi_psm (torch.Tensor): Output from PSM model, shape (N, C, H, W, D).
        pred_gtm (torch.Tensor): Predicted labels from GTM, shape (N, H, W, D).
        pred_stm (torch.Tensor): Predicted labels from STM, shape (N, H, W, D).
        mask_s (torch.Tensor): Mask M^S, shape (N, 1, H, W, D).
        mask_g (torch.Tensor): Mask M^G, shape (N, 1, H, W, D).
        delta (float): Weighting factor for the second term.

    Returns:
        torch.Tensor: The computed consistency loss.
    """
    # Compute Cross-Entropy losses
    ce_psm_stm = F.cross_entropy(phi_psm, pred_stm, reduction='none') # torch.Size([N, 128, 128, 128]) 
    ce_psm_gtm = F.cross_entropy(phi_psm, pred_gtm, reduction='none') # torch.Size([N, 128, 128, 128]) 
     

    # Apply masks
    ce_psm_stm_masked = ce_psm_stm * mask_s.squeeze(1) # torch.Size([N, 128, 128, 128]) 
    ce_psm_gtm_masked = ce_psm_gtm * mask_g.squeeze(1) # torch.Size([N, 128, 128, 128])
    pred_agreement_condition = (pred_gtm == pred_stm).float()
    ce_psm_agreement_masked = ce_psm_stm * pred_agreement_condition

    # Normalize by the number of positive positions in each mask
    mask_s_positive = mask_s.sum(dim=[1, 2, 3, 4], keepdim=True)
    mask_g_positive = mask_g.sum(dim=[1, 2, 3, 4], keepdim=True)
    mask_a_positive = pred_agreement_condition.sum(dim=[1,2,3])

    # Avoid division by zero
    mask_s_positive = torch.clamp(mask_s_positive, min=1.0)
    mask_g_positive = torch.clamp(mask_g_positive, min=1.0)
    mask_a_positive = torch.clamp(mask_a_positive, min=1.0)

    loss_s = (ce_psm_stm_masked.sum(dim=[1, 2, 3]) / mask_s_positive.squeeze(1).squeeze(1).squeeze(1).squeeze(1)).mean()
    loss_g = (ce_psm_gtm_masked.sum(dim=[1, 2, 3]) / mask_g_positive.squeeze(1).squeeze(1).squeeze(1).squeeze(1)).mean()
    loss_a = (ce_psm_agreement_masked.sum(dim=[1, 2, 3]) / mask_a_positive).mean()

    assert not torch.isnan(loss_s).any(), "Tensor contains NaN values"
    assert not torch.isnan(loss_g).any(), "Tensor contains NaN values"
    assert not torch.isnan(loss_a).any(), "Tensor contains NaN values"


    # # Combine losses with weighting factor delta
    # consistency_loss = loss_s + delta * loss_g

    return loss_s, loss_g, loss_a

def compute_consistency_loss_mse(
    phi_psm,
    pred_stm,
    pred_gtm,
    mask_s,
    mask_g,
    delta=1.0,
    phi_stm=None,
):
    """
    Compute the consistency loss with Cross-Entropy (CE) loss and masks.

    Args:
        phi_psm (torch.Tensor): Output from PSM model, shape (N, C, H, W, D).
        pred_gtm (torch.Tensor): Predicted labels from GTM, shape (N, H, W, D).
        pred_stm (torch.Tensor): Predicted labels from STM, shape (N, H, W, D).
        mask_s (torch.Tensor): Mask M^S, shape (N, 1, H, W, D).
        mask_g (torch.Tensor): Mask M^G, shape (N, 1, H, W, D).
        delta (float): Weighting factor for the second term.

    Returns:
        torch.Tensor: The computed consistency loss.
    """
    # Compute Cross-Entropy losses
    # ce_psm_stm = F.cross_entropy(phi_psm, pred_stm, reduction='none') # torch.Size([N, 128, 128, 128]) 
    ce_psm_gtm = F.cross_entropy(phi_psm, pred_gtm, reduction='none') # torch.Size([N, 128, 128, 128]) 
    mse_psm_stm = losses.softmax_mse_loss(phi_psm, phi_stm)

    # Apply masks
    mse_psm_stm_masked = mse_psm_stm * mask_s.squeeze(1) # torch.Size([N, 128, 128, 128]) 
    ce_psm_gtm_masked = ce_psm_gtm * mask_g.squeeze(1) # torch.Size([N, 128, 128, 128])
    pred_agreement_condition = (pred_gtm == pred_stm).float()
    mse_psm_agreement_masked = mse_psm_stm * pred_agreement_condition

    # Normalize by the number of positive positions in each mask
    mask_s_positive = mask_s.sum(dim=[1, 2, 3, 4], keepdim=True)
    mask_g_positive = mask_g.sum(dim=[1, 2, 3, 4], keepdim=True)
    mask_a_positive = pred_agreement_condition.sum(dim=[1,2,3])

    # Avoid division by zero
    mask_s_positive = torch.clamp(mask_s_positive, min=1.0)
    mask_g_positive = torch.clamp(mask_g_positive, min=1.0)
    mask_a_positive = torch.clamp(mask_a_positive, min=1.0)

    loss_s = (mse_psm_stm_masked.sum(dim=[1, 2, 3]) / mask_s_positive.squeeze(1).squeeze(1).squeeze(1).squeeze(1)).mean()
    loss_g = (ce_psm_gtm_masked.sum(dim=[1, 2, 3]) / mask_g_positive.squeeze(1).squeeze(1).squeeze(1).squeeze(1)).mean()
    loss_a = (mse_psm_agreement_masked.sum(dim=[1, 2, 3]) / mask_a_positive).mean()

    assert not torch.isnan(loss_s).any(), "Tensor contains NaN values"
    assert not torch.isnan(loss_g).any(), "Tensor contains NaN values"
    assert not torch.isnan(loss_a).any(), "Tensor contains NaN values"


    # # Combine losses with weighting factor delta
    # consistency_loss = loss_s + delta * loss_g

    return loss_s, loss_g, loss_a

def get_sam_img_embed(img3D, sam_model, device='cuda'):
        
    with torch.no_grad():
        image_embedding = sam_model.image_encoder(img3D.to(device)) # (1, 384, 16, 16, 16)
    return image_embedding

def get_sam_hidden(img3D, sam_model, device='cuda'):
        
    with torch.no_grad():
        _, hidden_states_out = sam_model.image_encoder(img3D.to(device), hidden_out=True) # (1, 384, 16, 16, 16)
    return hidden_states_out
        


def get_sam_pred(img3D, label_batch, sam_model, args, device='cuda', 
                                click_method='random', click_method_aft='random', num_clicks=10, prev_masks=None):
    if 'OASIS' not in args.data_dir:
        img3D = norm_transform(img3D.squeeze(dim=1)) # (N, C, W, H, D)
        img3D = img3D.unsqueeze(dim=1)
        
    with torch.no_grad():
        image_embedding = sam_model.image_encoder(img3D.to(device)) # (1, 384, 16, 16, 16)

    posterior_features = []
    fg_predictions = []
    bg_predictions = torch.zeros_like(label_batch.unsqueeze(1)).to(device)+0.5
    
    for c in range(1, args.num_classes):
        gt3D = (label_batch==c).to(torch.long).unsqueeze(1).to(device)
        if prev_masks is None:
            prev_masks = torch.zeros_like(gt3D).to(device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(args.crop_size//4,args.crop_size//4,args.crop_size//4))

        for num_click in range(num_clicks):
            with torch.no_grad():
                batch_points, batch_labels = click_methods[click_method](prev_masks.to(device), gt3D.to(device))

                points_co = torch.cat(batch_points, dim=0).to(device)  
                points_la = torch.cat(batch_labels, dim=0).to(device)  

                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=[points_co, points_la],
                    boxes=None,
                    masks=low_res_masks.to(device),
                )
            low_res_masks, _, upsampling_embeddings = sam_model.mask_decoder(
                image_embeddings=image_embedding.to(device), # (B, 384, 64, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
                dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
                multimask_output=False,
                middle_output=True,
                )
            prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        fg_predictions.append(torch.sigmoid(prev_masks))
        posterior_features.append(upsampling_embeddings)
    fg_predictions = torch.cat(fg_predictions, dim=1)
    predictions = torch.cat([bg_predictions, fg_predictions], dim=1)
    # masks = torch.argmax(torch.softmax(predictions, dim=1), dim=1)
    pred_soft = torch.softmax(predictions, dim=1)

    if args.num_classes < 3:
        posterior_features = torch.stack(posterior_features, dim=1).squeeze(1)
    else:
        posterior_features = torch.stack(posterior_features, dim=1).flatten(1,2)
    return image_embedding, posterior_features, pred_soft

# load dataset 3d
def load_dataset(args, num=None):
    test_list = 'test.txt'
    if 'Pancreas' in args.data_dir:
        return Pancreas(args.data_dir, split='train', num=num, patch_size=args.patch_size), test_list
    if 'BraTS2019' in args.data_dir:
        test_list = 'val.txt'
        return BraTS2019(args.data_dir, split='train', num=num, patch_size=args.patch_size), test_list
    if 'TBAD' in args.data_dir:
        test_list = os.path.join(args.data_dir,'ImageTBADlist/AD_{}/val.txt'.format(args.fold_num))
        return AortaDissection(args.data_dir, split='train', num=num, fold_num=args.fold_num), test_list

norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
def get_sam_features(img3D, label_batch, sam_model, args, device='cuda', 
                                click_method='random', click_method_aft='random', num_clicks=10, prev_masks=None,
                                GTM_rank_map=False, entropy_maps=None,
                                ):
    if 'OASIS' not in args.data_dir:
        img3D = norm_transform(img3D.squeeze(dim=1)) # (N, C, W, H, D)
        img3D = img3D.unsqueeze(dim=1)
        
    with torch.no_grad():
        image_embedding, hidden_states_out = sam_model.image_encoder(img3D.to(device), hidden_out=True) # (1, 384, 16, 16, 16)

    posterior_features = []
    fg_predictions = []
    # bg_predictions = torch.zeros_like(label_batch.unsqueeze(1)).to(device)+0.5
    
    for c in range(1, args.num_classes):
        gt3D = (label_batch==c).to(torch.long).unsqueeze(1).to(device)
        if prev_masks is None:
            prev_masks = torch.zeros_like(gt3D).to(device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(args.crop_size//4,args.crop_size//4,args.crop_size//4))

        for num_click in range(num_clicks):
            with torch.no_grad():
                if click_method == 'uncertain':
                    batch_points, batch_labels = click_methods[click_method](prev_masks.to(device), gt3D.to(device), entropy_maps)
                else:
                    batch_points, batch_labels = click_methods[click_method](prev_masks.to(device), gt3D.to(device))

                points_co = torch.cat(batch_points, dim=0).to(device)  
                points_la = torch.cat(batch_labels, dim=0).to(device)  

                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=[points_co, points_la],
                    boxes=None,
                    masks=low_res_masks.to(device),
                )
            low_res_masks, _, embeddings = sam_model.mask_decoder(
                image_embeddings=image_embedding.to(device), # (B, 384, 64, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
                dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
                multimask_output=False,
                middle_output=False,
                masked_feat_output=True,
                )
            prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        fg_predictions.append(torch.sigmoid(prev_masks))
        posterior_features.append(embeddings)
    fg_predictions = torch.cat(fg_predictions, dim=1)
    sum_foreground = torch.sum(fg_predictions, dim=1, keepdim=True)
    bg_predictions = torch.clamp(1.0 - sum_foreground, min=0.0)
    p_all = torch.cat([bg_predictions, fg_predictions], dim=1)
    masks = torch.argmax(p_all, dim=1)
    # pred_soft = torch.softmax(predictions, dim=1)

    if args.num_classes < 3:
        posterior_features = torch.stack(posterior_features, dim=1).squeeze(1)
    else:
        posterior_features = torch.stack(posterior_features, dim=1).flatten(1,2)

    if GTM_rank_map:
        # p_normalized = p_all / (p_all.sum(dim=1, keepdim=True) + 1e-8)
        p_normalized = torch.softmax(p_all, dim=1)
        entropy = -torch.sum(p_normalized * torch.log(p_normalized + 1e-8), dim=1)
        return hidden_states_out, posterior_features, masks, p_normalized
    else:
        return hidden_states_out, posterior_features, masks

def cal_dice(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    return (2 * intersect + smooth) / (z_sum + y_sum + smooth)

def load_from(sam_model, weight_path, device='cuda'):
    with torch.no_grad():
        model_dict = torch.load(weight_path, map_location=device)
        state_dict = model_dict['model_state_dict']
        sam_model.load_state_dict(state_dict)

    for n, p in sam_model.named_parameters():
        p.requires_grad = False
    return sam_model

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

