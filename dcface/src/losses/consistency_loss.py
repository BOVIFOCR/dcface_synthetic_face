import os, sys
import numpy as np

from src.general_utils.img_utils import temp_plot, prepare_text_img
import torch
import torch.nn.functional as F
from torch import nn

def calculate_x0_from_eps(eps, noisy_images, timesteps, scheduler, clip=True):
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    alpha_prod_t = scheduler.alphas_cumprod[timesteps.cpu()].clone()
    alpha_prod_t = alpha_prod_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(eps.device)
    beta_prod_t = 1 - alpha_prod_t
    x0_pred = (noisy_images - beta_prod_t ** (0.5) * eps) / alpha_prod_t ** (0.5)
    if clip:
        x0_pred = torch.clip(x0_pred, -1, 1)
    return x0_pred

def calc_time_depenent_loss(f1, f2, timesteps, version='simple_mean', max_timesteps=1000, metric='cossim', return_avg=True):
    if metric == 'cossim':
        losses = (1 - (f1 * f2).sum(-1))
    elif metric == 'l1':
        losses = mean_flat(torch.abs(f1 - f2))
    else:
        raise ValueError('')
    if version == 'simple_mean':
        losses = losses
    elif 'polynomial' in version:
        order = float(version.split('_')[1])
        weights = np.linspace(0, 1, max_timesteps+1)**order
        weights_tensor = torch.tensor(weights, dtype=losses.dtype, device=losses.device)
        losses = (losses * weights_tensor[timesteps])
    elif 'revpoly' in version:
        order = float(version.split('_')[1])
        weights = np.linspace(1, 0, max_timesteps+1)**order
        weights_tensor = torch.tensor(weights, dtype=losses.dtype, device=losses.device)
        losses = (losses * weights_tensor[timesteps])

    elif 'hindge' in version:
        # when T = 0 : original -> add threshold
        # when T = 1000 : random noise -> as close as possible
        # hindge_0.3 : cossine similarity has to be at least 0.3 to avoid loss when T = 0
        #                                           at least 1.0 to avoid loss when T = 1000
        assert metric == 'cossim'
        threshold = 1 - float(version.split('_')[1])
        assert threshold > 0.0
        threshold_tensor = np.linspace(1, 0, max_timesteps + 1)
        threshold_tensor = torch.tensor(threshold_tensor, dtype=losses.dtype, device=losses.device)
        threshold_tensor = threshold_tensor * threshold
        losses = torch.max(losses, threshold_tensor[timesteps])
        losses = losses

    elif 'polyhindg' in version:
        assert metric == 'cossim'
        order = float(version.split('_')[1])
        threshold = 1 - float(version.split('_')[2])
        assert threshold > 0.0
        threshold_tensor = np.linspace(1, 0, max_timesteps + 1) * threshold
        threshold_tensor = torch.tensor(threshold_tensor, dtype=losses.dtype, device=losses.device)
        threshold_tensor = threshold_tensor
        losses = torch.max(losses, threshold_tensor[timesteps])
        weights = np.linspace(0, 1, max_timesteps+1)**order
        weights_tensor = torch.tensor(weights, dtype=losses.dtype, device=losses.device)
        losses = losses * weights_tensor[timesteps]
        losses = losses
    else:
        raise ValueError('')

    if return_avg:
        return losses.mean()
    else:
        return losses

def calc_identity_consistency_loss(eps, timesteps, noisy_images, batch, pl_module, ):

    scheduler = pl_module.noise_scheduler
    recognition_model = pl_module.recognition_model
    x0_pred = calculate_x0_from_eps(eps, noisy_images, timesteps, scheduler)

    # # visual
    # for i in range(len(x0_pred)):
    #     text = prepare_text_img(f't={timesteps[i]}', width=112, height=30) / 255.
    #     text = torch.tensor(text.transpose(2,0,1)).to(clean_images.device)
    #     vis = torch.cat([text, clean_images[i], noisy_images[i], x0_pred[i]], dim=1)
    #     vis = vis.flip(0)
    #     temp_plot(vis, path='/mckim/temp/temp_{}.png'.format(timesteps[i]))

    x0_pred_feature, spatial = recognition_model(x0_pred)
    x0_pred_norm = torch.norm(x0_pred_feature, 2, -1, keepdim=True)
    x0_pred_feature = x0_pred_feature / x0_pred_norm
    if recognition_model.center is not None and pl_module.hparams.losses.identity_consistency_loss_source == 'center':
        center = recognition_model.center(batch['class_label'])
        cossim_loss = calc_time_depenent_loss(x0_pred_feature, center,
                                       timesteps=timesteps,
                                       version=pl_module.hparams.losses.identity_consistency_loss_version,
                                       max_timesteps=pl_module.hparams.sampler.num_train_timesteps,
                                       )
    elif pl_module.hparams.losses.identity_consistency_loss_source == 'image':
        orig_feature, _ = recognition_model(batch['image'])
        orig_feature_norm = torch.norm(orig_feature, 2, -1, keepdim=True)
        orig_feature = orig_feature / orig_feature_norm
        cossim_loss = calc_time_depenent_loss(x0_pred_feature, orig_feature,
                                              timesteps=timesteps,
                                              version=pl_module.hparams.losses.identity_consistency_loss_version,
                                              max_timesteps=pl_module.hparams.sampler.num_train_timesteps,
                                              )

    elif pl_module.hparams.losses.identity_consistency_loss_source == 'mix':
        if pl_module.hparams.losses.identity_consistency_loss_center_source == 'center':
            center = recognition_model.center(batch['class_label'])
            cossim_loss_center = calc_time_depenent_loss(x0_pred_feature, center,
                                                  timesteps=timesteps,
                                                  version=pl_module.hparams.losses.identity_consistency_loss_version,
                                                  max_timesteps=pl_module.hparams.sampler.num_train_timesteps,
                                                  return_avg=False,
                                                  )
        elif pl_module.hparams.losses.identity_consistency_loss_center_source == 'id_image':
            id_feature, _ = recognition_model(batch['id_image'])
            id_feature_norm = torch.norm(id_feature, 2, -1, keepdim=True)
            id_feature = id_feature / id_feature_norm
            cossim_loss_center = calc_time_depenent_loss(x0_pred_feature, id_feature,
                                                        timesteps=timesteps,
                                                        version=pl_module.hparams.losses.identity_consistency_loss_version,
                                                        max_timesteps=pl_module.hparams.sampler.num_train_timesteps,
                                                        return_avg=False,
                                                        )
        else:
            raise ValueError(f'{pl_module.hparams.losses.identity_consistency_loss_center_source} '
                             f'pl_module.hparams.losses.identity_consistency_loss_center_source not right')

        orig_feature, _ = recognition_model(batch['image'])
        orig_feature_norm = torch.norm(orig_feature, 2, -1, keepdim=True)
        orig_feature = orig_feature / orig_feature_norm
        cossim_loss_image = calc_time_depenent_loss(x0_pred_feature, orig_feature,
                                       timesteps=timesteps,
                                       version=pl_module.hparams.losses.identity_consistency_loss_version,
                                       max_timesteps=pl_module.hparams.sampler.num_train_timesteps,
                                       return_avg=False,
                                       )

        order = float(pl_module.hparams.losses.identity_consistency_mix_loss_version.split('_')[1])

        if pl_module.hparams.losses.identity_consistency_loss_weight_start_bias > 0:
            weight_start = pl_module.hparams.losses.identity_consistency_loss_weight_start_bias
            assert weight_start < 1.0
        else:
            weight_start = 0.0

        weights = np.linspace(weight_start, 1, pl_module.hparams.sampler.num_train_timesteps+1)**order
        weights_tensor = torch.tensor(weights, dtype=cossim_loss_center.dtype, device=cossim_loss_center.device)
        mix_weights = weights_tensor[timesteps]

        cossim_loss = cossim_loss_center * mix_weights + cossim_loss_image * (1-mix_weights)

        if pl_module.hparams.losses.identity_consistency_loss_time_cut > 0:
            time_cut_ratio = pl_module.hparams.losses.identity_consistency_loss_time_cut
            time_cut_index = int(len(weights_tensor) * time_cut_ratio)
            cut_weight = torch.ones_like(weights_tensor)
            cut_weight[:time_cut_index] = 0
            cossim_loss = cossim_loss * cut_weight[timesteps]

        cossim_loss = cossim_loss.mean()

    elif pl_module.hparams.losses.identity_consistency_loss_source == 'id_image':
        orig_feature, _ = recognition_model(batch['id_image'])
        orig_feature_norm = torch.norm(orig_feature, 2, -1, keepdim=True)
        orig_feature = orig_feature / orig_feature_norm
        cossim_loss = calc_time_depenent_loss(x0_pred_feature, orig_feature,
                                              timesteps=timesteps,
                                              version=pl_module.hparams.losses.identity_consistency_loss_version,
                                              max_timesteps=pl_module.hparams.sampler.num_train_timesteps,
                                              )

    if pl_module.hparams.losses.spatial_consistency_loss_lambda > 0.0:
        pred_mean, pred_var = extract_mean_var(spatial)
        orig_feature, orig_spatial = recognition_model(batch['image'])
        orig_mean, orig_var = extract_mean_var(orig_spatial)
        spat_mean_loss = calc_time_depenent_loss(pred_mean, orig_mean, timesteps=timesteps,
                                                 version=pl_module.hparams.losses.spatial_consistency_loss_version,
                                                 max_timesteps=pl_module.hparams.sampler.num_train_timesteps,
                                                 metric='l1')
        spat_var_loss = calc_time_depenent_loss(pred_var, orig_var, timesteps=timesteps,
                                                version=pl_module.hparams.losses.spatial_consistency_loss_version,
                                                max_timesteps=pl_module.hparams.sampler.num_train_timesteps,
                                                metric='l1')
        spatial_loss = (spat_mean_loss + spat_var_loss)/2
    else:
        spatial_loss = None

    return cossim_loss, spatial_loss


# Bernardo
def calc_identity_consistency_loss_precomputed_stylized_face(eps, timesteps, noisy_images, batch, pl_module,
                                                             x0_pred, x0_pred_feature, spatial):
    # scheduler = pl_module.noise_scheduler
    recognition_model = pl_module.recognition_model
    # x0_pred = calculate_x0_from_eps(eps, noisy_images, timesteps, scheduler)

    # # visual
    # for i in range(len(x0_pred)):
    #     text = prepare_text_img(f't={timesteps[i]}', width=112, height=30) / 255.
    #     text = torch.tensor(text.transpose(2,0,1)).to(clean_images.device)
    #     vis = torch.cat([text, clean_images[i], noisy_images[i], x0_pred[i]], dim=1)
    #     vis = vis.flip(0)
    #     temp_plot(vis, path='/mckim/temp/temp_{}.png'.format(timesteps[i]))

    # x0_pred_feature, spatial = recognition_model(x0_pred)
    # x0_pred_norm = torch.norm(x0_pred_feature, 2, -1, keepdim=True)
    # x0_pred_feature = x0_pred_feature / x0_pred_norm
    
    if recognition_model.center is not None and pl_module.hparams.losses.identity_consistency_loss_source == 'center':
        center = recognition_model.center(batch['class_label'])
        cossim_loss = calc_time_depenent_loss(x0_pred_feature, center,
                                       timesteps=timesteps,
                                       version=pl_module.hparams.losses.identity_consistency_loss_version,
                                       max_timesteps=pl_module.hparams.sampler.num_train_timesteps,
                                       )

    elif pl_module.hparams.losses.identity_consistency_loss_source == 'image':
        orig_feature, _ = recognition_model(batch['image'])
        orig_feature_norm = torch.norm(orig_feature, 2, -1, keepdim=True)
        orig_feature = orig_feature / orig_feature_norm
        cossim_loss = calc_time_depenent_loss(x0_pred_feature, orig_feature,
                                              timesteps=timesteps,
                                              version=pl_module.hparams.losses.identity_consistency_loss_version,
                                              max_timesteps=pl_module.hparams.sampler.num_train_timesteps,
                                              )
    # DCFace default
    elif pl_module.hparams.losses.identity_consistency_loss_source == 'mix':
        if pl_module.hparams.losses.identity_consistency_loss_center_source == 'center':
            center = recognition_model.center(batch['class_label'])
            cossim_loss_center = calc_time_depenent_loss(x0_pred_feature, center,
                                                  timesteps=timesteps,
                                                  version=pl_module.hparams.losses.identity_consistency_loss_version,
                                                  max_timesteps=pl_module.hparams.sampler.num_train_timesteps,
                                                  return_avg=False,
                                                  )
        # DCFace default
        elif pl_module.hparams.losses.identity_consistency_loss_center_source == 'id_image':
            id_feature, _ = recognition_model(batch['id_image'])
            id_feature_norm = torch.norm(id_feature, 2, -1, keepdim=True)
            id_feature = id_feature / id_feature_norm
            cossim_loss_center = calc_time_depenent_loss(x0_pred_feature, id_feature,
                                                        timesteps=timesteps,
                                                        version=pl_module.hparams.losses.identity_consistency_loss_version,
                                                        max_timesteps=pl_module.hparams.sampler.num_train_timesteps,
                                                        return_avg=False,
                                                        )
        else:
            raise ValueError(f'{pl_module.hparams.losses.identity_consistency_loss_center_source} '
                             f'pl_module.hparams.losses.identity_consistency_loss_center_source not right')

        orig_feature, _ = recognition_model(batch['image'])
        orig_feature_norm = torch.norm(orig_feature, 2, -1, keepdim=True)
        orig_feature = orig_feature / orig_feature_norm
        cossim_loss_image = calc_time_depenent_loss(x0_pred_feature, orig_feature,
                                       timesteps=timesteps,
                                       version=pl_module.hparams.losses.identity_consistency_loss_version,
                                       max_timesteps=pl_module.hparams.sampler.num_train_timesteps,
                                       return_avg=False,
                                       )

        order = float(pl_module.hparams.losses.identity_consistency_mix_loss_version.split('_')[1])

        if pl_module.hparams.losses.identity_consistency_loss_weight_start_bias > 0:
            weight_start = pl_module.hparams.losses.identity_consistency_loss_weight_start_bias
            assert weight_start < 1.0
        else:
            weight_start = 0.0

        weights = np.linspace(weight_start, 1, pl_module.hparams.sampler.num_train_timesteps+1)**order
        weights_tensor = torch.tensor(weights, dtype=cossim_loss_center.dtype, device=cossim_loss_center.device)
        mix_weights = weights_tensor[timesteps]

        cossim_loss = cossim_loss_center * mix_weights + cossim_loss_image * (1-mix_weights)

        if pl_module.hparams.losses.identity_consistency_loss_time_cut > 0:
            time_cut_ratio = pl_module.hparams.losses.identity_consistency_loss_time_cut
            time_cut_index = int(len(weights_tensor) * time_cut_ratio)
            cut_weight = torch.ones_like(weights_tensor)
            cut_weight[:time_cut_index] = 0
            cossim_loss = cossim_loss * cut_weight[timesteps]

        cossim_loss = cossim_loss.mean()

    elif pl_module.hparams.losses.identity_consistency_loss_source == 'id_image':
        orig_feature, _ = recognition_model(batch['id_image'])
        orig_feature_norm = torch.norm(orig_feature, 2, -1, keepdim=True)
        orig_feature = orig_feature / orig_feature_norm
        cossim_loss = calc_time_depenent_loss(x0_pred_feature, orig_feature,
                                              timesteps=timesteps,
                                              version=pl_module.hparams.losses.identity_consistency_loss_version,
                                              max_timesteps=pl_module.hparams.sampler.num_train_timesteps,
                                              )

    if pl_module.hparams.losses.spatial_consistency_loss_lambda > 0.0:
        pred_mean, pred_var = extract_mean_var(spatial)
        orig_feature, orig_spatial = recognition_model(batch['image'])
        orig_mean, orig_var = extract_mean_var(orig_spatial)
        spat_mean_loss = calc_time_depenent_loss(pred_mean, orig_mean, timesteps=timesteps,
                                                 version=pl_module.hparams.losses.spatial_consistency_loss_version,
                                                 max_timesteps=pl_module.hparams.sampler.num_train_timesteps,
                                                 metric='l1')
        spat_var_loss = calc_time_depenent_loss(pred_var, orig_var, timesteps=timesteps,
                                                version=pl_module.hparams.losses.spatial_consistency_loss_version,
                                                max_timesteps=pl_module.hparams.sampler.num_train_timesteps,
                                                metric='l1')
        spatial_loss = (spat_mean_loss + spat_var_loss)/2
    
    # DCFace default
    else:
        spatial_loss = None

    return cossim_loss, spatial_loss


# Bernardo
def euclid_distance(batch1, batch2):
    assert batch1.shape == batch2.shape
    distances = torch.norm(batch1 - batch2, dim=1)
    return distances


# Bernardo
def cosine_simil(batch1, batch2):
    assert batch1.shape == batch2.shape
    batch1 = F.normalize(batch1, p=2, dim=1)
    batch2 = F.normalize(batch2, p=2, dim=1)
    cosine_sim = torch.sum(batch1 * batch2, dim=1)
    return cosine_sim


# Bernardo
def calc_3dmm_consistency_loss(eps, timesteps, noisy_images, batch, pl_module, ):
    reconstruction_model = pl_module.reconstruction_model
    scheduler = pl_module.noise_scheduler
    recognition_model = pl_module.recognition_model

    x0_pred = calculate_x0_from_eps(eps, noisy_images, timesteps, scheduler)
    x0_pred_feature, spatial = recognition_model(x0_pred)
    x0_pred_norm = torch.norm(x0_pred_feature, 2, -1, keepdim=True)
    x0_pred_feature = x0_pred_feature / x0_pred_norm

    id_image = batch['id_image']
    id_feature, _ = recognition_model(id_image)
    id_feature_norm = torch.norm(id_feature, 2, -1, keepdim=True)
    id_feature = id_feature / id_feature_norm

    x0_pred_pointcloud, x0_pred_3dmm, x0_pred_render_image = reconstruction_model(x0_pred_feature)
    id_image_pointcloud, id_image_3dmm, id_image_render_image = reconstruction_model(id_feature)

    euclDist_x0Pred_idImage = euclid_distance(x0_pred_3dmm, id_image_3dmm)

    threeDMM_loss = euclDist_x0Pred_idImage.mean()
    return threeDMM_loss, \
           x0_pred, id_image, \
           x0_pred_pointcloud, x0_pred_3dmm, x0_pred_render_image, \
           id_image_pointcloud, id_image_3dmm, id_image_render_image


# Bernardo
def calc_bfm_consistency_loss(eps, timesteps, noisy_images, batch, pl_module):
    reconstruction_model = pl_module.reconstruction_model
    scheduler = pl_module.noise_scheduler
    recognition_model = pl_module.recognition_model

    x0_pred = calculate_x0_from_eps(eps, noisy_images, timesteps, scheduler)
    id_image = batch['id_image']
    orig = batch['orig']
    
    x0_pred_id,  x0_pred_exp,  x0_pred_angle,  x0_pred_trans  = reconstruction_model(x0_pred)
    id_image_id, id_image_exp, id_image_angle, id_image_trans = reconstruction_model(id_image)
    orig_id,     orig_exp,     orig_angle,     orig_trans     = reconstruction_model(orig)

    euclDist_ident_x0Pred_idImage = euclid_distance(x0_pred_id, id_image_id)
    
    euclDist_express_x0Pred_origImage = euclid_distance(x0_pred_exp, orig_exp)
    euclDist_pose_x0Pred_origImage    = euclid_distance(x0_pred_angle, orig_angle)

    bfm_loss = euclDist_ident_x0Pred_idImage.mean() + (euclDist_express_x0Pred_origImage.mean() + euclDist_pose_x0Pred_origImage.mean())
    return bfm_loss


# Bernardo
hash_size=32
hash_weights = None
hash_file_path = 'src/HRN/assets/hash_weights_randn_size=32.pt'
if not os.path.isfile(hash_file_path):
    hash_weights = torch.randn((3*112*112, hash_size), dtype=torch.float32, device='cuda')
    print(f'Saving hash_weights: {hash_file_path}')
    torch.save(hash_weights, hash_file_path)
else:
    print(f'Loading hash_weights from file: {hash_file_path}')
    hash_weights = torch.load(hash_file_path, map_location='cuda')
def compute_hash(tensor_batch):
    # tensor_batch = tensor_batch.to('cuda')
    B, C, H, W = tensor_batch.shape
    flattened = tensor_batch.view(B, -1)
    hash_tensor = torch.zeros((B, hash_size), dtype=torch.float32, device='cuda')
    hash_tensor = torch.matmul(flattened, hash_weights)
    hash_tensor_str = [str((hash_t[:4]+hash_t[4:8]-hash_t[8:12]).tolist()).replace(' ','') for hash_t in hash_tensor]
    return hash_tensor, hash_tensor_str


# Bernardo
def make_coeff_file_path(hash_str_list, coeffs_dir, ext='.pt'):
    for i in range(len(hash_str_list)):
        hash_str_list[i] = os.path.join(coeffs_dir, hash_str_list[i]+ext)
    return hash_str_list


# Bernardo
def exists_bfm_coeffs(coeff_paths):
    for path in coeff_paths:
        if not os.path.isfile(path):
            return False
    return True


# Bernardo
def save_bfm_coeffs(paths_list=[''], bfm_coeffs=torch.Tensor()):
    os.makedirs(os.path.dirname(paths_list[0]), exist_ok=True)
    for path, coeff in zip(paths_list, bfm_coeffs):
        torch.save(coeff, path)


# Bernardo
def load_bfm_coeffs(paths_list=['']):
    bfm_coeffs = torch.zeros((len(paths_list), 257), dtype=torch.float32, device='cuda:0')
    for i in range(len(paths_list)):
        bfm_coeffs[i] = torch.load(paths_list[i])
    return bfm_coeffs


# Bernardo
def make_batch_bfm_coeffs(bfm_coeffs_dict={}, hashes=['']):
    bfm_coeffs_batch = torch.zeros((len(hashes), 257), dtype=torch.float32, device='cuda:0')
    for i in range(len(hashes)):
        bfm_coeffs_batch[i] = bfm_coeffs_dict[hashes[i]]
    return bfm_coeffs_batch


# Bernardo
def split_bfm_coeffs(bfm_coeffs):
    batch_id    = bfm_coeffs[:, :80]
    batch_exp   = bfm_coeffs[:, 80:144]
    batch_angle = bfm_coeffs[:, 224:227]
    batch_trans = bfm_coeffs[:, 254:]
    return batch_id, batch_exp, batch_angle, batch_trans


# Bernardo
def calc_bfm_consistency_loss_precomputed_stylized_face(eps, timesteps, noisy_images, batch, pl_module,
                                                        x0_pred, x0_pred_feature, spatial, hparams, bfm_coeffs_dict):
    reconstruction_model = pl_module.reconstruction_model
    # scheduler = pl_module.noise_scheduler
    # recognition_model = pl_module.recognition_model

    # x0_pred = calculate_x0_from_eps(eps, noisy_images, timesteps, scheduler)
    id_image = batch['id_image']
    orig = batch['orig']

    if not bfm_coeffs_dict is None:
        id_image_hash_float, id_image_hash_str = compute_hash(id_image)
        orig_hash_float, orig_hash_str = compute_hash(orig)

        id_image_bfm_coeffs = make_batch_bfm_coeffs(bfm_coeffs_dict, id_image_hash_str)
        orig_bfm_coeffs = make_batch_bfm_coeffs(bfm_coeffs_dict, orig_hash_str)
    else:
        id_image_bfm_coeffs = reconstruction_model(id_image)
        orig_bfm_coeffs = reconstruction_model(orig)
    x0_pred_bfm_coeffs = reconstruction_model(x0_pred)

    id_image_bfm_coeffs = reconstruction_model.rescale_bfm_coeffs(id_image_bfm_coeffs)
    orig_bfm_coeffs = reconstruction_model.rescale_bfm_coeffs(orig_bfm_coeffs)
    x0_pred_bfm_coeffs = reconstruction_model.rescale_bfm_coeffs(x0_pred_bfm_coeffs)

    id_image_id, id_image_exp, id_image_angle, id_image_trans = split_bfm_coeffs(id_image_bfm_coeffs)
    orig_id,     orig_exp,     orig_angle,     orig_trans     = split_bfm_coeffs(orig_bfm_coeffs)
    x0_pred_id,  x0_pred_exp,  x0_pred_angle,  x0_pred_trans  = split_bfm_coeffs(x0_pred_bfm_coeffs)
    
    # euclDist_ident_x0Pred_idImage     = euclid_distance(x0_pred_id, id_image_id)
    # euclDist_express_x0Pred_origImage = euclid_distance(x0_pred_exp, orig_exp)
    # euclDist_pose_x0Pred_origImage    = euclid_distance(x0_pred_angle, orig_angle)

    cosSim_ident_x0Pred_idImage     = cosine_simil(x0_pred_id, id_image_id)
    cosSim_express_x0Pred_origImage = cosine_simil(x0_pred_exp, orig_exp)
    cosSim_pose_x0Pred_origImage    = cosine_simil(x0_pred_angle, orig_angle)

    bfm_id_loss      = (1.0 - cosSim_ident_x0Pred_idImage).mean()
    bfm_express_loss = (1.0 - cosSim_express_x0Pred_origImage).mean()
    bfm_pose_loss    = (1.0 - cosSim_pose_x0Pred_origImage).mean()
    return bfm_id_loss, bfm_express_loss, bfm_pose_loss


def extract_mean_var(spatial):
    spatial_feature = spatial[0]
    B, C, _, side = spatial_feature.shape
    patch_side = side // 2
    global_mean = nn.AvgPool2d(side, side)(spatial_feature).view(B, C, -1)
    global_var = global_mean - nn.AvgPool2d(side, side)(spatial_feature**2).view(B, C, -1)
    patch_mean = nn.AvgPool2d(patch_side, patch_side)(spatial_feature).view(B, C, -1)
    patch_var = patch_mean - nn.AvgPool2d(patch_side, patch_side)(spatial_feature**2).view(B, C, -1)
    mean_stat = torch.cat([global_mean, patch_mean], dim=2)
    var_stat = torch.cat([global_var, patch_var], dim=2)
    return mean_stat, var_stat


def mean_flat(tensor):
    if len(tensor.shape) == 1:
        return tensor
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
