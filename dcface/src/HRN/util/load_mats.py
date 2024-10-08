import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat
from array import array
import os.path as osp
import torch

# load expression basis
def LoadExpBasis(bfm_folder='asset/BFM'):
    n_vertex = 53215
    Expbin = open(osp.join(bfm_folder, 'Exp_Pca.bin'), 'rb')
    exp_dim = array('i')
    exp_dim.fromfile(Expbin, 1)
    expMU = array('f')
    expPC = array('f')
    expMU.fromfile(Expbin, 3*n_vertex)
    expPC.fromfile(Expbin, 3*exp_dim[0]*n_vertex)
    Expbin.close()

    expPC = np.array(expPC)
    expPC = np.reshape(expPC, [exp_dim[0], -1])
    expPC = np.transpose(expPC)

    expEV = np.loadtxt(osp.join(bfm_folder, 'std_exp.txt'))

    return expPC, expEV


# transfer original BFM09 to our face model
def transferBFM09(bfm_folder='BFM'):
    print('Transfer BFM09 to BFM_model_front......')
    original_BFM = loadmat(osp.join(bfm_folder, '01_MorphableModel.mat'))
    shapePC = original_BFM['shapePC']  # shape basis, 160470*199
    shapeEV = original_BFM['shapeEV']  # corresponding eigen value, 199*1
    shapeMU = original_BFM['shapeMU']  # mean face, 160470*1
    texPC = original_BFM['texPC']  # texture basis, 160470*199
    texEV = original_BFM['texEV']  # eigen value, 199*1
    texMU = original_BFM['texMU']  # mean texture, 160470*1

    expPC, expEV = LoadExpBasis()

    # transfer BFM09 to our face model

    idBase = shapePC*np.reshape(shapeEV, [-1, 199])
    idBase = idBase/1e5  # unify the scale to decimeter
    idBase = idBase[:, :80]  # use only first 80 basis

    exBase = expPC*np.reshape(expEV, [-1, 79])
    exBase = exBase/1e5  # unify the scale to decimeter
    exBase = exBase[:, :64]  # use only first 64 basis

    texBase = texPC*np.reshape(texEV, [-1, 199])
    texBase = texBase[:, :80]  # use only first 80 basis

    # our face model is cropped along face landmarks and contains only 35709 vertex.
    # original BFM09 contains 53490 vertex, and expression basis provided by Guo et al. contains 53215 vertex.
    # thus we select corresponding vertex to get our face model.

    index_exp = loadmat(osp.join(bfm_folder, 'BFM_front_idx.mat'))
    index_exp = index_exp['idx'].astype(np.int32) - 1  # starts from 0 (to 53215)

    index_shape = loadmat(osp.join(bfm_folder, 'BFM_exp_idx.mat'))
    index_shape = index_shape['trimIndex'].astype(
        np.int32) - 1  # starts from 0 (to 53490)
    index_shape = index_shape[index_exp]

    idBase = np.reshape(idBase, [-1, 3, 80])
    idBase = idBase[index_shape, :, :]
    idBase = np.reshape(idBase, [-1, 80])

    texBase = np.reshape(texBase, [-1, 3, 80])
    texBase = texBase[index_shape, :, :]
    texBase = np.reshape(texBase, [-1, 80])

    exBase = np.reshape(exBase, [-1, 3, 64])
    exBase = exBase[index_exp, :, :]
    exBase = np.reshape(exBase, [-1, 64])

    meanshape = np.reshape(shapeMU, [-1, 3])/1e5
    meanshape = meanshape[index_shape, :]
    meanshape = np.reshape(meanshape, [1, -1])

    meantex = np.reshape(texMU, [-1, 3])
    meantex = meantex[index_shape, :]
    meantex = np.reshape(meantex, [1, -1])

    # other info contains triangles, region used for computing photometric loss,
    # region used for skin texture regularization, and 68 landmarks index etc.
    other_info = loadmat(osp.join(bfm_folder, 'facemodel_info.mat'))
    frontmask2_idx = other_info['frontmask2_idx']
    skinmask = other_info['skinmask']
    keypoints = other_info['keypoints']
    point_buf = other_info['point_buf']
    tri = other_info['tri']
    tri_mask2 = other_info['tri_mask2']

    # save our face model
    savemat(osp.join(bfm_folder, 'BFM_model_front.mat'), {'meanshape': meanshape, 'meantex': meantex, 'idBase': idBase, 'exBase': exBase, 'texBase': texBase,
            'tri': tri, 'point_buf': point_buf, 'tri_mask2': tri_mask2, 'keypoints': keypoints, 'frontmask2_idx': frontmask2_idx, 'skinmask': skinmask})





# Bernardo
def LoadBFM09(bfm_folder='BFM', exp_folder='EXP', device='cpu'):
    print('Loading BFM09 basis...')
    original_BFM = loadmat(osp.join(bfm_folder, '01_MorphableModel.mat'))
    shapePC = original_BFM['shapePC']  # shape basis, 160470*199
    shapeEV = original_BFM['shapeEV']  # corresponding eigen value, 199*1
    shapeMU = original_BFM['shapeMU']  # mean face, 160470*1
    texPC = original_BFM['texPC']  # texture basis, 160470*199
    texEV = original_BFM['texEV']  # eigen value, 199*1
    texMU = original_BFM['texMU']  # mean texture, 160470*1

    print('Loading Expression basis...')
    # expPC, expEV = LoadExpBasis()           # original
    expPC, expEV = LoadExpBasis(exp_folder)   # Bernardo

    # transfer BFM09 to our face model

    idBase = shapePC*np.reshape(shapeEV, [-1, 199])
    idBase = idBase/1e5  # unify the scale to decimeter
    idBase = idBase[:, :80]  # use only first 80 basis
    shapeEV = shapeEV[:80, :]/1e5    # Bernardo

    exBase = expPC*np.reshape(expEV, [-1, 79])
    exBase = exBase/1e5  # unify the scale to decimeter
    exBase = exBase[:, :64]  # use only first 64 basis
    expEV = expEV[:64]/1e5           # Bernardo

    texBase = texPC*np.reshape(texEV, [-1, 199])
    texBase = texBase[:, :80]  # use only first 80 basis
    texEV = texEV[:80, :]/1e5        # Bernardo

    # our face model is cropped along face landmarks and contains only 35709 vertex.
    # original BFM09 contains 53490 vertex, and expression basis provided by Guo et al. contains 53215 vertex.
    # thus we select corresponding vertex to get our face model.

    # index_exp = loadmat(osp.join(bfm_folder, 'BFM_front_idx.mat'))   # original
    index_exp = loadmat(osp.join(exp_folder, 'BFM_front_idx.mat'))     # Bernardo
    index_exp = index_exp['idx'].astype(np.int32) - 1  # starts from 0 (to 53215)

    # index_shape = loadmat(osp.join(bfm_folder, 'BFM_exp_idx.mat'))   # original
    index_shape = loadmat(osp.join(exp_folder, 'BFM_exp_idx.mat'))     # Bernardo
    index_shape = index_shape['trimIndex'].astype(
        np.int32) - 1  # starts from 0 (to 53490)
    index_shape = index_shape[index_exp]

    idBase = np.reshape(idBase, [-1, 3, 80])
    idBase = idBase[index_shape, :, :]
    idBase = np.reshape(idBase, [-1, 80])

    texBase = np.reshape(texBase, [-1, 3, 80])
    texBase = texBase[index_shape, :, :]
    texBase = np.reshape(texBase, [-1, 80])

    exBase = np.reshape(exBase, [-1, 3, 64])
    exBase = exBase[index_exp, :, :]
    exBase = np.reshape(exBase, [-1, 64])

    meanshape = np.reshape(shapeMU, [-1, 3])/1e5
    meanshape = meanshape[index_shape, :]
    meanshape = np.reshape(meanshape, [1, -1])

    meantex = np.reshape(texMU, [-1, 3])
    meantex = meantex[index_shape, :]
    meantex = np.reshape(meantex, [1, -1])

    # other info contains triangles, region used for computing photometric loss,
    # region used for skin texture regularization, and 68 landmarks index etc.
    # other_info = loadmat(osp.join(bfm_folder, 'facemodel_info.mat'))   # original
    other_info = loadmat(osp.join(exp_folder, 'facemodel_info.mat'))     # Bernardo
    frontmask2_idx = other_info['frontmask2_idx']
    skinmask = other_info['skinmask']
    keypoints = other_info['keypoints']
    point_buf = other_info['point_buf']
    tri = other_info['tri']
    tri_mask2 = other_info['tri_mask2']

    if 'gpu' in device or 'cuda' in device:
        meanshape = torch.from_numpy(meanshape).float().to(device)
        meantex =   torch.from_numpy(meantex).float().to(device)
        idBase = torch.from_numpy(idBase).float().to(device)
        shapeEV = torch.from_numpy(shapeEV).float().to(device)
        exBase = torch.from_numpy(exBase).float().to(device)
        expEV = torch.from_numpy(expEV).float().to(device)
        texBase = torch.from_numpy(texBase).float().to(device)
        texEV = torch.from_numpy(texEV).float().to(device)
        tri = torch.from_numpy(tri).float().to(device)
        point_buf = torch.from_numpy(point_buf).float().to(device)
        tri_mask2 = torch.from_numpy(tri_mask2).float().to(device)
        keypoints = torch.from_numpy(keypoints).float().to(device)
        frontmask2_idx = torch.from_numpy(frontmask2_idx).float().to(device)
        skinmask = torch.from_numpy(skinmask).float().to(device)

    # save our face model
    # savemat(osp.join(bfm_folder, 'BFM_model_front.mat'), {'meanshape': meanshape, 'meantex': meantex, 'idBase': idBase, 'exBase': exBase, 'texBase': texBase,
    #         'tri': tri, 'point_buf': point_buf, 'tri_mask2': tri_mask2, 'keypoints': keypoints, 'frontmask2_idx': frontmask2_idx, 'skinmask': skinmask})
    return {'meanshape': meanshape, 'meantex': meantex, 'idBase': idBase, 'idEV':shapeEV, 'exBase': exBase, 'exEV':expEV, 'texBase': texBase, 'texEV': texEV,
            'tri': tri, 'point_buf': point_buf, 'tri_mask2': tri_mask2, 'keypoints': keypoints, 'frontmask2_idx': frontmask2_idx, 'skinmask': skinmask}





# load landmarks for standard face, which is used for image preprocessing
def load_lm3d(bfm_folder):

    Lm3D = loadmat(osp.join(bfm_folder, 'similarity_Lm3D_all.mat'))
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D

