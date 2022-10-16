import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from dataset import Dataset

# note:
# use itk here will cause deadlock after the first training epoch 
# when using multithread (dataloader num_workers > 0) but reason unknown
import SimpleITK as sitk 

def resample_image(source_image, size, spacing, origin):
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((int(size[0]), int(size[1]), int(size[2])))
    resampler.SetOutputSpacing((float(spacing[0]), float(spacing[1]), float(spacing[2])))
    resampler.SetOutputOrigin((float(origin[0]), float(origin[1]), float(origin[2])))
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    image = resampler.Execute(source_image)
    return image

def read_image(filename):
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    image = reader.Execute()
    size = np.array(image.GetSize(), dtype=np.int64)
    spacing = np.array(image.GetSpacing(), dtype=np.float)
    origin = np.array(image.GetOrigin(), dtype=np.float)
    image_array = sitk.GetArrayFromImage(image)
    image_array = image_array.astype(dtype=np.uint8)
    return {'data':image_array, 'size':size, 'spacing':spacing, 'origin':origin}

def write_image(image, filename):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(filename)
    writer.Execute(image)
    return

def predict(net, dataloader, criterion,loss_weights,gt_pb_dir, gt_oar_dir, output_file, output_dir):
    net.eval()
    pb_dice_dict = {}
    oar1_dice_dict = {}
    oar2_dice_dict = {}

    # create output directory if needed
    if output_file:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        #os.makedirs(output_dir)

    cur_casename = ''
    cur_spacing = np.zeros([], dtype=np.float)
    cur_origin = np.zeros([], dtype=np.float)
    pd_pb_volume = np.zeros([], dtype=np.uint8)

    val_loss = 0
    val_sample_num = 0
    for batch_id, batch in enumerate(dataloader):
        # fetch data
        imgs = batch['data']
        dimgs = batch['data_dsample']
        filenames = batch['filename']
        origin = batch['origin']
        spacing = batch['spacing']

        gt_pb_mask = batch['pb_label']
        gt_oar_mask = batch['oar_label']
        gt_pb_dmask = batch['pb_dlabel']
        gt_oar_dmask = batch['oar_dlabel']
        n = len(imgs)

        # convert to GPU memory
        imgs = imgs.cuda()
        dimgs = dimgs.cuda()
        gt_pb_mask = gt_pb_mask.cuda()
        gt_oar_mask = gt_oar_mask.cuda()
        gt_pb_dmask = gt_pb_dmask.cuda()
        gt_oar_dmask = gt_oar_dmask.cuda()

        # forward propagation
        pd_oar_prob, pd_pb_prob, pd_oar_dprob, pd_pb_dprob, _ = net(imgs, dimgs)

        with torch.no_grad():
            loss_pb = criterion(pd_pb_prob, gt_pb_mask)
            loss_oar = criterion(pd_oar_prob, gt_oar_mask)
            loss_doar = criterion(pd_oar_dprob, gt_oar_dmask)
            loss_dpb = criterion(pd_pb_dprob, gt_pb_dmask)
            loss = loss_weights[0] * loss_pb + loss_weights[1] * loss_oar + loss_weights[2] * loss_dpb + loss_weights[2] * loss_doar
            val_loss += n * loss.item()
            val_sample_num += n

        pd_pb_mask = torch.argmax(pd_pb_prob, dim=1, keepdim=True)
        pd_oar_mask = torch.argmax(pd_oar_prob, dim=1, keepdim=True)

        batch_size = pd_pb_mask.shape[0]
        
        for i in range(batch_size):
            filename = filenames[i]
            casename = filename.split('/')[0]
            slice_id = int(filename.split('/')[1].split('.')[0])

            pd_pb_array = (pd_pb_mask==1)[i,:].contiguous().cpu().numpy().astype(dtype=np.uint8)
            pd_oar1_array = (pd_oar_mask==1)[i,:].contiguous().cpu().numpy().astype(dtype=np.uint8)
            pd_oar2_array = (pd_oar_mask==2)[i,:].contiguous().cpu().numpy().astype(dtype=np.uint8)

            # if a new volume started or get the last sample
            if casename != cur_casename or (batch_id == len(dataloader)-1 and i == batch_size-1):
                # output last volume
                if cur_casename != '':
                    # get geometry info from ground-truth image
                    gt_pb_image = read_image(gt_pb_dir + cur_casename)
                    new_size = gt_pb_image['size']
                    new_origin = gt_pb_image['origin']
                    new_spacing = gt_pb_image['spacing']
                    gt_pb_volume = gt_pb_image['data']
                    gt_oar_image = read_image(gt_oar_dir + cur_casename)
                    new_size = gt_oar_image['size']
                    new_origin = gt_oar_image['origin']
                    new_spacing = gt_oar_image['spacing']
                    gt_oar_volume = gt_oar_image['data']
                    
                    if batch_id == len(dataloader)-1 and i == batch_size-1:
                        pd_pb_volume[slice_id,:] = pd_pb_array
                        pd_oar1_volume[slice_id,:] = pd_oar1_array
                        pd_oar2_volume[slice_id,:] = pd_oar2_array

                    pd_pb_image = sitk.GetImageFromArray(pd_pb_volume)
                    pd_pb_image.SetOrigin((float(cur_origin[0]), float(cur_origin[1]), float(new_origin[2])))
                    pd_pb_image.SetSpacing((float(cur_spacing[0]), float(cur_spacing[1]), float(new_spacing[2])))
                    pd_oar1_image = sitk.GetImageFromArray(pd_oar1_volume)
                    pd_oar1_image.SetOrigin((float(cur_origin[0]), float(cur_origin[1]), float(new_origin[2])))
                    pd_oar1_image.SetSpacing((float(cur_spacing[0]), float(cur_spacing[1]), float(new_spacing[2])))
                    pd_oar2_image = sitk.GetImageFromArray(pd_oar2_volume)
                    pd_oar2_image.SetOrigin((float(cur_origin[0]), float(cur_origin[1]), float(new_origin[2])))
                    pd_oar2_image.SetSpacing((float(cur_spacing[0]), float(cur_spacing[1]), float(new_spacing[2])))
                    
                    # resample predicted image to ground-truth image resolution
                    pd_pb_image = resample_image(pd_pb_image, size=new_size, spacing=new_spacing, origin=new_origin)
                    pd_pb_volume = sitk.GetArrayFromImage(pd_pb_image).astype(dtype=np.uint8)
                    pd_oar1_image = resample_image(pd_oar1_image, size=new_size, spacing=new_spacing, origin=new_origin)
                    pd_oar1_volume = sitk.GetArrayFromImage(pd_oar1_image).astype(dtype=np.uint8)
                    pd_oar2_image = resample_image(pd_oar2_image, size=new_size, spacing=new_spacing, origin=new_origin)
                    pd_oar2_volume = sitk.GetArrayFromImage(pd_oar2_image).astype(dtype=np.uint8)
                    
                    # output results to file if needed
                    if output_file:
                        write_image(pd_pb_image, output_dir + cur_casename + '-pb.nii.gz')
                        write_image(pd_oar1_image, output_dir + cur_casename + '-oar1.nii.gz')
                        write_image(pd_oar2_image, output_dir + cur_casename + '-oar2.nii.gz')
                                        
                    # calculate case-wise DSC
                    gt_pb_volume = np.reshape(gt_pb_volume, -1)
                    pd_pb_volume = np.reshape(pd_pb_volume, -1)
                    pb_dice_dict[cur_casename] = (np.sum(pd_pb_volume * gt_pb_volume) * 2 + 1) / (np.sum(pd_pb_volume * pd_pb_volume + gt_pb_volume * gt_pb_volume) + 1)
                    gt_oar_volume = np.reshape(gt_oar_volume, -1)
                    gt_oar1_volume = np.zeros(gt_oar_volume.shape, dtype=np.uint8)
                    gt_oar1_volume[gt_oar_volume == 1] = 1
                    gt_oar2_volume = np.zeros(gt_oar_volume.shape, dtype=np.uint8)
                    gt_oar2_volume[gt_oar_volume == 2] = 1
                    pd_oar1_volume = np.reshape(pd_oar1_volume, -1)
                    pd_oar2_volume = np.reshape(pd_oar2_volume, -1)
                    oar1_dice_dict[cur_casename] = (np.sum(pd_oar1_volume * gt_oar1_volume) * 2 + 1) / (np.sum(pd_oar1_volume * pd_oar1_volume + gt_oar1_volume * gt_oar1_volume) + 1)
                    oar2_dice_dict[cur_casename] = (np.sum(pd_oar2_volume * gt_oar2_volume) * 2 + 1) / (np.sum(pd_oar2_volume * pd_oar2_volume + gt_oar2_volume * gt_oar2_volume) + 1)

                    print('DSC for case {:s} [PB, bladder, rectum]: {:6.3f}%, {:6.3f}%, {:6.3f}%'.format(cur_casename, pb_dice_dict[cur_casename] * 100.0, oar1_dice_dict[cur_casename] * 100.0, oar2_dice_dict[cur_casename] * 100.0))

                # create a new volume
                if not (batch_id == len(dataloader)-1 and i == batch_size-1):
                    gt_pb_image = read_image(gt_pb_dir + casename)
                    slice_num = gt_pb_image['size'][2]
                    pd_pb_volume = np.zeros([slice_num, pd_pb_array.shape[1], pd_pb_array.shape[2]], dtype=np.uint8)
                    pd_pb_volume[slice_id,:] = pd_pb_array
                    gt_oar_image = read_image(gt_oar_dir + casename)
                    slice_num = gt_oar_image['size'][2]
                    pd_oar1_volume = np.zeros([slice_num, pd_oar1_array.shape[1], pd_oar1_array.shape[2]], dtype=np.uint8)
                    pd_oar1_volume[slice_id,:] = pd_oar1_array
                    pd_oar2_volume = np.zeros([slice_num, pd_oar1_array.shape[1], pd_oar1_array.shape[2]], dtype=np.uint8)
                    pd_oar2_volume[slice_id,:] = pd_oar2_array
                    cur_casename = casename
                    cur_spacing = spacing[i,:]
                    cur_origin = origin[i,:]
            else:
                pd_pb_volume[slice_id,:] = pd_pb_array
                pd_oar1_volume[slice_id,:] = pd_oar1_array
                pd_oar2_volume[slice_id,:] = pd_oar2_array

    val_loss = val_loss / val_sample_num

    return pb_dice_dict, oar1_dice_dict, oar2_dice_dict, val_loss


if __name__ == '__main__':
    step1_model = torch.load('/home/omnisky/data1/Experiments/DyAdaptNet_trained_models/model_20210807152922/fold-1_epoch_96.pth.tar')['model_state_dict']
    print("Model's state_dict:")
    for k,v in step1_model.items():
        print(k, "\t", v.size())