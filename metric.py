import os
import SimpleITK as sitk
import itk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ImageType = itk.Image[itk.SS, 3]
LabelType = itk.Image[itk.SS, 3]

def keep_largest_component(image):
    filter_1 = itk.ConnectedComponentImageFilter[LabelType, LabelType].New()
    filter_1.SetInput(image)
    filter_1.SetBackgroundValue(0)
    filter_1.Update()

    filter_2 = itk.LabelShapeKeepNObjectsImageFilter[LabelType].New()
    filter_2.SetInput(filter_1.GetOutput())
    filter_2.SetBackgroundValue(0)
    filter_2.SetNumberOfObjects(1)
    filter_2.Update()
    output_image = filter_2.GetOutput()

    arr = img2arr(output_image)
    output_arr = np.zeros(arr.shape, dtype=np.int16)
    output_arr[arr > 0] = 1
    output_image = arr2img(output_arr, output_image)

    return output_image

def get_image(filename):
    reader = itk.ImageFileReader[LabelType].New()
    reader.SetFileName(filename)
    reader.Update()
    image = reader.GetOutput()

    return image

def img2arr(image):
    image_array = itk.GetArrayFromImage(image)
    image_array = image_array.astype(dtype=np.uint8)
    return image_array

def arr2img(array, ref_image):
    image = itk.GetImageFromArray(array)
    image.SetSpacing(ref_image.GetSpacing())
    image.SetOrigin(ref_image.GetOrigin())
    image.SetDirection(ref_image.GetDirection())
    image.SetLargestPossibleRegion(ref_image.GetLargestPossibleRegion())
    return image

def dice(pd, gt):
    y = (np.sum(pd * gt) * 2 + 1) / (np.sum(pd * pd + gt * gt) + 1)
    return y

def asd(a, b):
    FloatImageType = itk.Image[itk.F, 3]

    a_array = itk.GetArrayFromImage(a)
    a_array = a_array.astype(dtype=np.float32)
    a_float = itk.GetImageFromArray(a_array)
    a_float.SetSpacing(a.GetSpacing())
    a_float.SetOrigin(a.GetOrigin())
    a_float.SetDirection(a.GetDirection())
    a_float.SetLargestPossibleRegion(a.GetLargestPossibleRegion())

    filter1 = itk.SignedMaurerDistanceMapImageFilter[FloatImageType, FloatImageType].New()
    filter1.SetInput(a_float)
    filter1.SetUseImageSpacing(True)
    filter1.SetSquaredDistance(False)
    filter1.Update()
    a_dist = filter1.GetOutput()

    a_dist = itk.GetArrayFromImage(a_dist)
    a_dist = np.abs(a_dist)
    a_edge = np.zeros(a_dist.shape, a_dist.dtype)
    a_edge[a_dist == 0] = 1
    a_num = np.sum(a_edge)
    
    b_array = itk.GetArrayFromImage(b)
    b_array = b_array.astype(dtype=np.float32)
    b_float = itk.GetImageFromArray(b_array)
    b_float.SetSpacing(b.GetSpacing())
    b_float.SetOrigin(b.GetOrigin())
    b_float.SetDirection(b.GetDirection())
    b_float.SetLargestPossibleRegion(b.GetLargestPossibleRegion())

    filter2 = itk.SignedMaurerDistanceMapImageFilter[FloatImageType, FloatImageType].New()
    filter2.SetInput(b_float)
    filter2.SetUseImageSpacing(True)
    filter2.SetSquaredDistance(False)
    filter2.Update()
    b_dist = filter2.GetOutput()

    b_dist = itk.GetArrayFromImage(b_dist)
    b_dist = np.abs(b_dist)
    b_edge = np.zeros(b_dist.shape, b_dist.dtype)
    b_edge[b_dist == 0] = 1
    b_num = np.sum(b_edge)

    a_dist[b_edge == 0] = 0.0
    b_dist[a_edge == 0] = 0.0
    asd = (np.sum(a_dist) + np.sum(b_dist)) / (a_num + b_num)

    return asd

dir_img_volume = '/home/omnisky/data1/DATA/Prostate_Bed/IMG/' # where you store original images ()
dir_pb_volume = '/home/omnisky/data1/DATA/Prostate_Bed/PB/' # where you store ground-truth segmentation (prostate bed)
dir_oar_volume = '/home/omnisky/data1/DATA/Prostate_Bed/OAR/' # where you store ground-truth segmentation (bladder and rectum)
dir_models = '/home/omnisky/data1/Experiments/DyAdaptNet_trained_models/' # where you store trained models and predicted segmentation

keep_largest = True # whether keep the largest component before metric calculation

model_names = [
    'saved/' # specific folder name of the trained model
    ]

exclude_cases = [ # cases excluded in the metric calculation (we exclude 5 cases with severe metal artifact caused by artificial femoral heads)
]

for model_name in model_names:
    print(model_name)
    dir_results = dir_models + model_name + 'results/'
    output_filename = dir_models + model_name + 'metric_asd_corrected.txt'

    with open(output_filename, 'w') as output_file:
        pb_dice_dict = {}
        pb_asd_dict = {}
        oar1_dice_dict = {}
        oar1_asd_dict = {}
        oar2_dice_dict = {}
        oar2_asd_dict = {}

        file_list = os.listdir(dir_img_volume)
        for id, casename in enumerate(file_list):
            pd_pb_filename = dir_results + casename + '-pb.nii.gz'
            pd_oar1_filename = dir_results + casename + '-oar1.nii.gz'
            pd_oar2_filename = dir_results + casename + '-oar2.nii.gz'
            output_line = "{id:>03d}/{len:3d} {case:<12s}".format(id=id+1, len=len(file_list), case=casename)

            if casename in exclude_cases:
                output_line += " excluded"
                print(output_line)
                continue
            
            # prostate bed
            if os.path.exists(pd_pb_filename):
                # ground-truth
                gt_pb_image = get_image(dir_pb_volume + casename)
                gt_pb_volume = img2arr(gt_pb_image)
                gt_pb_volume = np.reshape(gt_pb_volume, -1)
                # predict
                pd_pb_image = get_image(pd_pb_filename)
                if keep_largest:
                    pd_pb_image = keep_largest_component(pd_pb_image)
                pd_pb_volume = img2arr(pd_pb_image)
                pd_pb_volume = np.reshape(pd_pb_volume, -1)
                # dice
                pb_dice_dict[casename] = dice(pd_pb_volume, gt_pb_volume)
                # average symmetric surface distance
                pb_asd_dict[casename] = asd(pd_pb_image, gt_pb_image)
                
                output_line += " {dice:12.3f}% {asd:8.3f}mm".format(dice=pb_dice_dict[casename] * 100.0, asd=pb_asd_dict[casename])

            # bladder & rectum
            if os.path.exists(pd_oar1_filename):
                # ground-truth
                gt_oar_image = get_image(dir_oar_volume + casename)
                gt_oar_volume = img2arr(gt_oar_image)
                #gt_oar_volume = np.reshape(gt_oar_volume, -1)
                gt_oar1_volume = np.zeros(gt_oar_volume.shape, dtype=np.int16)
                gt_oar1_volume[gt_oar_volume == 1] = 1
                gt_oar1_image = arr2img(gt_oar1_volume, gt_oar_image)
                gt_oar1_volume = np.reshape(gt_oar1_volume, -1)
                gt_oar2_volume = np.zeros(gt_oar_volume.shape, dtype=np.int16)
                gt_oar2_volume[gt_oar_volume == 2] = 1
                gt_oar2_image = arr2img(gt_oar2_volume, gt_oar_image)
                gt_oar2_volume = np.reshape(gt_oar2_volume, -1)
                # predict
                pd_oar1_image = get_image(pd_oar1_filename)
                if keep_largest:
                    pd_oar1_image = keep_largest_component(pd_oar1_image)
                pd_oar1_volume = img2arr(pd_oar1_image)
                pd_oar1_volume = np.reshape(pd_oar1_volume, -1)
                pd_oar2_image = get_image(pd_oar2_filename)
                if keep_largest:
                    pd_oar2_image = keep_largest_component(pd_oar2_image)
                pd_oar2_volume = img2arr(pd_oar2_image)
                pd_oar2_volume = np.reshape(pd_oar2_volume, -1)
                # dice
                oar1_dice_dict[casename] = dice(pd_oar1_volume, gt_oar1_volume)
                oar2_dice_dict[casename] = dice(pd_oar2_volume, gt_oar2_volume)
                # average symmetric surface distance
                oar1_asd_dict[casename] = asd(pd_oar1_image, gt_oar1_image)
                oar2_asd_dict[casename] = asd(pd_oar2_image, gt_oar2_image)
                
                output_line += " {dice1:12.3f}% {asd1:8.3f}mm {dice2:12.3f}% {asd2:8.3f}mm".format(
                    dice1=oar1_dice_dict[casename] * 100.0, asd1=oar1_asd_dict[casename], 
                    dice2=oar2_dice_dict[casename] * 100.0, asd2=oar2_asd_dict[casename])

            print(output_line)
            output_file.write(output_line+'\n')

        output_line = '------------------------------------------------------------------------------------------'
        print(output_line)
        output_file.write(output_line+'\n')

        pb_dice = np.array(list(pb_dice_dict.values()), dtype=float)
        oar1_dice = np.array(list(oar1_dice_dict.values()), dtype=float)
        oar2_dice = np.array(list(oar2_dice_dict.values()), dtype=float)
        case_num = pb_dice.size if pb_dice.size > 0 else oar1_dice.size
        output_line = "Global DSC ({num:d}):\nPB: {pb_dsc_mean:.2f}({pb_dsc_std:.2f})%\tBladder: {oar1_dsc_mean:.2f}({oar1_dsc_std:.2f})%\tRectum: {oar2_dsc_mean:.2f}({oar2_dsc_std:.2f})%".format(
                    num=case_num,
                    pb_dsc_mean=pb_dice.mean()*100.0, pb_dsc_std=pb_dice.std(ddof=1)*100.0, 
                    oar1_dsc_mean=oar1_dice.mean()*100.0, oar1_dsc_std=oar1_dice.std(ddof=1)*100.0, 
                    oar2_dsc_mean=oar2_dice.mean()*100.0, oar2_dsc_std=oar2_dice.std(ddof=1)*100.0)
        print(output_line)
        output_file.write(output_line+'\n')    
        
        pb_asd = np.array(list(pb_asd_dict.values()), dtype=float)
        oar1_asd = np.array(list(oar1_asd_dict.values()), dtype=float)
        oar2_asd = np.array(list(oar2_asd_dict.values()), dtype=float)
        output_line = "Global ASD ({num:d}):\nPB: {pb_asd_mean:.2f}({pb_asd_std:.2f})mm\tBladder: {oar1_asd_mean:.2f}({oar1_asd_std:.2f})mm\tRectum: {oar2_asd_mean:.2f}({oar2_asd_std:.2f})mm".format(
                    num=case_num,
                    pb_asd_mean=pb_asd.mean(), pb_asd_std=pb_asd.std(ddof=1), 
                    oar1_asd_mean=oar1_asd.mean(), oar1_asd_std=oar1_asd.std(ddof=1), 
                    oar2_asd_mean=oar2_asd.mean(), oar2_asd_std=oar2_asd.std(ddof=1))
        print(output_line)
        output_file.write(output_line+'\n')
