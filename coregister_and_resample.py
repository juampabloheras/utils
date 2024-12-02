# Postprocess cyclegan
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import os
import SimpleITK as sitk
import torch
import numpy as np
import json


def command_iteration(method):
    """Callback invoked when the optimization has an iteration"""
    if method.GetOptimizerIteration() == 0:
        print("Estimated Scales: ", method.GetOptimizerScales())
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():7.5f} "
        + f": {method.GetOptimizerPosition()}"
    )


def coregister_and_resample(
    fixed_image, moving_image, save_path=None, remove_background=False, is_mask=False
):
    """
    Coregisters and resamples the moving image to the fixed image's space and desired target properties.

    Parameters:
    - fixed_image (SimpleITK.Image): The fixed reference image.
    - moving_image (SimpleITK.Image): The moving image to be aligned to the fixed image.
    - target_size (tuple): The desired output size (Width, Height, Depth).
    - target_spacing (tuple): The desired spacing (x, y, z).
    - target_origin (tuple): The desired origin (x, y, z).
    - target_direction (tuple): The desired direction (3x3 matrix flattened to a tuple).
    - is_mask (bool): If True, uses nearest neighbor interpolation (for masks). If False, uses linear interpolation.

    Returns:
    - torch.Tensor: Resampled image or mask as a PyTorch tensor with shape (Width, Height, Depth).
    """

    target_size, target_spacing, target_origin, target_direction = (
        fixed_image.GetSize(),
        fixed_image.GetSpacing(),
        fixed_image.GetOrigin(),
        fixed_image.GetDirection(),
    )
    print(
        "target_size, target_spacing, target_origin, target_direction ",
        target_size,
        target_spacing,
        target_origin,
        target_direction,
    )

    # Perform image registration using the correlation metric
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()

    # Use regular step gradient descent optimizer
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-5,
        numberOfIterations=500,
        gradientMagnitudeTolerance=1e-8,
    )
    R.SetOptimizerScalesFromIndexShift()

    # Initialize transformation with center of mass (3D transform for 3D images)
    tx = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, sitk.Similarity3DTransform()
    )
    R.SetInitialTransform(tx)

    # Use linear interpolator
    R.SetInterpolator(sitk.sitkLinear)

    # Add command to track optimization progress
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    # Execute registration and obtain transformation
    outTx = R.Execute(fixed_image, moving_image)

    print("-------")
    print(outTx)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")

    # Resample the moving image using the obtained transformation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(
        sitk.sitkLinear if not is_mask else sitk.sitkNearestNeighbor
    )
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)

    resampled_image = resampler.Execute(moving_image)

    if remove_background:
        fixed_mask = fixed_image != 0
        resampled_image = sitk.Mask(resampled_image, fixed_mask)

    print(
        "resampled_image.GetSize(), resampled_image.GetSpacing(), resampled_image.GetOrigin(), resampled_image.GetDirection()",
        resampled_image.GetSize(),
        resampled_image.GetSpacing(),
        resampled_image.GetOrigin(),
        resampled_image.GetDirection(),
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sitk.WriteImage(resampled_image, save_path)
        print(f"Saved image to: {save_path}")

    # Convert to NumPy array and transpose to (Width, Height, Depth)
    np_array = sitk.GetArrayFromImage(resampled_image)
    np_array = np_array.astype(np.float32) if not is_mask else np_array.astype(np.int64)
    np_array = np.transpose(np_array, (2, 1, 0))

    return torch.tensor(np_array)


def fixed_img_from_moving(
    full_path_to_moving_image="inference/train_result_148.nii",
    path_to_names_conversion="train_names_conversion.json",
):
    subj_id = full_path_to_moving_image.split("/")[-1].split("_")[-1].split(".")[0]
    with open(path_to_names_conversion) as json_data:
        names_conversion = json.load(json_data)
    return names_conversion[f"{subj_id}"]


if __name__ == "__main__":
    # Change these
    path_to_dwi = "/gscratch/kurtlab/brats2024/data/isles/dwi_cta"
    path_to_names_conversion = "/gscratch/kurtlab/brats2024/repos/isles2stage/cyclegan/3D-CycleGan-Pytorch-MedImaging/train_names_conversion.json"
    path_to_inference = "/gscratch/kurtlab/brats2024/repos/isles2stage/cyclegan/3D-CycleGan-Pytorch-MedImaging/Data_folder2/test/images/inference"
    path_to_save_coregistered = "/gscratch/kurtlab/brats2024/data/isles/synthDWI/v1"
    remove_background = True  

    # Run registration and resampling
    subjects_list = sorted(
        [
            filename.split("_")[-1].split(".")[0]
            for filename in os.listdir(path_to_inference)
        ]
    )
    print(f"Number of subjects: {len(subjects_list)}")
    for subj in subjects_list:

        path_to_moving_image = os.path.join(
            path_to_inference, f"train_result_{subj}.nii"
        )
        path_to_subject = fixed_img_from_moving(
            path_to_moving_image, path_to_names_conversion
        )
        original_id = path_to_subject.split("/")[-1].split("_")[0]

        path_to_fixed_image = os.path.join(path_to_dwi, path_to_subject)

        fixed_image = sitk.ReadImage(path_to_fixed_image, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(path_to_moving_image, sitk.sitkFloat32)

        save_path = os.path.join(path_to_save_coregistered, f"{original_id}.nii.gz")
        if not os.path.exists(save_path):
            resampled_image_tensor = coregister_and_resample(
                fixed_image,
                moving_image,
                save_path=save_path,
                remove_background=remove_background,
            )
        else:
            print(f'Save path {save_path} exists.. Skipping this subject!')
