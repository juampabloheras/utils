import os
import torch
import nibabel as nib
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import glob


def linear_windowing_torch(image, W_center, W_width, min_output=0, max_output=255):
    L = W_center - W_width / 2
    U = W_center + W_width / 2

    windowed_image = torch.clamp((image - L) / (U - L), 0, 1)
    windowed_image = windowed_image * (max_output - min_output) + min_output
    return windowed_image

def save_tensor_to_nii(tensor, filename, metadata):
    np_array = tensor.numpy()
    np_array = np_array.transpose(2, 1, 0)  # np and sitk have different conventions
    sitk_image = sitk.GetImageFromArray(np_array)

    sitk_image.SetOrigin(metadata["origin"])
    sitk_image.SetSpacing(metadata["spacing"])
    sitk_image.SetDirection(metadata["direction"])

    print("sitk size: ", sitk_image.GetSize())
    sitk.WriteImage(sitk_image, filename)


def extract_metadata(image):
    """Extract metadata from an image."""
    return {
        "origin": image.GetOrigin(),
        "spacing": image.GetSpacing(),
        "direction": image.GetDirection(),
    }


def load_tensor(path):
    return torch.tensor(
        nib.load(path).get_fdata(), dtype=torch.float32
    ), extract_metadata(sitk.ReadImage(path))


def save_windowed_ct(
    data_dir="/gscratch/kurtlab/brats2024/data/isles/ISLES_and_MRI_TR",
    save_dir="/gscratch/kurtlab/brats2024/data/isles/windowed_ncct",
    search_str="*stripped.nii.gz",
):
    def process_subject(subject):
        try:
            im_path = glob.glob(os.path.join(data_dir, subject, f"{search_str}"))[0]
        except IndexError:
            print(
                f"No file matching '{search_str}' found for subject {subject}. Skipping..."
            )
            return

        try:
            img, metadata = load_tensor(
                im_path
            )  # torch.tensor(nib.load(im_path).get_fdata())

            windowing_params = {
                "anat": {"W_center": 35, "W_width": 65},
                "blood": {"W_center": 90, "W_width": 180},
                "GW": {"W_center": 30, "W_width": 30},
            }

            # Apply windowing
            windowed_images = {
                roi: linear_windowing_torch(img, **params)
                for roi, params in windowing_params.items()
            }

            # Save results
            for roi, windowed_img in windowed_images.items():
                dir_name = os.path.join(save_dir, f"ncct_{roi}_window")
                os.makedirs(dir_name, exist_ok=True)
                save_tensor_to_nii(
                    windowed_img, os.path.join(dir_name, f"{subject}.nii.gz"), metadata
                )
            print(f"Processed subject {subject} successfully.")

        except Exception as e:
            print(f"Error processing subject {subject}: {e}", flush=True)

    subjects = sorted(
        [s for s in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, s))]
    )
    # subjects = ['0136']
    for subject in subjects:
        print("Subject: ", subject)
        process_subject(subject)
    # with ThreadPoolExecutor() as executor:
    #     list(executor.map(process_subject, subjects))


def get_files_with_string(directory, string):
    # Function to recursively find files in a directory with a specified string in their name, and output a list with all filenames.
    search_pattern = os.path.join(directory, "**", f"*{string}*")
    file_paths = glob.glob(search_pattern, recursive=True)
    file_paths = sorted([path for path in file_paths if os.path.isfile(path)])
    return file_paths


def print_resolutions():

    # Change these:
    data_dir = "/gscratch/kurtlab/brats2024/data/isles/ISLES_and_MRI_TR"  # Dir to search recursively for files with string in their name.
    string = "dwi_reg"

    # data_dir = '/gscratch/kurtlab/brats2024/repos/isles2stage/cyclegan/3D-CycleGan-Pytorch-MedImaging/Data_folder_mean_res/train/images'
    # string = '.nii'

    # No need to change anything below here
    filepaths = get_files_with_string(data_dir, string)
    resolutions_list = []
    for file in filepaths:
        img = sitk.ReadImage(file, sitk.sitkFloat32)
        resolution = img.GetSpacing()
        # print(f'Loaded {file}, resolution: {resolution}') # uncomment to see individual file resolutions
        resolutions_list.append(resolution)

    averages = [sum(values) / len(values) for values in zip(*resolutions_list)]
    maxes = [max(values) for values in zip(*resolutions_list)]
    mins = [min(values) for values in zip(*resolutions_list)]

    print(f"Average for each position: {averages}")
    print(f"Max for each position: {maxes}")
    print(f"Min for each position: {mins}")


def rename_files():
    import os
    
    directory = '/gscratch/kurtlab/brats2024/data/isles/dwi_cta/ncct_anat_window'

    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.nii.gz'):  # Check if the file is a .nii.gz file
            old_file = os.path.join(directory, filename)

            name = filename.split('.')[0]
            new_name = f"{name}_ncct_anat_window.nii.gz"
            new_file = os.path.join(directory, new_name)

            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed '{old_file}' to '{new_file}'")

##############____________________________________________________________________________##########
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a specified function with optional arguments."
    )
    parser.add_argument("function_name", type=str, help="The function to run")
    parser.add_argument(
        "additional_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for the function, passed as strings to the function",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """
    Example usage of a function in this script:
        python3 train.py train_seg_synth arg1 arg2
        python3 train.py eval arg1 arg2 arg3
        python3 train.py get_wandb_config
    NOTE: all args are passed as strings to the function called.
    """

    args = parse_args()

    # Validate function_name argument
    function_name = args.function_name

    if function_name in globals() and callable(globals()[function_name]):
        # Call the function by name, pass additional arguments as a list
        print(
            f"(@window_images.py) Running function: {function_name} with arguments: {args.additional_args}\n"
        )
        globals()[function_name](*args.additional_args)
    else:
        print(f"Function '{function_name}' not found.")
        print(
            f"Available functions: {[fn for fn in globals() if callable(globals()[fn])]}"
        )
