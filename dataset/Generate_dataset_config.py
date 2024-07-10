import os
from tqdm import tqdm


def generate_dataset_list(noisy_dir, clean_dir, output_file):
    noisy_files = os.listdir(noisy_dir)
    dataset_list = []

    # Initialize tqdm progress bar
    with tqdm(total=len(noisy_files), desc="Processing files", unit="file") as pbar:
        for noisy_file in noisy_files:
            # Extract the base name without the noise level and file extension
            base_name = '_'.join(noisy_file.split('_')[:2])

            # Find the corresponding clean speech file
            found = False
            for root, dirs, files in os.walk(clean_dir):
                for file in files:
                    if base_name in file:
                        noisy_path = os.path.join(noisy_dir, noisy_file)
                        clean_path = os.path.join(root, file)
                        dataset_list.append(f"{noisy_path} {clean_path}")
                        found = True
                        break
                if found:
                    break

            # Update progress bar
            pbar.update(1)

    # Write the dataset list to the specified output file
    with open(output_file, 'w') as f:
        for item in dataset_list:
            f.write(f"{item}\n")


import os

def delete_non_wav_files(directory):
    """
    Delete all files that are not in WAV format from the given directory and its subdirectories.
    Prints the path and name of each deleted file.

    :param directory: Path to the directory to scan.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


if __name__ == "__main__":
    directory_path = "/home/SE_dataset/VCTK_Demand_ch1/Test_noisy"
    if os.path.exists(directory_path):
        delete_non_wav_files(directory_path)
    else:
        print("The specified directory does not exist.")


# if __name__ == "__main__":
#     noisy_dir = "/home/SE_dataset/VCTK_Demand_ch1/Test_noisy"
#     clean_dir = "/home/SE_dataset/VCTK_Demand_ch1/Test_clean"
#     output_file = "/home/SE_dataset/valid_dataset.txt"
#
#     generate_dataset_list(noisy_dir, clean_dir, output_file)
#     print(f"Dataset list has been generated and saved to {output_file}")
