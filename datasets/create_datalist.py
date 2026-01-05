import os
import glob
from sklearn.model_selection import KFold, train_test_split

def read_samples(file_path):
    file_list = glob.glob(file_path + '/*.h5')
    samples = [os.path.basename(file).split('_slice_')[0] for file in file_list]
    all_slices = [file.split('/')[-1][:-3] for file in file_list]
    unique_samples = list(set(samples))
    slice_dict = {}
    for sample in unique_samples:
        slice_dict[sample] = []

    for slice_file in all_slices:
        sample = slice_file.split('_slice_')[0]
        slice_dict[sample].append(slice_file)
    return unique_samples, slice_dict

def write_samples(file_path, samples):
    with open(file_path, 'w') as file:
        for sample in samples:
            file.write(f"{sample}\n")

def create_folds(samples, slice_dict, output_dir, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_index, test_index) in enumerate(kf.split(samples)):
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_samples = [samples[i] for i in train_index]
        test_samples = [samples[i] for i in test_index]
        
        # Split train_samples into train and validation sets
        train_samples, val_samples = train_test_split(train_samples, test_size=0.125, random_state=42)
        
        train_file = os.path.join(fold_dir, 'train.txt')
        val_file = os.path.join(fold_dir, 'val.txt')
        test_file = os.path.join(fold_dir, 'test.txt')
        
        train_slices = []
        val_slices = []
        test_slices = []

        for i in train_samples:
            train_slices += slice_dict[i]
        for i in val_samples:
            val_slices += slice_dict[i]
        for i in test_samples:
            test_slices += slice_dict[i]
        write_samples(train_file, train_slices)
        write_samples(val_file, val_slices)
        write_samples(test_file, test_slices)

def save_slices(file_path, samples):
    for sample in samples:
        slice_files = glob.glob(file_path + f'/{sample}_slice_*.h5')
        for slice_file in slice_files:
            new_name = os.path.basename(slice_file)
            new_path = os.path.join(file_path, new_name)
            os.rename(slice_file, new_path)

# Example usage
file_path = "FILE_PATH"
output_dir = "OUTPUT_PATH"

samples, slice_dict = read_samples(file_path)
create_folds(samples, slice_dict, output_dir)
