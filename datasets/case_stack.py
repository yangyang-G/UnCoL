import os
from glob import glob

import h5py
import numpy as np
from torch.utils.data import Dataset


LABEL_CASE_MAPPING = {
    "ACDC": {"5%": 3, "10%": 7, "20%": 14, "100%": 70},
    "Prostate": {"5%": 2, "10%": 4, "20%": 7, "100%": 35},
    "Hippocampus": {"5%": 8, "10%": 16, "20%": 31, "100%": 156},
    "Vertebral": {"5%": 5, "10%": 10, "20%": 21, "100%": 106},
    "HepaticVessel": {"5%": 9, "10%": 18, "20%": 36, "100%": 181},
    "ATLAS": {"5%": 2, "10%": 4, "20%": 7, "100%": 36},
    "WORD": {"5%": 4, "10%": 7, "20%": 14, "100%": 72},
    "BTCV": {"5%": 1, "10%": 2, "20%": 4, "100%": 18},
}


def parse_patient_id(name):
    stem = os.path.basename(name).replace(".h5", "")
    if "_frame" in stem:
        return stem.split("_frame", 1)[0]
    if "_slice_" in stem:
        return stem.split("_slice_", 1)[0]
    return stem


def infer_dataset_name(*values):
    joined = " ".join(str(value) for value in values if value)
    for dataset_name in LABEL_CASE_MAPPING:
        if dataset_name in joined:
            return dataset_name
    return None


def labeled_cases_for_ratio(dataset_name, label_ratio):
    if not label_ratio:
        return None
    if dataset_name not in LABEL_CASE_MAPPING:
        raise ValueError("Unsupported dataset for label_ratio: {}".format(dataset_name))
    if label_ratio not in LABEL_CASE_MAPPING[dataset_name]:
        raise ValueError(
            "Unsupported label_ratio '{}' for {}. Choose from {}".format(
                label_ratio, dataset_name, sorted(LABEL_CASE_MAPPING[dataset_name])
            )
        )
    return LABEL_CASE_MAPPING[dataset_name][label_ratio]


def apply_label_ratio(args):
    label_ratio = getattr(args, "label_ratio", None)
    if not label_ratio:
        return args
    dataset_name = infer_dataset_name(
        getattr(args, "dataset", None),
        getattr(args, "data_dir", None),
    )
    args.labeled_num = labeled_cases_for_ratio(dataset_name, label_ratio)
    return args


class CaseStackDataset(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
        fold_num=0,
    ):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.sample_list = []
        self.slice_list = []
        self.patient_order = []

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For CTAugment, provide both weak and strong augmentation policies"

        entries = self._read_split_entries(fold_num)
        self._build_slice_index(entries)

        if num is not None and self.split == "train":
            self._keep_first_n_patients(num)

        self.sample_list = [
            "{}_slice_{}".format(item["case"], item["slice_idx"]) for item in self.slice_list
        ]
        print(
            "total {} slices from {} patients".format(
                len(self.slice_list), len({item["patient_id"] for item in self.slice_list})
            )
        )

    def _read_split_entries(self, fold_num):
        split_files = [
            os.path.join(
                self._base_dir,
                "datalist",
                "fold_{}".format(fold_num),
                "{}.txt".format(self.split),
            ),
            os.path.join(
                self._base_dir,
                "slicelist",
                "fold_{}".format(fold_num),
                "{}.txt".format(self.split),
            ),
            os.path.join(self._base_dir, "{}.list".format(self.split)),
            os.path.join(self._base_dir, "{}.txt".format(self.split)),
        ]
        for path in split_files:
            if os.path.exists(path):
                with open(path, "r") as f:
                    return [item.strip().replace(".h5", "") for item in f if item.strip()]
        raise FileNotFoundError(
            "Cannot find split file for '{}' under '{}'".format(self.split, self._base_dir)
        )

    def _resolve_entry_files(self, entry):
        exact_candidates = [
            os.path.join(self._base_dir, "data", "{}.h5".format(entry)),
            os.path.join(self._base_dir, "{}.h5".format(entry)),
        ]
        exact_files = [path for path in exact_candidates if os.path.exists(path)]
        if exact_files:
            return exact_files

        glob_patterns = [
            os.path.join(self._base_dir, "data", "{}_frame*.h5".format(entry)),
            os.path.join(self._base_dir, "{}_frame*.h5".format(entry)),
        ]
        files = []
        for pattern in glob_patterns:
            files.extend(glob(pattern))
        files = sorted(set(files))
        if files:
            return files

        raise FileNotFoundError(
            "Cannot find h5 file(s) for '{}' under '{}'".format(entry, self._base_dir)
        )

    def _build_slice_index(self, entries):
        seen_files = set()
        seen_patients = set()
        for entry in entries:
            for path in self._resolve_entry_files(entry):
                if path in seen_files:
                    continue
                seen_files.add(path)
                case = os.path.splitext(os.path.basename(path))[0]
                patient_id = parse_patient_id(case)
                if patient_id not in seen_patients:
                    self.patient_order.append(patient_id)
                    seen_patients.add(patient_id)
                with h5py.File(path, "r") as h5f:
                    image_shape = h5f["image"].shape
                num_slices = image_shape[0] if len(image_shape) > 2 else 1
                for slice_idx in range(num_slices):
                    self.slice_list.append(
                        {
                            "path": path,
                            "case": case,
                            "patient_id": patient_id,
                            "slice_idx": slice_idx,
                        }
                    )

    def _keep_first_n_patients(self, num_patients):
        labeled_patients = set(self.patient_order[:num_patients])
        self.slice_list = [
            item for item in self.slice_list if item["patient_id"] in labeled_patients
        ]
        self.patient_order = [
            patient_id for patient_id in self.patient_order if patient_id in labeled_patients
        ]

    def labeled_unlabeled_indices(self, labeled_num):
        labeled_patients = set(self.patient_order[:labeled_num])
        labeled_idxs = []
        unlabeled_idxs = []
        for idx, item in enumerate(self.slice_list):
            if item["patient_id"] in labeled_patients:
                labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
        return labeled_idxs, unlabeled_idxs

    def __len__(self):
        return len(self.slice_list)

    def __getitem__(self, idx):
        item = self.slice_list[idx]
        with h5py.File(item["path"], "r") as h5f:
            image_data = h5f["image"]
            label_data = h5f["label"]
            if len(image_data.shape) > 2:
                image = image_data[item["slice_idx"]]
                label = label_data[item["slice_idx"]]
            else:
                image = image_data[:]
                label = label_data[:]

        sample = {"image": np.asarray(image).squeeze(), "label": np.asarray(label).squeeze().astype(np.float32)}
        if self.transform and self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        sample["name"] = "{}_slice_{}".format(item["case"], item["slice_idx"])
        sample["case"] = item["case"]
        sample["patient_id"] = item["patient_id"]
        return sample
