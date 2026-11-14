# Dataset Description

## 1. Overview

The project uses pre-extracted feature representations stored as PyTorch (`.pt`) files. These files serve as input for the dimensionality-reduction and clustering experiments.

The dataset consists of **1,696 feature files**, organized into three class-specific folders.

## 2. Dataset Structure

The dataset follows the directory structure below:

```text
data/
├── T00B/    # 714 .pt files
├── T100/    # 486 .pt files
└── T109/    # 496 .pt files
```

## 3. File Format

- **Format**: PyTorch serialized files (`.pt`)
- **Approximate file size**: 5 KB per file
- **Content**: Pre-extracted feature representations
- **Organization**: Files are grouped into folders according to their respective class labels.

The feature files are loaded using PyTorch and combined to form the feature dataset used in the experiments.

## 4. Data Availability

The original `.pt` files are not included in this repository. The directory structure described above is provided to document the expected input organization.

To reproduce the experiments, the corresponding feature files must be placed in the respective class folders.

Important: The .pt files contain serialized PyTorch objects. Only load files from trusted sources.