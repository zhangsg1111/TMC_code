# HSFL ResNet-CIFAR10 Experiments

This repository contains Python experiment scripts for ResNet-based split/federated learning on CIFAR-10.

The code corresponds to:

S. Zhang, W. Wu, L. Song and X. Shen, "Efficient Model Training in Edge Networks With Hierarchical Split Learning," in IEEE Transactions on Mobile Computing, vol. 24, no. 10, pp. 10214-10229, Oct. 2025, doi: 10.1109/TMC.2025.3569407.

## Files

| File | Description |
|---|---|
| `hsfl_resnet_cifar10_noniid.py` | HSFL, the proposed method. |
| `baseline_cpsl_resnet_cifar10_noniid.py` | CPSL baseline. |
| `baseline_sfl_resnet_cifar10_noniid.py` | SFL baseline. |
| `baseline_sl_resnet_cifar10_noniid.py` | Split learning baseline. |
| `prepare_initial_models.py` | Generates the initial client/server model weights required by the experiments. |

Only the Non-IID setting is kept in this public version. The IID scripts were omitted because they mainly differ in the data partition setting.

## What Is Not Included

Datasets, trained models, initial model weights, logs, Excel results, and figures are not included.

The experiment scripts use `torchvision.datasets.CIFAR10(..., download=True)`, so CIFAR-10 will be downloaded automatically on first run.

The scripts expect initial model weights under:

```text
base_model/
```

Generate them locally with:

```bash
python prepare_initial_models.py
```

This creates files such as:

```text
base_model/base_client_model_state01.pth
base_model/base_server_model_state01.pth
...
base_model/base_client_model_state010.pth
base_model/base_server_model_state010.pth
```

Do not upload `base_model/`, downloaded datasets, or generated result files to GitHub.

## Environment

Install the main dependencies:

```bash
pip install torch torchvision numpy pandas openpyxl
```

Use a CUDA-enabled PyTorch installation if GPU acceleration is needed.

## How to Run

Run from the repository root.

First prepare the initial model weights:

```bash
python prepare_initial_models.py
```

Run HSFL:

```bash
python hsfl_resnet_cifar10_noniid.py
```

Run baselines:

```bash
python baseline_cpsl_resnet_cifar10_noniid.py
python baseline_sfl_resnet_cifar10_noniid.py
python baseline_sl_resnet_cifar10_noniid.py
```

Each training script repeats experiments for random seeds 1 to 10 and saves accuracy curves as Excel files in the current working directory.

## Notes

- CIFAR-10 is downloaded by torchvision into the local `dataset_mnist/` path used by the original scripts. The folder name is historical; the actual dataset is CIFAR-10.
- HSFL is the proposed method; CPSL, SFL, and SL are comparison methods.
- Generated `.xlsx`, `.pth`, dataset folders, and result folders should be ignored when uploading the code.

