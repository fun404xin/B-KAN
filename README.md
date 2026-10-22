# B-KAN
1.Environment


Recommended Environment
The code is implemented with Python and PyTorch.
A virtual environment is recommended:

python -m venv .venv

Linux / macOS

source .venv/bin/activate

Windows

.venv\Scripts\activate

Then upgrade pip:

python -m pip install --upgrade pip

2.Install Dependencies


Install the required Python packages using:

pip install -r requirements.txt

python -m pip install --upgrade pip

3.Install Dependencies


Install the required Python packages using:

pip install -r requirements.txt

4.Dataset Availability


The experiments in this study were conducted using the publicly available XJTU-SY and FEMTO-ST (PRONOSTIA / PHM 2012) bearing datasets. Due to the large size of the raw data files, these datasets are not included in this repository. Users can download them from the following sources.
XJTU-SY Bearing Dataset

Dataset webpage: https://biaowang.tech/xjtu-sy-bearing-datasets/
Google Drive: https://drive.google.com/open?id=1_ycmG46PARiykt82ShfnFfyQsaXv3_VK

Reference: Wang, B., Lei, Y., Li, N., & Li, N. (2020). A Hybrid Prognostics Approach for Estimating Remaining Useful Life of Rolling Element Bearings. IEEE Transactions on Reliability, 69(1), 401–412. https://doi.org/10.1109/TR.2018.2882682

FEMTO-ST Bearing Dataset

NASA Prognostics Data Repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
Direct download (ZIP): https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip

Reference: Nectoux, P., Gouriveau, R., Medjaher, K., Ramasso, E., Morello, B., Zerhouni, N., & Varnier, C. (2012). PRONOSTIA: An Experimental Platform for Bearings Accelerated Life Test. Proceedings of the IEEE International Conference on Prognostics and Health Management (PHM).

5.Quick Start

After installing the dependencies and preparing the dataset, run:

python train.py

The default configuration uses:

Epochs: 100
Learning rate: 0.01
Batch size: 32
Random seed: 42
Device: CUDA if available, otherwise CPU

The script automatically selects:

torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

6.Training B-KAN

To train the proposed B-KAN model:

python train.py --model b-kan
