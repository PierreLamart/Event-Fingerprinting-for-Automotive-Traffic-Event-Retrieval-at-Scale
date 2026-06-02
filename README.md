# Event Fingerprinting

Identifying and analyzing traffic events in large-scale, unstructured video data from vehicle-mounted cameras is a significant challenge for enhancing advanced driver assistance systems (ADAS). In this paper, we present a model to extract pedestrian crossing feature with dimesionality reduction using space-filling curves (SFCs). We compare 4 conceptually different approaches: Optical flow, attention map, YOLO and 4 Vision Language Models (VLMs). Our result shows that VLMs have valuable detection potentiel but struggle with real-world data, where Optical flow and attention map methods perform better.

## Running the pipeline

### 1. Prerequisites

- [Conda](https://docs.anaconda.com/free/miniconda/index.html)
- [FFmpeg](https://ffmpeg.org/download.html)
- [Ollama](https://ollama.com/download)
- NVIDIA CUDA-enabled GPU (optional)

### 2. Create environment

```bash
conda create -n pyTED python=3.9
conda activate pyTED
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

### 3. Prepare data

Make sure the dataset follows the structure below.

```txt
data/
├── [dataset_name]/
│   ├── 1/             # folder for video 1
│   │   ├── file_1
|   |   ├── ....
│   │   ├── file_n
│   ├── ...
│   ├── n/             # folder for video n
```

Scripts for processing datasets into the correct structure are provided in `src/scripts`. The project currently supports the following datasets:

- [SMIRK](https://www.ai.se/en/labs/data-factory/datasets/smirk-dataset)
- [ZOD](https://www.zod.zenseact.com)

```bash
# Running the script to format the ZOD dataset, for example
python src/scripts/zod/process.py path/to/original/dataset data/zod --mode random --nr-videos 10
```

**Important**: Depending on the Conda environment, ffmpeg may not work. If you cannot process the datasets, deactivate the environment.

### 4. Optional: Enable GPU-accelerated Optical Flow (NVIDIA CUDA-enabled GPUs only)

To enable running the optical flow model on the GPU, compile opencv from source with the cudaoptflow module:

1. Create a new conda environment, `pyTED-cuda-cv`, using the instructions from [Step 2](#2-create-environment)
1. Uninstall the current version of opencv `pip uninstall opencv-python`
1. Follow [this guide](https://danielhavir.com/notes/install-opencv/) by Daniel Havir. Note:
    - Get the latest versions of opencv and opencv_contrib from the official repositories: [opencv](https://github.com/opencv/opencv/releases) and [opencv_contrib](https://github.com/opencv/opencv_contrib/tags)
    - Use the newly created environment `pyTED-cuda-cv` instead of `cv`
    - Replace references to python3.6 with python3.9
    - Ensure all the environment variables are correctly defined before running the cmake command. Example values:
        - `$python_exec: /home/elias/miniconda3/envs/pyTED-cuda-cv/bin/python`
        - `$include_dir: /home/elias/miniconda3/envs/pyTED-cuda-cv/include/python3.9`
        - `$library: /home/elias/miniconda3/envs/pyTED-cuda-cv/lib/libpython3.9.so`
        - `$default_exec: /home/elias/miniconda3/envs/pyTED-cuda-cv/bin/python3.9`
1. Test your installation with `python -c "import cv2; print('CUDA is available:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"`
    - If you get an error about GCC version 12.0.0 being required, run `conda install conda-forge::libgcc-ng==12`

Only use the `pyTED-cuda-cv` environment when running the optical flow model.

### 5. Run the processing pipeline

To run the pipeline with any model other than YOLO, run the following command:

```bash
python src/pipeline.py -d path/to/dataset -o path/to/output -c path/to/config.yml -m [mlnet | transalnet | tasednet | optical-flow | llava | gemma | minicpmv | gpt] [--cpu] [--annotations-path=path/to/annotations] #CHANGE CODE ACCORDINGLY (gemma and minicpmv in 'if' and chatgpt to gpt)
```

The pipeline will extract features from the videos using the selected method, convert cell values to Morton codes and run the event detection. The results will be placed in the output directory. Evaluation will be ran if the annotations path is provided.

To run the pipeline with YOLO, run the code in the first cell from the jupiter notebook run_yolo.ipynb from the YOLO folder.


### 6. Evaluate

To evaluate the event detection in terms of F1-score, sensitivity, specificity and mean IoU, use the following command:

`python src/evaluate.py path/to/event_window.csv path/to/annotations.yml`

## Datasets

When evaluating the different approaches on ZOD, we used a subset of videos containing 16 postive samples and 68 negative samples. The video IDs are listed below:

| ZOD Positives |                         ZOD Negatives                         |
| ------------- | ------------------------------------------------------------- |
| 000011        | 000082        | 000007        | 000143        | 000024        |
| 000046        | 000084        | 000019        | 000217        | 000168        |
| 000113        | 000098        | 000236        | 000229        | 000231        |
| 000169        | 000137        | 000238        | 000230        | 000234        |
| 000237        | 000161        | 000390        | 000232        | 000411        |
| 000292        | 000162        | 000414        | 000296        | 000464        |
| 000314        | 000306        | 000530        | 000461        | 000614        |
| 000316        | 000327        | 000583        | 000541        | 000680        |
| 000383        | 000684        | 000869        | 000603        | 000705        |
| 000389        | 000864        | 000905        | 000871        | 000877        |
| 000398        | 000865        | 000935        | 000881        | 000880        |
| 000433        | 000870        | 001012        | 000956        | 001049        |
| 000521        | 000900        | 001091        | 000977        | 001300        |
| 000653        | 000934        | 001199        | 001011        | 001307        |
| 000860        | 001326        | 001273        | 001067        | 001328        |
| 000893        | 001352        | 001294        | 001200        | 001341        |
|               | 001457        | 001245        | 001295        | 001412        |

We also used a subset of SMIRK to evaluate and compare the 4 VLMs. The subset contains the 253 videos listed below

0unxktasWzRk1LQiPpcAd
0XT6cv2RHZrN2NHqAQEDs
199ikecae9X7N59MxBBoc
1DxpmukFh0BVxLNUZhlSe
1G8XsWn477stGtIhOCiMO
1iQ4AVGbvpDqjggkUTKcF
1Jcu8GLc2zKl1FthhrUdG
1Na0oQaLZrIw88gbwBczl
1oOpT9379MevPcce0Q4u2
1pL6GwkVw1rtUZ20kiPmY
29KWLAyx2k83zYxf6GfD1
2M1tM2cQ3aLnfebG2Kx4K
3cqaCT3tyubALKBpoOIQb
3VvKwqpnect3fhFGgZ2hc
44YzqoTMZzjkEmg5ClxwF
4aVwFkg3yhbQ73gwuCZmO
4HFgop7SaG1ZGyR8a34UV
4jl2TolHXVz4oO46DEHjw
4kRZnfKBFiTAG09dlW1A3
5rtj5YLvD5Nm9naeNM9vm
5v7l19HuHPZKgjGnOahaB
66K8NnSAdYUT0vwkJVlqu
6eQvqaDAbz3URYwJnsZpi
6g8t2wBVDgTpWa23jSj50
6KAyTTnFWoPpVDCEIOW5s
6zKhrCKSdD1eqJDoewjS3
7cy1IgezCj1ZVo7yHkptW
7DXvvrHZ7r6t85UGYharG
7i61A4NZsvWkOHddlIw4W
7M6CwVNDAriFknAe5mDME
7TZTH21nfBFFx7bBHGBQy
7u4MdvDJ6aO96DHUgtZUn
8D3hYYdo8x86d0yUCMgfb
8jZTwYr0l8Boil1YTiJ6E
8uwNRifiocYgagy1Iz2z7
8w2XXqsopKva9YcaAPAkD
8yiesKfOfO0kngEFqkDOA
9f9hIIYsf7kDSmtrYPucK
9HpuMnDOSmWMXlbMFQPkF
9uGRUmg1lLhT8fyIQh9gM
a5lPgkhvEjM2vN0MzeDRx
acq1FSoAYGiVe0NaawLpc
aewWERv6GIhw6OIaHmYAL
AR22Lbv7yDGJYVrlf65pd
aWUQBtls89nqiD8sfYGmg
AwvCUP7xtgDpiadYmGpys
aXYd8pxAl2HVvzV55tRYS
AY5bKMK8iqT2XGTp4qHOQ
B0akZYLqdc47yhkzDYbtn
b7SBUvzSR2oUWRuxD1IqJ
BCH2ITzik0CJWUCZAxhiN

BCuWpV68DwzMEeTpdrtIT
BhCYPN6lzTYg2jSydsHnQ
bMrl6gTF8DnJpbDBsZzoM
bpF2pmEODgwTI9BkbtMe7
BvF4l6GItLZz98EI89Uhp
bxUPfZGW2Yy8y4RFLiwuq
c5FaSRdPoUVLfXS9WI9j3
ccRsc5FfdWqiGk5t0G2X5
CddrRMiQ7EisOGoZTGQ8L
cGTHyxJsEQzmkXmqsaAZO
chYUD6zeAjiLbKNlAsVmN
CqTAJCUuoRbtLQoL994pB
CQWTopvs4h4Kdqb1Z6UWI
cseoxBKAqBzCQU8uYbyl3
csHZGQ4i41pv9aPyty1ar
cUvmyTYFQO0pbHbEX3P6T
cwsYgz1m1dF9f16h3NWuk
DcaL5EbThwDh5g5TYwjSf
dcAyIKxWTny2aTT7yqIUs
dDkxE6PSFK2bSyvgL3QYD
dDVsxDBh3AAJAr0q7mRmA
DjUanYqNRZNMuNDtxPsmb
DN4BDYTV9nOz0wXXzElUh
Ds1vt7TSxsOHdApqXGe7e
DvLIMp3ncZ1haHCXeVpHU
DwSfOgGJd6xPRwflDiFBE
Dwu4P7AnOONv5yNcBrHPg
DxSQvHS1aZiGdbx523GA5
dxuwKKbrbeokcXlNFgIwa
E6JLPo8cRVelaYK1UTYOC
E7eTUGAgQ6CYg64iYdLnx
EA8dkohinPEmnlmEQv9Ld
EIsfkzjKVUUM2k27NI6Hn
ESp3cfI1MHA8X4bghJDC9
EVUhzRKULVQBthX4cLwjx
fAW6im95EwW6d2ebxXuW1
fcTCgzH9v1G2zbp2ahAxu
fnJP9mjoJnMMkcRqqaFHI
FO8LmYegfWdhPDDlKEF19
Fr4nEyKKrV6fQ4NGmRWuz
FRn1bShKhZO7lsus1rdRI
fUHNiX7LyL0LicTcjOtPy
Fx3iNxWHTj8JJFPoMOJm2
GE7Kp3TQiqqgYgL70ydW0
GErHtQbfU97epepBAmgDW
GEteKR9aTOEA8VQguwHxm
ghomvJIfCZgnd13cAuZMJ
GjM5rH09x4S4nm00J4UGN
GsgVMbvDoF7HqpDdMb4bV
GWHGau1K6nBhxMkGlzaEr
gzHXjDdoB0qZTxMkglUz4

H6g0Ui01nMmOw9XVAc5KY
h9kc6C8pEunk1qsKMEG7k
HbRBshZL4TLFtrtnqXikc
HbTQSEAstcDjg4mEDvTeW
hozneFVU9WCdSUiPBRX0f
hVPeBUSRdlEDhDNfjG2PM
HWz5Oiwpm2FIphXTq6WVU
i6uoH8RgcmadHpIYDlygx
IFGmZTtK9U3ntir243GGt
imetfUYDv4FFbFlEEiNZm
imk6svrvwTqYqbRo7ye8f
IQMdOq7RMEO6cCGTJIMvO
ISKf5HHjEjbhoS7Seid4P
iUEW3A7Xm5VV2Ux9T1101
Iz0SIIYGEss4122aIJAuG
j2hQjWaO1jHurLkhVFrh2
jKXfODXjpYOnV9lyOdtBz
JVSjTfBp37Zh0bGQRzGxQ
kAWKnOWtacMRb4Q1oBfHJ
keUkrpmxplXIBYdEdxfCi
KMDfv0dG9foDZNVaOeXn7
KmUFO5iCWRY40xyiiA6gK
kRgBg5qkuNg6EX8qFD0z1
KyUP5zR9YzATWB6ybRw2I
l1774saDnYvjh6EvBsimw
l4XXtwvnAckiG4NaojSls
l8Noy08RZlY8TNrykyTFR
l9OdUvAJERcWBpbLWoKvY
LHS10GAD6VhyeM0kyGyYX
lMzVVCqXfTccZxx5HVxAB
LsGsa2ss9tHgnebbkE9Q4
lwdN0IURN7vHmfTK9KQ4W
lzldY8c9CAs1Mb1xm4FWB
m2GDtR0Gt3o8qmzDmYvu7
m7H5RZZXjaiOKKuzLLVIR
m9ApCxu73n8ARDCRqMr5V
mBZyaoLOGc6WwcFYitELM
mF5ZBTubkQCKFahM2Ik7o
MGDxvgfXTZX03mSr6rfyu
mhH1xK6sYxiKTMw8BAVFE
MIhx2YI1fJ7ERLjxVAwLU
MjMLmTHdp6OCp1Qj7BwqN
mKOC1RZqRM8E6lhbw7zGb
MqmzqL1d9zHO7pqAwwrid
mS092GyYB86za0EclXiwW
msoGcqpBciweOE1S2aS8O
mu2P4ZVQBpRIFu82auCR3
mVm63WLZVMQ02FmCdsKT6
MylXkdl3xoYFD92nMf9GF
N1XNslq9vd0xvMQ7ltTol
n5eggmnqDxmGKsODjyXVB

nbPqhMsg8m3ICeZ0FTHM0
NijNwkK8dvjtx6X8BM3c5
nooCbUlzm1fXXC0yzsL7O
nqmnFovy8nKg01kwGm5z7
nuZN9r6PfIyoqlMN82mb0
NWHNk9fOnazDn3WBk0EEL
nyRvfUvpbGwMfX3KMjkJO
O5oXcEIfFkZnzkiKYONjw
OjS0RlsLVqjMf6FBWCikP
Oty7aQpnV2KAox24XDYdQ
oxUvTrJ4suaAaQ7aSs7Gk
Oy6z0pUAE7OhMcNwyJrPu
Oz0aqoYCzOsam7GZ01IZd
P711l9ZD9arWNlT4BnKpk
pBAOz7pLoxkZBZCvc4x5F
PhRni1eLgNKYfBgSYlqi7
Pi5g26l9RlVdYyA6Nfvnx
PLoCrsdYMNCex9AE3ZL3R
PmIxOhNtYGt9gkJFHUqcw
q2K8g5kN963mq2bhTtuso
q56smRWTPVNxfMnjNqrms
q62IQTPaADtmAYdH06a5O
q6uleHxxLqgwOAQieFdnG
QCPUOwnAr33BeZKuwXR5a
qd9VExgqHOzfgbWYut3XJ
Qg18rhdrT0ZROyY89EGLF
qGtdnWOIxiPtFVsadbf6G
Qh4TY06r6xr1MamlLNiuv
qMnr2TWrVktZCRGaoQZfu
QvU0hESU6h28WDdWJgGxp
RbNikfrZ6NqbGa49K4YcI
rCOVDcvbHs56ax8Yqslzb
RDdicre4EiyzIjAf2lxiL
rgmbvHiVfVyGkWPboJPcC
RISPxF7nvfXMEcdcprVAG
Rl3HaQ9heq47rfedSJVLE
rP5OoZfdld3N4fLk7kwMx
rqf2MGnaSoCzQE3Fh46Cx
rslTvjADdsnaVK7YVL911
RvdDGdrnJIEcR3mXH5Dly
S5eImjtZ2zXohy4ooABmK
s6Wt5yvaMd8DICh70I5QM
SDjFA8ahTi8N2Lp4JpjMS
shfUeUZwSSgwWRbSuzjko
STaiso9zpvvcXp2DAQWUf
SWbiVM1aji8nI5h9YIjXB
sZwjKjfRnoL0F6NrEHdT2
t0S2QzcymJvMfRU8yMjIV
t1HiG7iUVskkq7crVIPQf
T2OgrDYvyFETypjMxqM7a
tBfaNf2Etp3pfzYW8KN0x

TC0npG2COtOiRY3WzCBa3
te6lgGaPnWlshYqBRu9nl
tgWjGySwvo3UX83BEsMVE
TjiJw6vYb1bq6yhPxQsEH
TrLHo9J1EDLMGaJsz9csj
TSPQltYSAWdNeSSIXSrni
TTD3VIdU7wpD8hqMqmLMq
U3n2a6n2r7RGStqM5PJbd
uCw2kXqnR6shi3x9V4G8L
UdVsYFxNpVuLLrsA4FmGd
uj8LP8r42ZUzG2quBu7sj
usmXpMrB6jIRxHEZ1HPYS
vaWyFRS954GRKq30IP8MA
VFN1D4KMZT5UYTcwCOcog
VQ4lRVjHiHu00ohwnD7tW
VXoWkY7DDaORJn2dt6V7Y
W8XR9y1cG1yVo3s9UeA3W
waCilUnSf1E1Bsa3pmy1m
wbmJ1vng2PCCTjaKvaUQQ
WjItYFTTS0r4CYFSqV6H1
wJywt82yOIFKLs5ep26Si
wkwF74rn0VqijFsPqMhVC
wNJwFX8PI7dPO5JvlBcGf
WOuyssyxcw4c4ireUMWQX
wT57PM7G6exJbs0yQB6lx
wy2sQvx8USDy3B2YpukeF
wYqEyCxJlC5dHPhi1wknS
xbrdaCoXYZvlkkXEKZVgW
xHRVvfJsvqPSkPygU3ine
xNhUS8iNhtCtb0XblE7pH
XZNVWN2oQwO23qg610323
xZzgR2HYyYqrTshawkArK
y6N76sbEnwlD1tJBniw9s
YAkIEyxA2q9aJUBGbExwh
Yb9uofi8LMJpNMKF5746H
YE42o2PdrMZFKzuQWm9QJ
YI6Cmsep8nO3sYbTNAWni
yJB9hO6KvGfkEtR3rZMd4
YjHRja4aOAPPMkiJi3JZw
yMsSVU9jl2MGaIGg5hX18
YSf7NJ3n3aZNq7LWzAzod
YSpxkhYrmtug30Atn4iin
YxGFrwzGKl4HqoN5KOfpV
yYQcTKRvKbAXpFUGQ6Tto
z62CijQ9mDbdGErDGzHwa
zCPBj0YwhePI5fFxL39qN
Zkz6L94DJ4PO0L9hoCxwq
zohOtnzkQprVjSVngsIMl
zzIEDiLdsKpFtp9zU1UGG

## Licensing Information Notice

For this dataset, Zenseact AB has taken all reasonable measures to remove all personally identifiable information, including faces and license plates. To the extent that you like to request removal of specific images from the dataset, please contact privacy@zenseact.com.
