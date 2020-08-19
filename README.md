# Towards Reliable Evaluation of Algorithms for Road Network Reconstruction from Aerial Images

Existing connectivity-oriented performance measures rank road delineation algorithms inconsistently, which makes it diffcult to
decide which one is best for a given application. We show that these inconsistencies stem from design flaws that make the metrics insensitive to whole classes of errors. This insensitivity is undesirable in metrics intended for capturing overall general quality of road reconstructions. In particular, the scores do not reflect the time needed for a human to fix the errors, because each one has to be fixed individually. To provide more reliable evaluation, we design three new metrics that are sensitive to all classes of errors. This sensitivity makes them more consistent even though they use very different approaches to comparing ground-truth and reconstructed road networks. We use both synthetic and real data to demonstrate this and advocate the use of these corrected metrics as a tool to gauge future progress.

Plase cite our paper if you find the new metrics useful.

```
@inproceedings{Citraro20,
    author = {L. Citraro, M. Koziński and P. Fua},
    title = {Towards Reliable Evaluation of Algorithms for Road Network Reconstruction from Aerial Images},
    booktitle = {ECCV},
    year = {2020}
}
```

## Content
Our new evaluation methods:
- OPT-J (Junction based)
- OPT-P (Path based)
- OPT-G (Subgraph based)

Other evaluation methods available in this repository: 
- correctness, completeness and quality 
- toolong/tooshort
- holes & marbles

The Junction metric can be found here https://github.com/mitroadmaps/roadtracer while APLS metric here https://github.com/CosmiQ/apls

## Prerequisites

- numpy
- scipy
- imageio
- networkx
- sklearn
- matplotlib

## Installation
add this to you python path
```
export PYTHONPATH="...location of this folder...:$PYTHONPATH"
```

## Usage
check the examples in folder `examples`
