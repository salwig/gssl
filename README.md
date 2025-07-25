# Graph-Based Semi-Supervised Learning (GSSL)

**Graph-Based Semi-Supervised Learning (GSSL)** is a Python/C++ package providing efficient implementations of ten algorithms for large-scale graph-based semi-supervised learning. All algorithms share a unified, easy to use API.
To get started, explore the provided [example](#run-the-demo).

## Available Algorithms

- **AGR** – Anchor Graph Regularization [[1]](#related-publications)
- **EARG** – Efficient Anchor Graph Regularization [[2]](#related-publications)
- **f-FME** – fast Flexible Manifold Embedding [[3]](#related-publications)
- **r-FME** – reduced Flexible Manifold Embedding [[3]](#related-publications)
- **MiMoLaP<sup>LG</sup>** – 'local and global consistency' Mixture Model Label Propagation[[4]](#related-publications)
- **MiMoLaP<sup>H</sup>** – 'harmonic' Mixture Model Label Propagation [[4]](#related-publications)
- **DDGL** – Data Distribution Based Graph Learning [[5]](#related-publications)
- **MFA-GL** – Mixture of Factor Analyzers Graph Learning [[6]](#related-publications)
- **v-GMM<sup>d</sup>-GL** – variational Gaussian Mixture Model Graph Learning [[6]](#related-publications)
- **v-MFA-GL** – variational Mixture of Factor Analyzers Graph Learning [[6]](#related-publications)

Refer to the [related publications](#related-publications) for more details.

## Installation

### Requirements

Ensure the following requirements are met for installation:

- A C++ compiler that supports the C++17 Standard, such as the [GNU g++ Compiler](https://gcc.gnu.org/)
- [Python 3](https://www.python.org/) `version >= 3.9` (for the Python API)
- [OpenMP](https://www.openmp.org/) (for parallel execution)
- [Git](https://git-scm.com/)

Please note that the code has only been tested on Linux distributions.

### Setup

1. **Clone the Repository**

    Clone this repository with the `--recursive` flag to include the required submodules:

    ```bash
    git clone --recursive git@gitlab.uni-oldenburg.de:ml-oldb/labelpropagation/gssl.git
    cd gssl/
    ```

    **Note:** if you have cloned the repository without the `--recursive` flag, run the following command inside the repository to initialize and update the submodules manually:
    Alternatively, manually clone the necessary libraries into the project directory:

    ```bash
    git submodule update --init --recursive
    ```


    This will download further requirements: [VAMM](https://github.com/variational-sublinear-clustering/vamm.git) and [EMMI](https://github.com/variational-sublinear-clustering/emmi.git), for mixture model optimization, together with their dependencies.

2. **Install Python Packages**

    We recommend using [Anaconda](https://www.anaconda.com/) to manage the installation and create a new environment for the project:

    ```bash
    conda create -n gssl python=3.9
    conda activate gssl
    ```

    Next, install the package with [pip](https://pypi.org/project/pip/):

    ```bash
    pip install .
    ```

    This command builds the C++ libraries and installs the required Python dependencies, including the packages **EMMI** and **VAMM** for mixture model optimization. (Therefore, you can disregard the installation instructions in the respective README files for **EMMI** and **VAMM**.)


## Run the Demo

After installation, you can run the demo to see the **GSSL** algorithms in action. The demo script runs the different graph based semi-supervised learning algorithms on the [Letter Recognition](https://archive.ics.uci.edu/dataset/59/letter+recognition) dataset and reports the classification error rates of these algorithms.

To run the demo, navigate to the `examples` directory and execute the script:

```bash
cd examples/
python3 demo.py
```

## Related Publications

[1] W. Liu, J. He, and S. F. Chang, "Large graph construction for scalable semi-supervised learning", ICML, pp. 679-686 (2010).

[2] M. Wang, W. Fu, S. Hao, D. Tao, and X. Wu, "Scalable semi-supervised learning by efficient anchor graph regularization", TKDE 28.7, pp. 1864-1877 (2016).

[3] S. Qiu, F. Nie, X. Xu, C. Qing, and D. Xu, "Accelerating flexible manifold embedding for scalable semi-supervised learning", TCSVT 29.9, pp. 2786-2795 (2019).

[4] M. Chi, X. He, and S. Yu, "Mixture model label propagation", CIKM, pp. 1889-1892 (2010).

[5] Y. Zhang, S. Ji, C. Zou, X. Zhao, S. Ying, and Y. Gao, "Graph learning on millions of data in seconds: Label propagation acceleration on graph using data distribution", TPAMI 45.2, pp. 1835-1847 (2023).

[6] S. Salwig*, T. Kahlke* and J. Lücke, "Variational Graph-Based Semi-Supervised Learning", proposed work.
*joint first authorship.
