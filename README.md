# Quaternion Dynamical Systems


This is the implementation of Quaternion Dynamical Systems, or quaternion-DS introduced in Section III of [paper](https://arxiv.org/abs/2403.16366)  <i>"SE(3) Linear Parameter Varying Dynamical Systems for Globally Asymptotically Stable End-Effector Control"</i>  This code can be used standalone as a planning/control method for orientation, or as part of the [SE(3) LPV-DS](https://github.com/sunan-sun/se3_lpvds) pipeline.

## Dataset

We primarily use the real-world datasets from this [repo](https://github.com/sayantanauddy/clfd) for benchmarking, including tasks such as pouring, box opening and etc. To test your own dataset, please examine and follow the data structure of the loaded inputs prior to the "<i>Process data"</i> section in the example code `main.py`.


## Usage

### 1. Create a Virtual Environment

It is recommended to use a virtual environment with Python >= 3.9. You can create one using [conda](https://docs.conda.io/en/latest/):

```bash
conda create -n venv python=3.9
conda activate venv
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Run the Interactive Code

Select the data of your choice and run `main.ipynb`


## References

If you find this code useful for you project, please consider citing the paper:

```
@INPROCEEDINGS{10801844,
  author={Sun, Sunan and Figueroa, Nadia},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={SE(3) Linear Parameter Varying Dynamical Systems for Globally Asymptotically Stable End-Effector Control}, 
  year={2024}
}
```

## Contact

Contact: [Sunan Sun](https://sunan-sun.github.io/) (sunan@seas.upenn.edu)