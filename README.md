# Deep-Generalized-Legendre-Galerkin

Here is the official Pytorch implementation of the paper DGLG: A Novel Deep Generalized Legendre–Galerkin Approach to Optimal Filtering Problem.

The paper is coauthored by Ji Shi, Xiaopei Jiao (equally contributed) and Stephen Shing-Toung Yau.

It has been accepted by IEEE transactions on automatic control.

Journal version: https://ieeexplore.ieee.org/document/10735353

# Code explanation

PINN_train_scaling_system_cubic.py: Code for training neural basis in terms of each Generalized Legendre polynomial by Physics-Informed Neural Network.

filter_algorithms.py: Code for relavant required functions such as definition of cubic filtering system, Generalized Legendre Polynomial, Spectral matrices etc.

filter_implement.py: Code for filtering result, including Monte-Carlo sampling trajectory of system equation, DGLG implement.

neural_basis folder: Code for trained Neural basis offline.

# Citations
If you feel this code is helpful, please kindly cite our paper.
```bibtex
@article{shi2024dglg,
  title={DGLG: A Novel Deep Generalized Legendre-Galerkin Approach To Optimal Filtering Problem},
  author={Shi, Ji and Jiao, Xiaopei and Yau, Stephen S-T},
  journal={IEEE Transactions on Automatic Control},
  year={2024},
  publisher={IEEE}
}
```
