# Softmax-DQN

This repository provides the Python code for the softmax DQN algorithms in the following paper.

Zhao Song, Ronald E. Parr, and Lawrence Carin, "[Revisiting the Softmax Bellman Operator: New Benefits and New Perspective](http://people.ee.duke.edu/~lcarin/ICML2019_softmax.pdf)", *Proceedings of the 36th International Conference on Machine Learning (ICML 2019)*, Long Beach, CA, June, 2019


## Acknowledgements

This implementation is built on the the publicly available code from https://github.com/spragunr/deep_q_rl/, by Nathan Sprague and other contributors. Please read `LICENSE.txt` (where the copyright notice is located) from Nathan Sprague under the `./softmax_dqn` folder, and accept the corresponding terms, before you start.

Distribution and use of this code is subject to the following agreement:

>*This Program is provided by the authors as a service to the research community. It is provided without cost or restrictions, except for the User's acknowledgement that the Program is provided on an "As Is" basis and User understands that the authors make no express or implied warranty of any kind. The authors specifically disclaim any implied warranty or merchantability or fitness for a particular purpose, and make no representations or warranties that the Program will not infringe the intellectual property rights of others. The User agrees to indemnify and hold harmless the authors from and against any and all liability arising out of User's use of the Program.*


## Quick Start

To start, install all required libraries in `./softmax_dqn/dep_script.sh`, and then run the following bash script. 
```
$./softmax_dqn/deep_q_rl/job.sh
```

## Citation

If you find this code useful, please cite the work with the following bibtex entry
```
@inproceedings{softmax2019,
    title={Revisiting the Softmax Bellman Operator: New Benefits and New Perspective},
    author={Song, Zhao and Parr, Ronald E. and Carin, Lawrence},
    booktitle={Proceedings of the 36th International Conference on Machine Learning},
    year={2019}
}
```
