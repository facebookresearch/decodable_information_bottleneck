# decodable_information_bottleneck
This repostiory contains our implementation of the NeurIPS 2020 paper:

Learning Optimal Representations with the Decodable Information Bottleneck
NeurIPS 2020 (Spotlight)
Yann Dubois, Douwe Keila, David Schwab, Rama Vedantam 

The project describes what optimal representations for supervised learning
look like from the perspective of a function family of interest, and
suggests a novel objective which can be used in practice to estimate
such optimal representations from a dataset.

In addition the project also evaluates various generalization measures
for supervised learning following experiments from:

Jiang, Yiding, Behnam Neyshabur, Hossein Mobahi, Dilip Krishnan, and Samy Bengio. 2019. “Fantastic Generalization Measures and Where to Find Them.” in International Conference on Learning Representations (ICLR), 2020

## Usage
Cleaning up in progress more details in the readme soon!

## Requirements
decodable_information_bottleneck requires or works with

* Linux
* Pytorch
* Hydra
* Pandas
* Numpy
* Scikit-Learn
* Skorch
* Matplotlib
* Seaborn

## Installing decodable_information_bottleneck
Please install all the dependencies using:

```
pip3 install -r requirements.txt
```

## How decodable_information_bottleneck works
Decodable information bottleneck proposes notions of minimality
and sufficiency with respect to a function family of interest e.g.
2 layer MLP and shows how to use these notions in a practical
objective for supervised learning. 

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
decodable_information_bottleneck is MIT licensed, as found in the LICENSE file.