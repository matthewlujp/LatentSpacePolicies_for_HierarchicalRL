# LatentSpacePolicies_for_HierarchicalRL
Pytorch implementation of "[Latent Space Policies for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1804.02808)"


## Install required packages with pip3
```sh
pip3 install -r requirements.txt
```

## Try out
Train agent in "HumanoidBulletEnv" in PyBullet library.  
A dedicated configuration file for this environment is in configs directory.
```sh
python3 main.py --config configs/config_lsphrl_humanoid.toml --save-dir results/test_humanoid
```

A model file is saved in ```results/test_humanoid/checkpoints/model.pth```.
Load this file to run the trained policy.
```sh
python3 main.py --config configs/config_lsphrl_humanoid.toml -m results/test_humanoid/checkpoints/model.pth -e -r
```

Options
```
--config              Config file path
--save-dir            Save directory
--visualize-interval  Interval to draw graphs of metrics.
--device              Device for computation.
-e, --eval            Run model evaluation.
-m, --model-filepath  Path to trained model for evaluation.
-r, --render          Render agent behavior during evaluation.
```


## Implementation notes
hoge



## Evaluations
foo

|HumanoidBulletEnv|
|---|
|![](https://user-images.githubusercontent.com/13263381/78898710-35e09e80-7aaf-11ea-8190-e08189f42e08.png)|




