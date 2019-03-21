## Implementation of Meta-gradient method for A3C (Asynchronous Advantage Actor-Critic)

#### Running

```
python train.py --model_dir ./tmp/a3c --env Breakout-v0 --t_max 5 --eval_every 10 --parallelism 8
```

To monitor training progress in Tensorboard:

```
tensorboard --logdir=/tmp/a3c
```

#### Components

- [`train.py`](train.py) contains the main method to start training.
- [`estimators.py`](estimators.py) contains the Tensorflow graph definitions and the training operations for the Policy and Value networks. 
- [`worker.py`](worker.py) contains the codes to drive each worker.
- [`policy_monitor.py`](policy_monitor.py) evaluates the policy network and generates summaries.
