# profiler

Find below the existing configs for the profiler folder. To have your YAML file indexed, add a docstring at the beginning of the file.
The docstring should be a series of comment lines starting with two '#' characters.

Example:
```
## This is a docstring
## describing the YAML file.
key: value
```


## advanced.yaml

This profiler uses Python’s cProfiler to record more detailed information about time spent in each function call recorded during a given action.
The output is quite verbose and you should only use this if you want very detailed reports. Works for multinode and single node profiling.
https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.profilers.AdvancedProfiler.html#lightning.pytorch.profilers.AdvancedProfiler


## none.yaml

Default profiler for the trainer, does not do anything


## pytorch_profiler.yaml

This profiler uses PyTorch’s Autograd Profiler and lets you inspect the cost of different operators inside your model - both on the CPU and GPU.
It also can generate a trace.json file that can run in Chrome’s chrome://tracing tool to visualize the performance of your model.
https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.profilers.PyTorchProfiler.html#lightning.pytorch.profilers.PyTorchProfiler
It is the recommended profiler. The only limitation is that it currently does not work well for multinode. For multinode, use advanced profiler.


## simple.yaml

Simple profiler that simply records the duration of actions (in seconds) and reports the mean duration of each action and the total time spent over the entire training run.
https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.profilers.SimpleProfiler.html#lightning.pytorch.profilers.SimpleProfiler
