import torch
class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    """FixedScheduler that does not modify the learning rate at all

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        The optimizer for the model
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, last_epoch: int = -1, **kwargs
    ):
        super(FixedScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(
        self,
        step: int,
    ) -> float:
        return 1.0
class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    """WarmupLinearScheduler combines warm up and linear decay the learning rate

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        The optimizer for the model
    warmup_steps: int
        The number of steps for LR to reach the peak value
    scheduler_steps: int,
        The total number of expected training iterations
    min_ratio: float
        The minimum ratio of the learning rate (starting and ending ratio)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        scheduler_steps: int,
        min_ratio: float,
        last_epoch: int = -1,
        **kwargs,
    ):

        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(
        self,
        step: int,
    ) -> float:

        if step < self.warmup_steps:
            return (1 - self.min_ratio) * step / float(
                max(1, self.warmup_steps)
            ) + self.min_ratio

        return max(
            0.0,
            1.0
            + (self.min_ratio - 1)
            * (step - self.warmup_steps)
            / float(max(1.0, self.scheduler_steps - self.warmup_steps)),
        ) + 5e-5