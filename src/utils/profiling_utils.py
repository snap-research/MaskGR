import dataclasses
import logging
import os
import pickle
import lightning as L
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig
from torch.profiler.profiler import ProfilerAction
from src.utils.file_utils import copy_to_remote
@dataclasses.dataclass(frozen=True)
class Schedule:
    """Class that can be used as profiler ``schedule`` argument.
    This was copied from https://github.com/pytorch/pytorch/blob/fd8631f4da01e378769f19714ea0926e7af5ff12/torch/profiler/profiler.py
    which was a PR that never got merged to PyTorch but actually provides a pickable schedule for the PyTorch profiler.

    The profiler will skip the first ``skip_first`` steps, then wait for
    ``wait`` steps, then do the warmup for the next ``warmup`` steps, then do
    the active recording for the next ``active`` steps and then repeat the
    cycle starting with ``wait`` steps. The optional number of cycles is
    specified with the ``repeat`` parameter, the zero value means that the
    cycles will continue until the profiling is finished.
    """

    wait: int
    warmup: int
    active: int
    repeat: int = 0
    skip_first: int = 0

    def __post_init__(self) -> None:
        if not self.warmup:
            logging.warning(
                "Profiler won't be using warmup, this can skew profiler results"
            )

    def __call__(self, step: int) -> ProfilerAction:
        assert step >= 0
        if step < self.skip_first:
            return ProfilerAction.NONE
        else:
            step -= self.skip_first
        num_steps = self.wait + self.warmup + self.active
        if self.repeat > 0 and step / num_steps >= self.repeat:
            return ProfilerAction.NONE
        mod_step = step % num_steps
        if mod_step < self.wait:
            return ProfilerAction.NONE
        elif mod_step < self.wait + self.warmup:
            return ProfilerAction.WARMUP
        else:
            return (
                ProfilerAction.RECORD
                if mod_step < num_steps - 1
                else ProfilerAction.RECORD_AND_SAVE
            )
@rank_zero_only
def log_and_save_profiler_output(
    trainer: L.Trainer, cfg: DictConfig, log: logging.LoggerAdapter
) -> None:
    # If profiler is set on trainer, we print the profiler summary, store its output to disk, and copy it to remote
    # if output path is a GCS path.
    if hasattr(trainer, "profiler") and not isinstance(
        trainer.profiler, L.pytorch.profilers.PassThroughProfiler
    ):
        log.info("Printing profiler summary")
        log.info(trainer.profiler.summary())

        try:
            # We save the profiler outputs as a pickle file. This allows us to
            # load it later and sort it by different metrics for analysis.
            profiler_output_path = cfg.paths.profile_dir + "/profiler_output.pkl"
            log.info(
                f"Saving profiler output to disk. The pickled file will be saved at {profiler_output_path}"
            )

            data = trainer.profiler.function_events.key_averages(
                group_by_input_shapes=trainer.profiler._group_by_input_shapes
            )
            pickle.dump(data, open(profiler_output_path, "wb"))
            # check if output dir is a gcs path
        except Exception as e:
            log.error(f"Error saving profiler output: {e}")

        # check if path exists and copy to remote if it does
        if os.path.exists(profiler_output_path):
            copy_to_remote(cfg.paths.profile_dir, cfg.paths.output_dir)