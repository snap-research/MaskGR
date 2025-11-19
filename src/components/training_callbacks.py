from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override
from src.utils import RankedLogger
from src.utils.file_utils import copy_to_remote, file_exists_local_or_remote
command_line_logger = RankedLogger(__name__, rank_zero_only=True)
class ModelCheckpointToGCS(ModelCheckpoint):
    def __init__(
        self,
        gcs_path: str,
        upload_after_n_checkpoints: Optional[int] = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.dirpath.startswith("gs://"):
            command_line_logger.warning(
                f"Local checkpoint path {self.dirpath} is a GCS path but should be a local path."
                f"Changing it to default dirpath './checkpoints'."
            )
            self.dirpath = "./checkpoints"

        self._gcs_path = gcs_path
        self._upload_after_n_checkpoints = upload_after_n_checkpoints
        self._checkpoint_counter = 0

    def _copy_to_gcs(self, trainer):
        # Only copy one per node
        if trainer.local_rank == 0 or trainer.local_rank is None:
            if file_exists_local_or_remote(self.dirpath):
                try:
                    copy_to_remote(self.dirpath, self._gcs_path)
                except FileNotFoundError as e:
                    command_line_logger.error(
                        f"Checkpoint {self.dirpath} not found. Skipping copy to GCS."
                    )

    @override
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        After the usual checkpointing, we copy the checkpoint to GCS if we meet the criteria for `_upload_after_n_checkpoints`.
        """
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        skip_batch = self._every_n_train_steps < 1 or (
            trainer.global_step % self._every_n_train_steps != 0
        )

        if not skip_batch:
            self._checkpoint_counter += 1
            if self._checkpoint_counter % self._upload_after_n_checkpoints == 0:
                # We copy the entire directory as distributed checkpoints might have multiple files.
                self._copy_to_gcs(trainer)
                self._checkpoint_counter = 0

    def teardown(self, trainer, pl_module, stage):
        """
        Copy the checkpoint to GCS after the training is done.
        """
        self._copy_to_gcs(trainer)
        super().teardown(trainer, pl_module, stage)