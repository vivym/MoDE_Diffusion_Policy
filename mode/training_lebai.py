from __future__ import annotations

import logging
from pathlib import Path
import sys
sys.tracebacklimit = None
import os 
from functools import lru_cache
import wandb
import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

import numpy as np
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TrF
from torch.utils.data import ConcatDataset

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import mode.models.mode_agent as models_m
from mode.utils.utils import get_git_commit_hash, get_last_checkpoint, initialize_pretrained_weights, print_system_env_info

from mode.h5ds.dataset import H5Dataset, Mappers, Statistics, Statistic

# Add local repo to path
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def clear_cuda_cache():
    """Clear CUDA cache and garbage collect unused memory."""
    if torch.cuda.is_available():
        # Empty CUDA cache
        torch.cuda.empty_cache()
        # Force garbage collection
        import gc
        gc.collect()
        # Log memory stats
        for i in range(torch.cuda.device_count()):
            memory_stats = torch.cuda.memory_stats(i)
            allocated = memory_stats.get('allocated_bytes.all.current', 0) / (1024**3)
            reserved = memory_stats.get('reserved_bytes.all.current', 0) / (1024**3)
            logger.info(f"GPU {i} Memory: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)

def setup_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    return [hydra.utils.instantiate(cb) for cb in callbacks_cfg.values()]

def setup_logger(cfg: DictConfig, model: LightningModule):
    pathlib_cwd = Path.cwd()
    if "group" in cfg.logger:
        cfg.logger.group = pathlib_cwd.parent.name
        cfg.logger.name = f"{pathlib_cwd.parent.name}/{pathlib_cwd.name}"
        cfg.logger.id = cfg.logger.name.replace("/", "_")
    return hydra.utils.instantiate(cfg.logger)


class LebaiH5Dataset(H5Dataset):
    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)

        return {
            "actions": item["actions"],
            "rgb_obs": {
                "rgb_static": item["observations"]["rgb_primary"],
                "rgb_gripper": item["observations"]["rgb_primary"],
            },
            "robot_obs": item["observations"]["proprio"],
            "lang_text": item["tasks"]["language_instruction"],
        }


def action_mapper(
    action: np.ndarray,
    *,
    statistic: Statistic | None = None,
) -> np.ndarray:
    if statistic is None:
        return action

    action = np.concatenate(
        [action[:, 48:48 + 6], action[:, -1:]],
        axis=1,
    )

    mean = np.concatenate(
        [statistic.mean[48:48 + 6], statistic.mean[-1:]],
        axis=0,
    )
    std = np.concatenate(
        [statistic.std[48:48 + 6], statistic.std[-1:]],
        axis=0,
    )

    action = (action - mean) / std

    return torch.from_numpy(action).float()


@lru_cache
def get_rgb_transform(
    image_size: int,
    random_crop: bool,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    training: bool,
) -> T.Compose:
    if training:
        if random_crop:
            transforms = [
                T.RandomResizedCrop(
                    size=image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    antialias=True,
                )
            ]
        else:
            transforms = [
                T.Resize(image_size, antialias=True),
                T.CenterCrop(image_size),
            ]

        transforms.extend([
            T.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
            ),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),
        ])

        return T.Compose(transforms)
    else:
        return T.Compose([
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),
        ])


def rgb_mapper(
    rgb: np.ndarray | None,
    *,
    statistic: Statistic | None = None,
    image_size: int = 224,
    random_crop: bool = True,
    mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
    std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
    training: bool = True,
) -> np.ndarray:
    assert rgb is not None

    rgb = np.transpose(rgb, (0, 3, 1, 2))
    rgb = torch.from_numpy(rgb)
    rgb = tv_tensors.Video(rgb)

    # Pad to square
    if rgb.shape[-1] != rgb.shape[-2]:
        max_size = max(rgb.shape[-1], rgb.shape[-2])
        left = (max_size - rgb.shape[-1]) // 2
        right = max_size - rgb.shape[-1] - left
        top = (max_size - rgb.shape[-2]) // 2
        bottom = max_size - rgb.shape[-2] - top
        rgb = TrF.pad(rgb, padding=(left, top, right, bottom))

    transform = get_rgb_transform(
        image_size=image_size,
        random_crop=random_crop,
        mean=mean,
        std=std,
        training=training,
    )

    return transform(rgb)


def proprio_mapper(
    proprio: np.ndarray,
    *,
    statistic: Statistic | None = None,
) -> np.ndarray:
    if statistic is None:
        return proprio

    proprio = np.concatenate(
        [proprio[:, 48:48 + 6], proprio[:, -1:]],
        axis=1,
    )

    mean = np.concatenate(
        [statistic.mean[48:48 + 6], statistic.mean[-1:]],
        axis=0,
    )
    std = np.concatenate(
        [statistic.std[48:48 + 6], statistic.std[-1:]],
        axis=0,
    )

    proprio = (proprio - mean) / std
    return torch.from_numpy(proprio).float()


def language_instruction_mapper(
    text: str | list[str],
    *,
    statistic: Statistic | None = None,
):
    return text


class LebaiDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage: str):
        mappers = Mappers(
            action=action_mapper,
            observation={
                "rgb_primary": rgb_mapper,
                "proprio": proprio_mapper,
            },
            task={
                "language_instruction": language_instruction_mapper,
            }
        )

        current_dir = Path(__file__).absolute().parent.parent

        repeat_factor = 20

        ds1 = LebaiH5Dataset(
            str(current_dir / "data/pick_and_place_banana_v0.2.h5"),
            obs_seq_len=1,
            action_seq_len=32,
            mappers=mappers,
            training=True,
            repeat=2 * repeat_factor,
        )

        ds2 = LebaiH5Dataset(
            str(current_dir / "data/pick_and_place_apple_v0.2.h5"),
            obs_seq_len=1,
            action_seq_len=32,
            mappers=mappers,
            training=True,
            repeat=2 * repeat_factor,
        )

        ds3 = LebaiH5Dataset(
            str(current_dir / "data/pick_and_place_lemon_v0.2.h5"),
            obs_seq_len=1,
            action_seq_len=32,
            mappers=mappers,
            training=True,
            repeat=4 * repeat_factor,
        )

        ds4 = LebaiH5Dataset(
            str(current_dir / "data/pick_and_pour_v0.1.h5"),
            obs_seq_len=1,
            action_seq_len=32,
            mappers=mappers,
            training=True,
            repeat=1 * repeat_factor,
        )

        ds = ConcatDataset([ds1, ds2, ds3, ds4])

        self.train_datasets = {"lang": ds}

    def train_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                batch_size=64,
                num_workers=4,
                pin_memory=True,
                shuffle=True,
            )
            for key, dataset in self.train_datasets.items()
        }


@hydra.main(config_path="../conf", config_name="config_lebai")
def train(cfg: DictConfig) -> None:
    try:
        # Setup environment
        # os.environ['HYDRA_FULL_ERROR'] = '1'
        # Set memory allocation configuration
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        seed_everything(cfg.seed, workers=True)
        torch.set_float32_matmul_precision('medium')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Clear CUDA cache before initialization
        clear_cuda_cache()
        
        # Initialize components
        log_rank_0(f"\nInitializing training for seed {cfg.seed}")
        #datamodule = hydra.utils.instantiate(cfg.datamodule)
        datamodule = LebaiDataModule()
        model = hydra.utils.instantiate(cfg.model) if get_last_checkpoint(Path.cwd()) is None else \
               getattr(models_m, cfg.model["_target_"].split(".")[-1]).load_from_checkpoint(get_last_checkpoint(Path.cwd()).as_posix())
        
        if "pretrain_chk" in cfg:
            initialize_pretrained_weights(model, cfg)
            
        # Setup training
        train_logger = setup_logger(cfg, model)
        callbacks = setup_callbacks(cfg.callbacks) + [LearningRateMonitor(logging_interval="step")]
        
        # Set unique working directory for each seed
        work_dir = Path.cwd() / f"seed_{cfg.seed}"
        work_dir.mkdir(exist_ok=True)
        os.chdir(work_dir)
        
        trainer_args = {
            **cfg.trainer,
            "logger": train_logger,
            "callbacks": callbacks,
            "benchmark": False,
            "strategy": "ddp_find_unused_parameters_true",
            "accelerator": "gpu",
            "devices": cfg.trainer.devices,
            "use_distributed_sampler": True,
            "default_root_dir": work_dir,
            "sync_batchnorm": True,
        }
        
        # Log configuration
        log_rank_0(f"Training config for seed {cfg.seed}:\n{cfg}")
        log_rank_0(f"Git commit: {get_git_commit_hash(Path(hydra.utils.to_absolute_path(__file__)))}")
        log_rank_0(print_system_env_info())
                
        # Clear CUDA cache again before training
        clear_cuda_cache()
        
        # Initialize trainer and train
        trainer = Trainer(**trainer_args)
        
        try:
            trainer.fit(model, datamodule=datamodule)
        except Exception as e:
            log_rank_0("\nDetailed Error Information:")
            log_rank_0("=" * 80)
            log_rank_0(f"Error Type: {type(e).__name__}")
            log_rank_0(f"Error Message: {str(e)}")
            log_rank_0("\nFull Traceback:")
            import traceback
            log_rank_0(''.join(traceback.format_tb(e.__traceback__)))
            log_rank_0("\nLocal Variables at Crash Point:")
            tb = e.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            log_rank_0(f"{traceback.extract_tb(tb)}")
            log_rank_0("=" * 80)
            raise e
                
    except Exception as e:
        logger.error(f"\nTraining failed for seed {cfg.seed}:")
        logger.error(f"{'='*80}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        import traceback
        logger.error(traceback.format_exc())
        logger.error(f"{'='*80}")
        raise e
    finally:
        # Clear CUDA cache one final time
        clear_cuda_cache()
        # Clean up
        cleanup_distributed()
        if wandb.run is not None:
            wandb.finish()

def cleanup_distributed():
    """Cleanup distributed training resources"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # Set environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = 'True'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    # Add repo to path
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
    
    try:
        train()
    except Exception as e:
        logger.error(f"\nTraining script failed:")
        logger.error(f"{'='*80}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        import traceback
        logger.error(traceback.format_exc())
        logger.error(f"{'='*80}")
        sys.exit(1)
