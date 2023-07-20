import hydra
from omegaconf import DictConfig
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datasets.ModelNet40Ply2048 import ModelNet40Ply2048DataModule
from model import Adapt_classf_pl
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WANDB_API_KEY"] = "04a5d6fba030b76e5b620f5bd6509cf7dffebb8b"

def train(cfg, train_loader, test_loader):

    device = "cuda" if cfg.cuda else "cpu"
    if cfg.model.name == "Adapt_classf":
        model = Adapt_classf_pl(cfg, cfg.model.embed_dim, cfg.n_points, cfg.n_classes, cfg.model.n_blocks, cfg.model.groups)
    else:
        raise Exception("Model not supported")
    
    if cfg.wandb:
        wandb_logger = WandbLogger(name=cfg.experiment.name, project=cfg.experiment.project)
        wandb_logger.watch(model)
        wandb_logger.log_hyperparams(cfg)
        wandb_logger.log_hyperparams(model.hparams)

    trainer = pl.Trainer(profiler="simple",max_epochs=cfg.train.epochs, accelerator=device, logger=[wandb_logger] if cfg.wandb else None, devices=1, gradient_clip_val=2)
    trainer.fit(model, train_loader, test_loader)



    return None

def test(cfg, test_loader):
    raise NotImplementedError

def visualize(cfg):
    raise NotImplementedError

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.experiment.seed)
    if cfg.wandb:
        wandb.login()
        wandb.init(config=cfg)

    cfg.cuda = cfg.cuda and torch.cuda.is_available()
    if cfg.cuda:
        print(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(cfg.experiment.seed)
    else:
        print('Using CPU')

    if cfg.experiment.dataset == "ModelNet40":
        dataset = ModelNet40Ply2048DataModule(batch_size=cfg.train.batch_size)
    else:
        raise Exception("Dataset not supported")
    dataset.setup()
    train_loader = dataset.train_dataloader()
    test_loader = dataset.val_dataloader()
    cfg.n_classes = dataset.num_classes
    cfg.n_points = dataset.num_points

    if not cfg.eval:
        train(cfg, train_loader, test_loader)
    else:
        if not cfg.visualize_pc:
            test(cfg, test_loader)
        else:
            visualize(cfg)

if __name__ == "__main__":
    main()