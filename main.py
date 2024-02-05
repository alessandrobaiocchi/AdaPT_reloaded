import hydra
from omegaconf import DictConfig
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datasets.ModelNet40Ply2048 import ModelNet40Ply2048DataModule
from model import Adapt_classf_pl
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import LearningRateMonitor
from point_transformer_cls import PCT_PL
from fvcore.nn import FlopCountAnalysis

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["WANDB_API_KEY"] = "04a5d6fba030b76e5b620f5bd6509cf7dffebb8b"

def train(cfg, train_loader, test_loader):

    device = "cuda" if cfg.cuda else "cpu"
    if cfg.model.name == "Adapt_classf":
        model = Adapt_classf_pl(cfg, cfg.model.embed_dim, cfg.n_points, cfg.n_classes, cfg.model.n_blocks, cfg.model.groups)
    elif cfg.model.name == "PCT_reproduce":
        model = PCT_PL()
    else:
        raise Exception("Model not supported")
    
    if cfg.wandb:
        wandb_logger = WandbLogger(name=cfg.experiment.name, project=cfg.experiment.project)
        wandb_logger.watch(model)
        wandb_logger.log_hyperparams(cfg)
        wandb_logger.log_hyperparams(model.hparams)
        lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(max_epochs=cfg.train.epochs, accelerator=device, logger=[wandb_logger] if cfg.wandb else None, devices=1, gradient_clip_val=2, callbacks=[lr_monitor] if cfg.wandb else None)#, default_root_dir='saved_models')
    trainer.fit(model, train_loader, test_loader)

    if cfg.wandb:
        wandb.finish()

    return None

def test(cfg, test_loader):
    raise NotImplementedError

def visualize(cfg):
    raise NotImplementedError

def eval_time(cfg,x):
    
    device = "cuda" if cfg.cuda else "cpu"
    x = torch.randn(cfg.train.batch_size, 512, 3)
    x = x.to(device)
    model = Adapt_classf_pl(cfg, cfg.model.embed_dim, cfg.n_points, cfg.n_classes, cfg.model.n_blocks, cfg.model.groups)
    model = model.to(device)
    model.eval()
    flops = FlopCountAnalysis(model, x)
    print(flops.total()/cfg.train.batch_size)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):

    torch.set_float32_matmul_precision('high')
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
        #eval_time(cfg, next(iter(test_loader))[0][:,:,:])
        train(cfg, train_loader, test_loader)
    else:
        if not cfg.visualize_pc:
            test(cfg, test_loader)
        else:
            visualize(cfg)

if __name__ == "__main__":
    main()









################################################## OLD CODE ##################################################
"""   
    with torch.no_grad():
        data, label = next(iter(test_loader))
        decisions = []
        model = model.to(device)
        data = data.to(device)
        for budg in range(cfg.train.n_budgets):
            _, decision = model(data, budg=budg)
            decisions.append(decision[-1].reshape(-1).cpu())
        
        print(decisions[0].shape)
        decisions = torch.stack(decisions, dim=0).sum(dim=0)
        print(decisions.shape)
        optimal = torch.zeros_like(decisions)
        targets = cfg.model.drop_rate[-1]*(torch.arange(cfg.train.n_budgets)/(cfg.train.n_budgets-1))
        random_decision = torch.zeros_like(decisions)
        for targ in targets:
            optimal[:int(targ*len(optimal))] += 1
            ind = random.sample(range(len(optimal)), int(targ*len(optimal)))
            random_decision[ind] += 1

        print(targets)
        print(decisions)
        print(optimal)
        optimal = optimal.cpu().numpy()
        decisions = decisions.cpu().numpy()
        optimal_histo = np.histogram(optimal, bins=cfg.train.n_budgets+1)
        decision_histo = np.histogram(decisions, bins=cfg.train.n_budgets+1)
        random_decision_histo = np.histogram(random_decision, bins=cfg.train.n_budgets+1)

        fig = plt.figure()
        plt.hist([optimal, decisions, random_decision], alpha=0.5, bins=cfg.train.n_budgets+1, label=["optimal", "decision", "random_decision"])
        plt.legend(loc='upper right')
        plt.savefig("histo.png")
"""