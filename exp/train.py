import torch
torch.set_float32_matmul_precision('high')

from ml4opf import ACProblem
from ml4opf.models.ac_pgpfva.model import AC_PGPFVA_NeuralNet

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


logger = [
    TensorBoardLogger(save_dir="logs/dl", name="dl"),
    WandbLogger(project="dl", log_model=False, save_dir="logs/dl", entity="..."),
]

problem = ACProblem("data/1354_pegase")

config = {
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "loss": "mse",
    "hidden_sizes": [2048, 2048, 2048],
    "activation": "softplus",
    "boundrepair": "none"
}

model = AC_PGPFVA_NeuralNet(config, problem)

model.train(trainer_kwargs={
        'max_epochs': 200,
        'accelerator':'auto',
        'logger':logger
    },
    dataset_kwargs={
        'dl_kwargs': {
            "batch_size": 32,
            "num_workers": 1
        },
        'combos': {
            "input": ["input/pd"],
            "target": ["primal/pg", "primal/pf", "primal/va"]
        },
        'order': ["input", "target"]
    },
    compile_kwargs={},
)

print(f"Saving checkpoint to {model.trainer.logger.log_dir}...")
model.save_checkpoint(model.trainer.logger.log_dir)