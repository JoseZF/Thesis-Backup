import ray
from ESTorchAlgo import ESConfig, ES, EvolutionNN, MyCallbacks
from ray.rllib.models import ModelCatalog

# Get config
config = ESConfig()
# Register Model
ModelCatalog.register_custom_model("torch_model", EvolutionNN)
# Create a Algorithm
algo = ES(env="CarRacing-v2", config={
    "env_config" : {"continuous": False},
    "framework": "torch",
    "episodes_per_batch": 5,
    "pop_size": 20,
    "noise_std": 0.001,
    "mutation": "normal",
    "sampler": "residual",
    "num_workers": 8,
    "callbacks": MyCallbacks,
    "num_gpus": 0,
    "model": {  
    "custom_model": "torch_model",
    # "custom_model_config": {},
    },
})
# Train
if __name__ == "__main__":
    for i in range(50):
        algo.train()