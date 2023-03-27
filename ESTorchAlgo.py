from collections import namedtuple, Counter, defaultdict, OrderedDict
import logging
import numpy as np
import random
import time
import gym
from typing import Optional
from encoder import Autoencoder
import torchvision.transforms as transforms

import ray
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.es import optimizers, utils
from ray.rllib.algorithms.es.es_tf_policy import ESTFPolicy, rollout
from ray.rllib.env.env_context import EnvContext
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils import FilterManager
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_TRAINED,
)
from ray.rllib.utils.torch_utils import set_torch_seed
from ray.rllib.utils.typing import AlgorithmConfigDict

logger = logging.getLogger(__name__)

from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

from ray.rllib.algorithms.callbacks import DefaultCallbacks

class MyCallbacks(DefaultCallbacks):
    
    @override(DefaultCallbacks)
    def on_train_result(
        self,
        *,
        algorithm: Optional["Algorithm"] = None,
        result: dict,
        trainer=None,
        **kwargs,
    ) -> None:
        """Called at the end of Trainable.train().
        Args:
            algorithm: Current trainer instance.
            result: Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        if trainer is not None:
            algorithm = trainer

        if result['perf']['cpu_util_percent'] and result['perf']['gpu_util_percent0']:
            print(f"Iter: %.i;    mean_reward=%.2f,    max_reward=%.2f,   N_eff=%.2f,    episode_duration=%.2f s,   cpu_percent:%.2f,  gpu_percent:%.2f"\
             % (result['iterations_since_restore'], result['episode_reward_mean'], result['episode_reward_max'], result['n_eff'], result['episode_duration'],\
                result['perf']['cpu_util_percent'], result['perf']['gpu_util_percent0']))
        
        elif result['perf']['gpu_util_percent0']:
            print(f"Iter: %.i;    mean_reward=%.2f,    max_reward=%.2f,   N_eff=%.2f,    episode_duration=%.2f s,    gpu_percent:%.2f"\
             % (result['iterations_since_restore'], result['episode_reward_mean'], result['episode_reward_max'], result['n_eff'], result['episode_duration'],\
                result['perf']['gpu_util_percent0']))

        elif result['perf']['cpu_util_percent']:
            print(f"Iter: %.i;    mean_reward=%.2f,    max_reward=%.2f,   N_eff=%.2f,    episode_duration=%.2f s,   cpu_percent:%.2f"\
             % (result['iterations_since_restore'], result['episode_reward_mean'], result['episode_reward_max'], result['n_eff'], result['episode_duration'],\
                result['perf']['cpu_util_percent']))

        else:
            print(f"Iter: %.i;    mean_reward=%.2f,    max_reward=%.2f,   N_eff=%.2f,    episode_duration=%.2f s"\
             % (result['iterations_since_restore'], result['episode_reward_mean'], result['episode_reward_max'], result['n_eff'], result['episode_duration']))


class ESConfig(AlgorithmConfig):
    """Defines a configuration class from which an ES Algorithm can be built.
    Example:
        >>> from ray.rllib.algorithms.es import ESConfig
        >>> config = ESConfig().training(sgd_stepsize=0.02, report_length=20)\
        ...     .resources(num_gpus=0)\
        ...     .rollouts(num_rollout_workers=4)
        >>> print(config.to_dict())
        >>> # Build a Algorithm object from the config and run 1 training iteration.
        >>> trainer = config.build(env="CartPole-v1")
        >>> trainer.train()
    Example:
        >>> from ray.rllib.algorithms.es import ESConfig
        >>> from ray import tune
        >>> config = ESConfig()
        >>> # Print out some default values.
        >>> print(config.action_noise_std)
        >>> # Update the config object.
        >>> config.training(rollouts_used=tune.grid_search([32, 64]), eval_prob=0.5)
        >>> # Set the config object's env.
        >>> config.environment(env="CartPole-v1")
        >>> # Use to_dict() to get the old-style python config dict
        >>> # when running with tune.
        >>> tune.Tuner(
        ...     "ES",
        ...     run_config=ray.air.RunConfig(stop={"episode_reward_mean": 200}),
        ...     param_space=config.to_dict(),
        ... ).fit()
    """

    def __init__(self):
        """Initializes a ESConfig instance."""
        super().__init__(algo_class=ES)

        # fmt: off
        # __sphinx_doc_begin__

        # ES specific settings:
        self.pop_size = 100
        self.noise_std = 0.01
        self.episodes_per_batch = 1000
        self.report_length = 10
        self.mutation = "normal"
        self.sampler = "universal"
        self.initial_pop = None
        # Override some of AlgorithmConfig's default values with ES-specific values.
        # self.train_batch_size = 1
        self.num_workers = 10
        self.callbacks = MyCallbacks
        self.observation_filter = "NoFilter"
        self.evaluation_config["num_envs_per_worker"] = 1
        self.evaluation_config["observation_filter"] = "NoFilter"

        # __sphinx_doc_end__
        # fmt: on

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        pop_size: Optional[int] = None,
        noise_std: Optional[float] = None,
        episodes_per_batch: Optional[int] = None,
        report_length: Optional[int] = None,
        sampler: Optional[str] = None,
        mutation: Optional[str] = None,
        initial_pop: Optional[dict] = None,
        **kwargs,
    ) -> "ESConfig":
        """Sets the training related configuration.
        Args:
            action_noise_std: Std. deviation to be used when adding (standard normal)
                noise to computed actions. Action noise is only added, if
                `compute_actions` is called with the `add_noise` arg set to True.
            l2_coeff: Coefficient to multiply current weights with inside the globalg
                optimizer update term.
            noise_stdev: Std. deviation of parameter noise.
            episodes_per_batch: Minimum number of episodes to pack into the train batch.
            eval_prob: Probability of evaluating the parameter rewards.
            stepsize: SGD step-size used for the Adam optimizer.
            noise_size: Number of rows in the noise table (shared across workers).
                Each row contains a gaussian noise value for each model parameter.
            report_length: How many of the last rewards we average over.
        Returns:
            This updated AlgorithmConfig object.
        """
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)
        
        if pop_size is not None:
            self.pop_size = pop_size
        if noise_std is not None:
            self.noise_std = noise_std
        if episodes_per_batch is not None:
            self.episodes_per_batch = episodes_per_batch
        if report_length is not None:
            self.report_length = report_length
        if mutation is not None:
            self.mutation = mutation
        if sampler is not None:
            self.sampler = sampler
        if initial_pop is not None:
            self.initial_pop = initial_pop
        return self
    
def get_flat_weights(policy):
        # Get the parameter tensors.
        theta_dict = policy.model.state_dict()
        # Flatten it into a single np.ndarray.
        theta_list = []
        for k in sorted(theta_dict.keys()):
            theta_list.append(torch.reshape(theta_dict[k], (-1,)))
        cat = torch.cat(theta_list, dim=0)
        return cat.cpu().numpy()
    
def create_weights_dict(policy, pop_size):
    w_flat = get_flat_weights(policy)
    matrix =  np.random.rand(pop_size, len(w_flat))
    weights_dict = {idx + 1 : matrix[idx] for idx in range(len(matrix))}
    return weights_dict

@ray.remote
class Worker:
    def __init__(
        self,
        config,
        policy_params,
        env_creator,
        worker_index,
        min_task_runtime=0.2,
    ):

        # Set Python random, numpy, env, and torch/tf seeds.
        seed = config.get("seed")
        if seed is not None:
            # Python random module.
            random.seed(seed)
            # Numpy.
            np.random.seed(seed)
            # Torch.
            if config.get("framework") == "torch":
                set_torch_seed(seed)
        self.seed = seed
        self.min_task_runtime = min_task_runtime
        self.config = config
        self.config.update(policy_params)
        self.config["single_threaded"] = True

        env_context = EnvContext(config["env_config"] or {}, worker_index)
        self.env = env_creator(env_context)

        from ray.rllib import models

        self.preprocessor = models.ModelCatalog.get_preprocessor(
            self.env, config["model"]
        )

        _policy_class = get_policy_class(config)
        self.policy = _policy_class(
            self.env.observation_space, self.env.action_space, config
        )

    @property
    def filters(self):
        return {DEFAULT_POLICY_ID: self.policy.observation_filter}

    def sync_filters(self, new_filters):
        for k in self.filters:
            self.filters[k].sync(new_filters[k])

    def get_filters(self, flush_after=False):
        return_filters = {}
        for k, f in self.filters.items():
            return_filters[k] = f.as_serializable()
            if flush_after:
                f.reset_buffer()
        return return_filters

    def rollout(self, env_seed, timestep_limit: Optional[int] = None):
        # Compute a simulation episode.
        rewards = []
        t = 0
        max_timestep_limit = 999999
        env_timestep_limit = (self.env.spec.max_episode_steps if (hasattr(self.env, "spec") and hasattr(self.env.spec, "max_episode_steps")) else max_timestep_limit)
        timestep_limit = (env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit))
        for i in range(self.config["episodes_per_batch"]):
            obs = self.env.reset(seed=env_seed)
            reward = 0
            for _ in range(timestep_limit or max_timestep_limit): 
                act, _, __ = self.policy.compute_single_action(obs)
                # act, _, __ = self.policy.compute_actions(obs)
                obs, r , done, _ = self.env.step(act)
                reward += r 
                t +=1
                if done:
                    break
            rewards.append(reward)
        rewards = np.mean(rewards)
        return rewards, t

    def do_rollouts(self, params, env_seed, timestep_limit = None):
        # Set the network weights.
        self.policy.set_flat_weights(params)
        rewards, length = self.rollout(env_seed)
        result = (rewards, length)
        return result
        


def get_policy_class(config):
    if config["framework"] == "torch":
        from ESTorchPolicy import ESTorchPolicy

        policy_cls = ESTorchPolicy
    else:
        policy_cls = ESTFPolicy
    return policy_cls

@ray.remote(num_gpus=2)
def covariance(matrix, aweights):
    warnings.filterwarnings("ignore")
    matrix = matrix.T
    matrix_gpu = torch.from_numpy(matrix).to(device)
    aweights_gpu = torch.from_numpy(aweights).to(device)
    torch.cuda.synchronize()
    cov_matrix = torch.cov(matrix_gpu, correction=0, aweights=aweights_gpu)
    torch.cuda.synchronize()
    return cov_matrix


class ES(Algorithm):
    """Large-scale implementation of Evolution Strategies in Ray."""

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return ESConfig().to_dict()

    @override(Algorithm)
    def validate_config(self, config: AlgorithmConfigDict) -> None:
        # Call super's validation method.
        super().validate_config(config)

        if config["num_gpus"] > 1:
            raise ValueError("`num_gpus` > 1 not yet supported for ES!")
        if config["num_workers"] <= 0:
            raise ValueError("`num_workers` must be > 0 for ES!")
        if config["evaluation_config"]["num_envs_per_worker"] != 1:
            raise ValueError(
                "`evaluation_config.num_envs_per_worker` must always be 1 for "
                "ES! To parallelize evaluation, increase "
                "`evaluation_num_workers` to > 1."
            )
        if config["evaluation_config"]["observation_filter"] != "NoFilter":
            raise ValueError(
                "`evaluation_config.observation_filter` must always be "
                "`NoFilter` for ES!"
            )

    @override(Algorithm)
    def setup(self, config):
        # Setup our config: Merge the user-supplied config (which could
        # be a partial config dict with the class' default).
        if isinstance(config, dict):
            self.config = self.merge_trainer_configs(
                self.get_default_config(), config, self._allow_unknown_configs
            )
        else:
            self.config = config.to_dict()

        # Call super's validation method.
        self.validate_config(self.config)

        # Generate the local env.
        env_context = EnvContext(self.config["env_config"] or {}, worker_index=0)
        env = self.env_creator(env_context)

        self.callbacks = self.config["callbacks"]()
        self.pop_size = self.config["pop_size"]
        self.noise_std = self.config["noise_std"] 
        self._policy_class = get_policy_class(self.config)
        self.policy = self._policy_class(
            obs_space=env.observation_space,
            action_space=env.action_space,
            config=self.config,
        )
        self.report_length = self.config["report_length"]

        # Create weights table
        logger.info("Creating population.")
        if self.config["initial_pop"]:
            self.weights_dict = self.config["initial_pop"]
        else:
            self.weights_dict = create_weights_dict(self.policy, self.pop_size)

        # Create the actors.
        logger.info("Creating actors.")
        self.workers = [
            Worker.remote(self.config, {}, self.env_creator, idx + 1)
            for idx in range(self.config["num_workers"])
        ]

        self.episodes_so_far = 0
        self.reward_list = []
        self.tstart = time.time()

    @override(Algorithm)
    def get_policy(self, policy=DEFAULT_POLICY_ID):
        if policy != DEFAULT_POLICY_ID:
            raise ValueError(
                "ES has no policy '{}'! Use {} "
                "instead.".format(policy, DEFAULT_POLICY_ID)
            )
        return self.policy
    
    @override(Algorithm)
    def step(self):
        # perform a step in the training
        config = self.config

       # Collect results from rollouts
        fitness, lengths, num_episodes, num_timesteps = self._collect_results()
        # Update our sample steps counters.
        self._counters[NUM_AGENT_STEPS_SAMPLED] += num_timesteps
        self._counters[NUM_ENV_STEPS_SAMPLED] += num_timesteps
        self.episodes_so_far += num_episodes
        # Assemble the results.
        fitness = np.array(fitness)
        print(fitness)
        mean_fitness = np.mean(fitness)
        max_fitness = np.max(fitness)
        std_fitness = np.std(fitness)
        eval_lengths = np.array(lengths)
        # Resampler Selection
        if self.config["sampler"] == "universal":
            survivors = self.uni_sampler(fitness)
        elif self.config["sampler"] == "residual":
            survivors = self.res_sampler(fitness)
        else:
            raise Exception("Resampler not supported")
        # Mutation Selection
        if self.config["mutation"] == "normal":
            self.noise_adaptation(mean_fitness)
            self.mutate(survivors)
        elif self.config["mutation"] == "covariance":
            self.covariance_sampling(survivors)
        else:
            raise Exception("Mutation model not supported") 
        # Store the rewards
        self.reward_list.append(mean_fitness)
        # Calculate N-eff
        norm_fitness = fitness/np.sum(fitness)
        N_eff = 1/(np.sum(norm_fitness**2))
        # Define Callbacks to be return at the end of trainer.train()
        info = {
            "episodes_so_far": self.episodes_so_far,
        }

        result = dict(
            episode_reward_mean=mean_fitness,
            episode_reward_std=std_fitness,
            episode_reward_max=max_fitness,
            n_eff=N_eff,
            episode_duration=round(time.time()-self.start, 2),
            episode_len_mean=eval_lengths.mean(),
            info=info,
        )

        return result
    
    @override(Algorithm)
    def compute_single_action(self, observation, *args, **kwargs):
        action, _, _ = self.policy.compute_actions([observation], update=False)
        if kwargs.get("full_fetch"):
            return action[0], [], {}
        return action[0]

    @override(Algorithm)
    def cleanup(self):
        # workaround for https://github.com/ray-project/ray/issues/1516
        for w in self.workers:
            w.__ray_terminate__.remote()

    def _collect_results(self):
        # get results from rollouts
        self.start = time.time()
        num_episodes, num_timesteps = 0, 0
        num_workers = self.config["num_workers"]
        results = []
        seed = np.random.randint(0,10000)
        
        rollout_ids = [{i:self.workers[(i-1) % num_workers].do_rollouts.remote(self.weights_dict[i], seed) for i in range(1, len(self.weights_dict)+1)}]
        
        # Get the results of the rollouts.
        res = {}
        for i in rollout_ids:
            res.update(i)
        rollout_ids = dict(sorted(res.items()))
        rollout_values = [ray.get(rollout_id) for rollout_id in rollout_ids.values()]
        for result in rollout_values:
            results.append(result)
        fitness = [x[0] for x in results]
        lengths = [x[1] for x in results]
        num_episodes = len(lengths)
        num_timesteps = sum(lengths)
        return fitness, lengths, num_episodes, num_timesteps


    def __getstate__(self):
        return {
            "algorithm_class": type(self),
            "config": self.config,
            "weights": self.weights_dict,
        }

    def __setstate__(self, state):
        self.weights_dict = state["weights"]
        self.config = state["config"]


    # Resamplers
    def uni_sampler(self, fitness):
        # Stochastic Universal Resampler
        fitness = np.array(fitness)
        min_fitness = np.min(fitness)
        if min_fitness < 0:
            fitness += np.abs(min_fitness)
        # Select random number from 0 to  1/N
        selected = np.random.uniform(0, 1/self.pop_size)
        # Create a list with the location of the selected particles
        points = []
        points.append(selected)
        for i in range(self.pop_size - 1):
            selected += 1/self.pop_size
            points.append(selected)
        # Normalize fitness between 0 and 1 
        zero_one_fitness = fitness/np.sum(fitness)
        norm_fitness = np.cumsum(zero_one_fitness)
        # See where the selected points fall in the fitness scale
        survivors = []
        for point in points:
            for i in range(self.pop_size):  
                if point <= norm_fitness[i]:
                    survivors.append(i)
                    break
        # Organize survivors into a dictionary with: 
        # key: sample number, and value: number of times it was selected
        survivor_dict = {}
        for survivor in survivors:
            if survivor+1 in survivor_dict:
                survivor_dict[survivor+1] += 1
            else:
                survivor_dict[survivor+1] = 1
        return survivor_dict 
    
    def res_sampler(self, fitness):
        # Residual Resampler
        min_fitness = np.min(fitness)
        if min_fitness < 0:
            fitness += np.abs(min_fitness)
        norm_fitness = fitness/np.sum(fitness)
        z = self.pop_size * norm_fitness
        int_z = []
        survivors = []
        # Get integer values of the resampler
        for count, value in enumerate(z):
            value = int(value)
            int_z.append(value)
            if value > 0:
                for i in range(value):
                    survivors.append(count)
        #Get float values in weighted distribution
        float_z = z - int_z
        norm_float_z = np.cumsum(float_z/np.sum(float_z))
        dim = round(np.sum(float_z))
        #Randomly select points from distribution
        points = np.random.uniform(0,1, dim)
        points.sort()
        for point in points:
            for i in range(self.pop_size):  
                if point <= norm_float_z[i]:
                    survivors.append(i)
                    break
        # Organize survivors into a dictionary with: 
        # key: sample number, and value: number of times it was selected
        survivor_dict = {}
        for survivor in survivors:
            if survivor+1 in survivor_dict:
                survivor_dict[survivor+1] += 1
            else:
                survivor_dict[survivor+1] = 1
        return survivor_dict

    "add new resampler here"
    
    # Mutations
    def mutate(self, survivors):
        # Mutate new population from survivors
        # Delete the non-survivors
        keys = []
        for i in range(1, self.pop_size+1):
            if i not in survivors.keys():
                keys.append(i)
                del self.weights_dict[i]
        # Create new samples
        new_samples = []
        for key, value in survivors.items():
            if value != 1:
                for i in range(value-1):
                    weights = self.weights_dict[key]
                    noise = np.random.normal(0, self.noise_std, self.weights_dict[key].shape)
                    sample = weights + noise
                    new_samples.append(sample)
        # Add the new samples back to the weight dict
        for key, new_sample in zip(keys, new_samples):
            self.weights_dict[key] = new_sample
    
    def noise_adaptation(self, mean_fitness):
        self.noise_std *= 0.7
        # try:
        #     if self.prev_fitness >= mean_fitness:
        #         self.noise_std *= 0.5 
        #     self.prev_fitness = mean_fitness
        # except:
        #     self.prev_fitness = mean_fitness

    
    def covariance_sampling(self, survivors):
        # Apply Covariance Matrix Adaptation to the survivors
        w_dict = survivors
        # get wi from the list of survivors from Universal Resampler. wi=z/N
        for key in w_dict:
            w_dict[key] /= self.pop_size
        # Count the number of models to be sampled from gaussian 
        N_sampled = self.pop_size - len(w_dict)
        # Delete not selected models
        for i in range(1, self.pop_size+1):
            if i not in w_dict.keys():
                del self.weights_dict[i]
        # Calculate Weighed mean of Gaussian
        mean_matrix = []
        w_all = np.array(list(w_dict.values()))
        weights_table = np.array(list(self.weights_dict.values()))
        for i in range(weights_table.shape[1]):
            mean_i = np.dot(np.array(w_all), weights_table[:,i])
            mean_matrix.append(mean_i)
        mean_matrix = np.array(mean_matrix)
        # Calculate weighted Covariance mean of Gaussian
        bessel = 1 / (1 - np.sum([wi ** 2 for wi in w_all]))
        # Calculate Covariance
        if self.config["num_gpus"] > 1:
            cov_matrix = covariance.remote(weights_table, aweights=w_all)
            cov_matrix = ray.get(cov_matrix)
            cov_matrix = cov_matrix.cpu().numpy()
            torch.cuda.empty_cache()
        else:
            # cov_matrix = {}
            # for i in range(weights_table.shape[1]):
            #     cov_matrix[i] = (np.dot(w_all, weights_table[:, i] - mean_matrix[i]) / weights_table.shape[0]*2)
            # cov_matrix = np.array(list(cov_matrix.values()))
            cov_matrix = np.cov(weights_table, rowvar=False, aweights=w_all)
            cov_matrix = np.array(cov_matrix)
        cov_matrix *= bessel
        # Regularization
        eigenvalue = 10**(-10)
        cov_matrix += eigenvalue*np.identity(weights_table.shape[1])
            # for i in range(N_sampled):
            #     eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            #     # Construct the diagonal matrix
            #     D = np.array(eigenvalues)
            #     noise = np.random.normal(0, 1, size=D.shape[0])
            #     random_vector = (D * noise) + mean_matrix
            #     noise_array = np.array(random_vector)
            #     weights_table = np.vstack((weights_table, noise_array))
        # LDLT decomposition
        L = np.linalg.cholesky(cov_matrix)
        # Sample new models and append them to weight matrix 
        if N_sampled > 0:
            for i in range(N_sampled):
                noise = np.random.normal(0, 1, size=L.shape[0])
                random_vector = (L @ noise) + mean_matrix
                noise_array = np.array(random_vector)
                weights_table = np.vstack((weights_table, noise_array))
        self.weights_dict = {}
        for i, row in enumerate(weights_table):
            self.weights_dict[i+1] = row
    
    # Method to get the mean weights of the population, calculate the reward and render
    @override(Algorithm)
    def evaluate(self, evaluation_runs = 10, render=True):
        # Turn weights_dict into weights_table
        weights_table = np.array([self.weights_dict[i] for i in range(1, len(self.weights_dict)+1)])
        mean_weights = np.mean(weights_table, axis=0)
        self.policy.set_flat_weights(mean_weights)
        rewards = {}
        len_episodes = []
        for i in range(evaluation_runs):
            obs = self.env.reset()
            t = 0
            reward=0
            if render:
                while True: 
                     # act = self.policy.compute_single_action(torch.FloatTensor(obs))
                    act = self.policy.compute_single_action(obs)
                    obs, r , done, _ = self.env.step(act)
                    reward += r 
                    t +=1
                    self.env.render()
                    if done:
                        self.env.close()
                        break
            else: 
                while True: 
                     # act = self.policy.compute_single_action(torch.FloatTensor(obs))
                    act = self.policy.compute_single_action(obs)
                    obs, r , done, _ = self.env.step(act)
                    reward += r 
                    t +=1
                    if done:
                        self.env.close()
                        break
            len_episodes.append(t)
            rewards.update({i:reward})
        mean = np.array(list(rewards.values())).mean()
        print(f"Mean Rewards = {mean} \n Rewards = {rewards}, episode lengths = {len_episodes}") 

# --------------------Encoder-------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Load the saved encoder part
loaded_encoder = torch.load('./model_weights/good_encoder.pth', map_location="cpu")
# Use the loaded encoder
encoder = Autoencoder().to(device)
encoder.encoder = loaded_encoder
# --------------------Encoder-------------------------

# Image transformation
transform = transforms.Compose([
# transforms.CenterCrop(70),
# transforms.Grayscale(1),
])


from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
#Subclass that inherits from pytorch's nn.Module and RlLib's TorchModelV2
class EvolutionNN(TorchModelV2, nn.Module):
    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.num_inputs = 1600
        # Neural Netwteork Structure
        self.net = nn.Sequential(
            nn.Linear(self.num_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_outputs),
        )
    # Forward Pass    
    def forward(self, input_dict, state, seq_lens):
        obs = transform(input_dict["obs"][0].permute(2, 0, 1)).to(device).float()
        obs = encoder.encoder(obs)
        obs = obs.flatten()
        obs = obs[None, :]
        obs = obs.expand(32,self.num_inputs).to(device).float()
        return self.net(obs), []
        # return self.net(torch.flatten(input_dict["obs"], 1, 3).float()), []