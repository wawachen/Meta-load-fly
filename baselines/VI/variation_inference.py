"""
The trainer is where everything is combined.
"""
import math

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import os
import torch

def is_next_cycle(current, period):
    return period > 0 and current % period == 0

class LatentTrainer(object):

    def __init__(self, params, model, dataset_train,
                 dataset_holdout):
        self._model = model
        # self._policy = policy
        self._dataset_train = dataset_train
        self._dataset_holdout = dataset_holdout
        # self._env_train = env_train  # TODO put back in
        # self._env_holdout = env_holdout
        self._base_optimizer = optim.Adam(model.base_parameters(), lr=float(params.dynamics_learning_rate))
        self._latent_optimizer = optim.Adam(model.latent_parameters(), lr=float(params.latent_learning_rate))

        self._latent_train_every_n_steps = params.latent_train_every_n_steps  # 0 if fixed latents
        self._sample_every_n_steps = int(params.sample_every_n_steps)  # how often to sample relative to one loop
        self._train_every_n_steps = int(params.train_every_n_steps)  # how often to train relative to one loop
        self._holdout_every_n_steps = int(params.holdout_every_n_steps)

        self._max_steps = int(params.max_steps)  # how many overall loops
        self._max_train_data_steps = int(params.max_train_data_steps)  # how many samples to collect & add
        self._max_holdout_data_steps = int(params.max_holdout_data_steps)  # how many samples to collect & add

        self._log_every_n_steps = int(params.log_every_n_steps)  # must be a multiple of _train_every_n_steps
        self._save_every_n_steps = int(params.save_every_n_steps)  # must be a multiple of _train_every_n_steps
        assert self._log_every_n_steps % self._train_every_n_steps == 0
        assert self._sample_every_n_steps % self._train_every_n_steps == 0

        self._save_checkpoints = params.save_checkpoints

        self._current_step = 0
        self._current_train_step = 0
        self._current_train_loss = math.inf
        self._current_holdout_step = 0
        self._current_holdout_loss = math.inf
        self._current_latent_step = 0
        self._current_latent_loss = math.inf

        self.models_dir = "/home/wawa/catkin_meta/src/MBRL_transport/VI_model"

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device("cpu")

        # self._reset_curr_episode()

    def run(self,logger):
        """
        This is the main loop:
            - gather data
            - train the model
            - save the model
            - log progress
        """

        # training loop
        while self._current_step <= self._max_steps:
            # NOTE: always have some form of timing so that you can find bugs
            if is_next_cycle(self._current_step, self._holdout_every_n_steps):
                holdout_loss = self._holdout_step(logger)

            if is_next_cycle(self._current_step, self._train_every_n_steps):
                loss, latent_loss = self._train_step(logger)

                # self._writer.writerow([str(self._current_train_loss), str(self._current_holdout_loss)])

                if self._current_step > 0 and is_next_cycle(self._current_step, self._log_every_n_steps):
                    self._log()

                if self._current_step > 0 and is_next_cycle(self._current_step, self._save_every_n_steps):
                    self._save()

            self._current_step += 1

    def _save(self):
        base_fname = "model.pt"
        if self._save_checkpoints:
            base_fname = "chkpt_{:07d}.pt".format(self._current_step)
        path = os.path.join(self.models_dir, base_fname)
        torch.save({'step': self._current_step,
                    'train_step': self._current_train_step,
                    'holdout_step': self._current_holdout_step,
                    'latent_step': self._current_latent_step,
                    'train_loss': self._current_train_loss,
                    'holdout_loss': self._current_holdout_loss,
                    'latent_loss': self._current_latent_loss,
                    'model': self._model.state_dict()}, path)

    def _log(self):
        print('[{}] (steps, loss) -> TRAIN: ({}, {}), LATENT: ({} {}), HOLDOUT: ({}, {})'
                    .format(self._current_step,
                            self._current_train_step, self._current_train_loss,
                            self._current_latent_step, self._current_latent_loss,
                            self._current_holdout_step, self._current_holdout_loss)
                    )

    def _get_current_train_data_step(self):
        # raise NotImplementedError
        return len(self._dataset_train)

    def _get_current_holdout_data_step(self):
        # raise NotImplementedError
        return len(self._dataset_holdout)

    def _train_step(self,logger):
        inputs, outputs = self._dataset_train.get_batch()
        self._model.train()
        loss = self._model.base_loss(inputs, outputs, logger, i=self._current_step)
        self._base_optimizer.zero_grad()
        loss.backward()
        self._base_optimizer.step()
        self._current_train_loss = loss.item()
        logger.add_scalar("train_step", self._current_train_step, self._current_step)
        self._current_train_step += 1

        latent_loss = None
        if is_next_cycle(self._current_train_step, self._latent_train_every_n_steps):
            # get another batch
            inputs, outputs = self._dataset_train.get_batch()
            latent_loss = self._model.latent_loss(inputs, outputs,logger, i=self._current_step)
            self._latent_optimizer.zero_grad()
            latent_loss.backward()
            self._latent_optimizer.step()
            self._current_latent_loss = latent_loss.item()
            logger.add_scalar("train_latent_step", self._current_latent_step, self._current_step)
            self._current_latent_step += 1

        return loss, latent_loss

    def _holdout_step(self,logger):
        inputs, outputs = self._dataset_holdout.get_batch()
        self._model.eval()
        loss = self._model.base_loss(inputs, outputs,logger,i=self._current_step)
        self._current_holdout_loss = loss.item()
        logger.add_scalar("holdout_step", self._current_holdout_step, self._current_step)
        self._current_holdout_step += 1

        return loss

    # def _env_step(self, env, dataset, obs, goal):
    #     # TODO implement online training
    #     action = self._policy.get_action(self._model, obs, goal)
    #     next_obs, next_goal, done = env.step(action)
    #     self._curr_episode_obs = AttrDict.leaf_combine_and_apply(
    #         [self._curr_episode_obs, next_obs], lambda vs: vs[0] + [vs[1]])
    #     self._curr_episode_actions = AttrDict.leaf_combine_and_apply(
    #         [self._curr_episode_actions, action], lambda vs: vs[0] + [vs[1]])
    #     self._curr_episode_goals = AttrDict.leaf_combine_and_apply(
    #         [self._curr_episode_goals, next_goal], lambda vs: vs[0] + [vs[1]])
    #     self._curr_episode_dones.append(done)
    #     if done:
    #         dataset.add_episode(self._curr_episode_obs, self._curr_episode_goals, self._curr_episode_actions,
    #                             self._curr_episode_dones)
    #         self._reset_curr_episode()
    #         next_obs, next_goal = env.reset()
    #     return next_obs, next_goal

