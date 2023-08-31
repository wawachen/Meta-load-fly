import numpy as np
import torch
from torch import nn
import torch.distributions as D

from torch_utils import log_gaussian_prob, kl_regularization


class GaussianLatentObject(torch.nn.Module):
    def __init__(self, params):
        super(GaussianLatentObject, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device("cpu")

        self._init_params_to_attrs(params)
        self._init_setup()


    def _init_params_to_attrs(self, params):
        self._num_latent_classes = params.num_latent_classes[0]  # latent classes
        self._latent_dim = params.latent_dim[0]

        self._beta_kl = params.beta_kl

        known_latent_mu = params.known_latent_default_mu[0]
        known_latent_log_sigma = params.known_latent_default_log_sigma[0]

        known_latent_mu = np.zeros((self._num_latent_classes, self._latent_dim))
        known_latent_log_sigma = np.zeros((self._num_latent_classes, self._latent_dim))

        self._latent_default_mu = np.broadcast_to(np.array(known_latent_mu, dtype=np.float32),
                                                  (self._num_latent_classes, self._latent_dim))
        self._latent_default_log_sigma = np.broadcast_to(np.array(known_latent_log_sigma, dtype=np.float32),
                                                         (self._num_latent_classes, self._latent_dim))

        # mean and log_sigma midpoints are used for the online mean
        self._online_latent_default_mu = self._latent_default_mu.mean(axis=0)
        self._online_latent_default_log_sigma = np.zeros(self._latent_dim, dtype=np.float32)

    def _init_setup(self):
        # TODO these might get overriden by model restore...
        # noinspection PyArgumentList
        self.online_mu = nn.Parameter(torch.from_numpy(self._online_latent_default_mu).to(self.device))
        # noinspection PyArgumentList
        self.online_log_sigma = nn.Parameter(torch.from_numpy(self._online_latent_default_log_sigma).to(self.device))

        start_mu = []
        start_log_sig = []
        for i in range(self._num_latent_classes):
            start_mu.append(np.ones(self._latent_dim, dtype=np.float32) * self._latent_default_mu[i])
            start_log_sig.append(np.ones(self._latent_dim, dtype=np.float32) * self._latent_default_log_sigma[i])
        
        assert(self._num_latent_classes==4)
        self.mu_0 = nn.Parameter(torch.from_numpy(start_mu[0]).to(self.device))
        self.log_sigma_0 = nn.Parameter(torch.from_numpy(start_log_sig[0]).to(self.device))

        self.mu_1 = nn.Parameter(torch.from_numpy(start_mu[1]).to(self.device))
        self.log_sigma_1 = nn.Parameter(torch.from_numpy(start_log_sig[1]).to(self.device))

        self.mu_2 = nn.Parameter(torch.from_numpy(start_mu[2]).to(self.device))
        self.log_sigma_2 = nn.Parameter(torch.from_numpy(start_log_sig[2]).to(self.device))

        self.mu_3 = nn.Parameter(torch.from_numpy(start_mu[3]).to(self.device))
        self.log_sigma_3 = nn.Parameter(torch.from_numpy(start_log_sig[3]).to(self.device))

    def get_online_latent_mu_logsig(self):
        return self.online_mu, self.online_log_sigma

    def get_latent_mu_logsig(self):
        mus, logsigs = [self.mu_0,self.mu_1,self.mu_2,self.mu_3], [self.log_sigma_0,self.log_sigma_1,self.log_sigma_2,self.log_sigma_3]
        return mus, logsigs

    def reset_online_latent_mu_logsig(self):
        self.online_mu.data = torch.from_numpy(self._online_latent_default_mu).to(self.device)
        self.online_log_sigma.data = torch.from_numpy(self._online_latent_default_log_sigma).to(self.device)

    def loss(self, inputs, outputs, get_model_out, logger, i=0):
        distributions = self(inputs)
        inputs['latent'] = distributions['sample']
        model_outputs = get_model_out(inputs)
        loss, logprob, kl = self._get_latent_loss(distributions['mu'], distributions['log_sigma'],
                                     model_outputs['next_obs'], model_outputs['next_obs_sigma'],
                                     outputs['next_obs'].unsqueeze(-2))

        logger.add_scalar("latent_loss", loss.item(), i)
        logger.add_scalar("latent_loss_kl", kl.item(), i)
        logger.add_scalar("latent_loss_logprob", logprob.item(), i)

        for j in range(self._latent_dim):
            logger.add_scalar("online_latent_mu_dim=%d" % j, self.online_mu[j].item(), i)
            logger.add_scalar("online_latent_log_sigma_dim=%d" % j, self.online_log_sigma[j].item(), i)

        mus, logsigs = self.get_latent_mu_logsig()
        for d, (mu, lsig) in enumerate(zip(mus, logsigs)):
            for j in range(self._latent_dim):
                logger.add_scalar("latent_%d_mu_dim=%d" % (d, j), mu[j].item(), i)
                logger.add_scalar("latent_%d_log_sigma_dim=%d" % (d, j), lsig[j].item(), i)

        return loss

    def _get_latent_loss(self, mu_lat, logs_lat, mu_next_obs, sigma_next_obs, targ_next_obs):
        log_prob = log_gaussian_prob(mu_next_obs, sigma_next_obs, targ_next_obs)  # P(s' | s, a, z)
        kl = kl_regularization(mu_lat, logs_lat)  # KL(q_phi || N(0,1))

        return - log_prob + self._beta_kl * kl, log_prob, kl

    def forward(self, inputs, obs_lowd=None):
        """
        Given inputs, map them to the appropriate latent distribution

        :param inputs (AttrDict): holds obs, prev_obs, prev_act, latent and "act"
        :param training (bool):
        :return: AttrDict: parametrizes distribution of latents, holds mu, log_sigma
        """

        # assert inputs['latent'].dtype in [torch.short, torch.int, torch.long], \
        #     "Latent is type: " + str(inputs['latent'].type())
        orig = inputs['latent'].view(inputs['latent'].shape[0])  # should be (batch, 1)
        # map latent classes to mu, log_sig
        mus = []
        log_sigs = []
        for latent_class in orig:
            # -1 class specifies online inference
            if latent_class.item() == -1:
                mus.append(self.online_mu)
                log_sigs.append(self.online_log_sigma)
            else:
                if latent_class == 0:
                    mus.append(self.mu_0)
                    log_sigs.append(self.log_sigma_0)
                if latent_class == 1:
                    mus.append(self.mu_1)
                    log_sigs.append(self.log_sigma_1)
                if latent_class == 2:
                    mus.append(self.mu_2)
                    log_sigs.append(self.log_sigma_2)
                if latent_class == 3:
                    mus.append(self.mu_3)
                    log_sigs.append(self.log_sigma_3)

        mu = torch.stack(mus)
        log_sigma = torch.stack(log_sigs)

        # torch_distribution = D.normal.Normal(loc=mu, scale=log_sigma.exp())
        # sample = torch_distribution.rsample()  # sample from latent diagonal gaussian (reparam trick for gradient)
        sample = mu + torch.randn_like(mu) * log_sigma.exp()

        return {
            'mu': mu,
            'log_sigma': log_sigma,
            'sample': sample
        }


