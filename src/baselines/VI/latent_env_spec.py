import numpy as np

class LatentEnvSpec:
    def __init__(self, params):
        self.names_shapes_limits_dtypes = params
        self.names_shapes_limits_dtypes += [('done', (), (False, True), np.bool)]

        self.names_to_shapes = {}
        self.names_to_limits = {}
        self.names_to_dtypes = {}
        for name, shape, limit, dtype in self.names_shapes_limits_dtypes:
            self.names_to_shapes[name] = shape
            self.names_to_limits[name] = limit
            self.names_to_dtypes[name] = dtype
    
    @property
    def observation_names(self):
        """
        Returns:
            list(str)
        """
        return ['obs', 'prev_obs', 'prev_act', 'latent']

    @property
    def output_observation_names(self):
        return ['next_obs', 'next_obs_sigma']

    @property
    def goal_names(self):
        """
        The only difference between a goal and an observation is that goals are user-specified

        Returns:
            list(str)
        """
        return ['goal_obs']

    @property
    def action_names(self):
        """
        Returns:
            list(str)
        """
        return ['act']
