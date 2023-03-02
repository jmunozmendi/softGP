import gpytorch
import torch


class AGP(gpytorch.models.ApproximateGP):
    def __init__(self, num_inputs: int, num_tasks: int, num_inducing: int = 128) -> None:
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(num_inputs, num_inducing, num_inputs)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_inputs])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_inputs,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_inputs]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RQKernel(ard_num_dims = num_inputs, batch_shape=torch.Size([num_inputs])),
            batch_shape=torch.Size([num_inputs])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    


class DGPHiddenLayer(gpytorch.models.deep_gps.DeepGPLayer):
    def __init__(self, input_dims: int, output_dims: int, num_inducing: int = 128, mean_type: str = 'constant') -> None:
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        if mean_type == 'constant':
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = gpytorch.means.LinearMean(input_dims)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RQKernel(ard_num_dims = input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    




class DGP1(gpytorch.models.deep_gps.DeepGP):
    def __init__(self, num_inputs: int , num_tasks: int, num_hidden_dgp_dims: int = 10) -> None:
        hidden_layer = DGPHiddenLayer(
            input_dims=num_inputs,
            output_dims=num_hidden_dgp_dims,
            mean_type='linear'
        )
        
        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_tasks,
            mean_type='constant'
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer

        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, test_x):
        with torch.no_grad():

            # The output of the model is a multitask MVN, where both the data points
            # and the tasks are jointly distributed
            # To compute the marginal predictive NLL of each data point,
            # we will call `to_data_independent_dist`,
            # which removes the data cross-covariance terms from the distribution.
            preds = self.likelihood(self(test_x)).to_data_independent_dist()

        return preds.mean.mean(0), preds.variance.mean(0)
    



class DGP2(gpytorch.models.deep_gps.DeepGP):
    def __init__(self, num_inputs: int, num_tasks: int, num_hidden_dgp_dims: int = 10) -> None:
        hidden_layer = DGPHiddenLayer(
            input_dims=num_inputs,
            output_dims=num_hidden_dgp_dims,
            mean_type='linear'
        )
        
        second_hidden_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_hidden_dgp_dims,
             mean_type='linear'
        ) 
        
        
        last_layer = DGPHiddenLayer(
            input_dims=second_hidden_layer.output_dims,
            output_dims=num_tasks,
            mean_type='constant'
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.second_hidden_layer = second_hidden_layer
        self.last_layer = last_layer

        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        hidden_rep2 = self.second_hidden_layer(hidden_rep1)
        output = self.last_layer(hidden_rep2)
        return output

    def predict(self, test_x):
        with torch.no_grad():

            # The output of the model is a multitask MVN, where both the data points
            # and the tasks are jointly distributed
            # To compute the marginal predictive NLL of each data point,
            # we will call `to_data_independent_dist`,
            # which removes the data cross-covariance terms from the distribution.
            preds = self.likelihood(self(test_x)).to_data_independent_dist()

        return preds.mean.mean(0), preds.variance.mean(0)
    




class DGP3(gpytorch.models.deep_gps.DeepGP):
    def __init__(self, num_inputs: int, num_tasks: int, num_hidden_dgp_dims: int = 10) -> None:
        hidden_layer = DGPHiddenLayer(
            input_dims=num_inputs,
            output_dims=num_hidden_dgp_dims,
            mean_type='linear'
        )
        
        second_hidden_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_hidden_dgp_dims,
             mean_type='linear'
        ) 
        
        third_hidden_layer = DGPHiddenLayer(
            input_dims=second_hidden_layer.output_dims,
            output_dims=num_hidden_dgp_dims,
             mean_type='linear'
        ) 
        
        
        last_layer = DGPHiddenLayer(
            input_dims=third_hidden_layer.output_dims,
            output_dims=num_tasks,
            mean_type='constant'
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.second_hidden_layer = second_hidden_layer
        self.third_hidden_layer = third_hidden_layer
        self.last_layer = last_layer

        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        hidden_rep2 = self.second_hidden_layer(hidden_rep1)
        hidden_rep3 = self.third_hidden_layer(hidden_rep2)
        output = self.last_layer(hidden_rep3)
        return output

    def predict(self, test_x):
        with torch.no_grad():

            # The output of the model is a multitask MVN, where both the data points
            # and the tasks are jointly distributed
            # To compute the marginal predictive NLL of each data point,
            # we will call `to_data_independent_dist`,
            # which removes the data cross-covariance terms from the distribution.
            preds = self.likelihood(self(test_x)).to_data_independent_dist()

        return preds.mean.mean(0), preds.variance.mean(0)
    





class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, num_tasks: int, num_inputs: int, train_x: torch.Tensor, train_y: torch.Tensor , likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RQKernel(ard_num_dims = num_inputs), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)