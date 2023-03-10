import torch
import numpy as np
import pandas as pd

from torch import Tensor
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.optim import NGD
from gpytorch.mlls import VariationalELBO
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from ..core import Logger
from .use_cases import AGP, DGP1, DGP2, DGP3, ExactGP
from ..entities import softGPInfo

dir = 'soft_gp/softGP/models/'


# [motor 1, motor 2, motor 3, pitch, yaw]
means = np.array([-0.652753, -0.035599, 0.634844, -1.911363, 0.100071])
stds = np.array([118.812632, 113.574488, 113.688108, 21.753780, 26.242482])


class softGP:

    def __init__(self, kinematics: str, type: str) -> None:

        if kinematics not in ["IK", "FK"]:
            raise ValueError("kinematics must be FK or IK")

        if type not in ["AGP", "DGP1", "DGP2", "DGP3", "ExactGP"]:
            raise ValueError("model must be AGP, DGP1, DGP2, DGP3 or ExactGP")

        self.kinematics = kinematics
        self.type = type

        if kinematics == "IK":

            num_inputs = 2
            num_tasks = 3

        else:
    
            num_inputs = 3
            num_tasks = 2


        match type:

            case "AGP":

                self.model: AGP = AGP(num_inputs=num_inputs, num_tasks=num_tasks)
                self.likelihood: MultitaskGaussianLikelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)
                state_dict = torch.load(dir + kinematics + '/' + type + '/likelihood.pth',map_location=torch.device('cpu'))
                self.likelihood.load_state_dict(state_dict)
                state_dict = torch.load(dir + kinematics + '/' + type + '/model.pth',map_location=torch.device('cpu'))
                self.model.load_state_dict(state_dict)
                
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                    self.likelihood = self.likelihood.cuda()

            case "DGP1":

                self.model: DGP1 = DGP1(num_inputs=num_inputs, num_tasks=num_tasks)
                state_dict = torch.load(dir + kinematics + '/' + type + '/model.pth',map_location=torch.device('cpu'))
                self.model.load_state_dict(state_dict)

                if torch.cuda.is_available():
                    self.model = self.model.cuda()

            case "DGP2":

                self.model: DGP2 = DGP2(num_inputs=num_inputs, num_tasks=num_tasks)
                state_dict = torch.load(dir + kinematics + '/' + type + '/model.pth',map_location=torch.device('cpu'))
                self.model.load_state_dict(state_dict)

                if torch.cuda.is_available():
                    self.model = self.model.cuda()

            case "DGP3":

                self.model: DGP3 = DGP3(num_inputs=num_inputs, num_tasks=num_tasks)
                state_dict = torch.load(dir + kinematics + '/' + type + '/model.pth',map_location=torch.device('cpu'))
                self.model.load_state_dict(state_dict)

                if torch.cuda.is_available():
                    self.model = self.model.cuda()

            case "ExactGP":

                train = pd.read_csv(dir + kinematics + '/' + type + '/train.csv') 

                if kinematics == "FK":

                    train_x = torch.Tensor(train[['motor1','motor2','motor3']].values).contiguous()
                    train_y = torch.Tensor(train[['pitch','yaw']].values).contiguous()

                else:

                    train_y = torch.Tensor(train[['motor1','motor2','motor3']].values).contiguous()
                    train_x = torch.Tensor(train[['pitch','yaw']].values).contiguous()

                self.likelihood: MultitaskGaussianLikelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)
                state_dict = torch.load(dir + kinematics + '/' + type + '/likelihood.pth',map_location=torch.device('cpu'))
                self.likelihood.load_state_dict(state_dict)
                self.model: ExactGP = ExactGP(num_tasks=num_tasks, num_inputs=num_inputs, train_x=train_x, train_y=train_y, likelihood=MultitaskGaussianLikelihood(num_tasks=num_tasks))
                state_dict = torch.load(dir + kinematics + '/' + type + '/model.pth',map_location=torch.device('cpu'))
                self.model.load_state_dict(state_dict)

                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                    self.likelihood = self.likelihood.cuda()

                self.model.eval()
                self.likelihood.eval()





    def predict(self, input_data: np.ndarray) -> softGPInfo:

        if input_data.ndim != 2:
            raise ValueError("input data must have 2 dimensions")

        num_rows, num_cols = input_data.shape

        if self.kinematics == "IK" and num_cols != 2:

            raise ValueError("number of columns must be 2")

        if self.kinematics == "FK" and num_cols != 3:

            raise ValueError("number of columns must be 3")
        
        if self.kinematics == "FK":

            normalized_data = (input_data - means[:3])/stds[:3]

        else:

            normalized_data = (input_data - means[3:5])/stds[3:5]

        test_x: torch.Tensor = torch.Tensor(normalized_data).cuda() if torch.cuda.is_available() else torch.Tensor(normalized_data)


        match self.type:

            case "AGP":

                predictions = self.likelihood(self.model(test_x))

                if self.kinematics == "FK":

                    predictive_means = predictions.mean.detach().cpu().numpy()*stds[3:5] + means[3:5]
                    predictive_stds = predictions.stddev.detach().cpu().numpy()*stds[3:5] + means[3:5]
                    predictive_vars = predictions.variance.detach().cpu().numpy()*stds[3:5] + means[3:5]

                else:

                    predictive_means = predictions.mean.detach().cpu().numpy()*stds[:3] + means[:3]             
                    predictive_stds = predictions.stddev.detach().cpu().numpy()*stds[:3] + means[:3] 
                    predictive_vars = predictions.variance.detach().cpu().numpy()*stds[:3] + means[:3] 


            case "DGP1":

                mean, var = self.model.predict(test_x)
                std = np.sqrt(var)

                if self.kinematics == "FK":

                    predictive_means = mean.numpy()*stds[3:5] + means[3:5]
                    predictive_stds = std.numpy()*stds[3:5] + means[3:5]
                    predictive_vars = var.numpy()*stds[3:5] + means[3:5]

                else:

                    predictive_means = mean.numpy()*stds[:3] + means[:3]             
                    predictive_stds = std.numpy()*stds[:3] + means[:3] 
                    predictive_vars = var.numpy()*stds[:3] + means[:3]                     

            case "DGP2":

                mean, var = self.model.predict(test_x)
                std = np.sqrt(var)

                if self.kinematics == "FK":

                    predictive_means = mean.numpy()*stds[3:5] + means[3:5]
                    predictive_stds = std.numpy()*stds[3:5] + means[3:5]
                    predictive_vars = var.numpy()*stds[3:5] + means[3:5]

                else:

                    predictive_means = mean.numpy()*stds[:3] + means[:3]             
                    predictive_stds = std.numpy()*stds[:3] + means[:3] 
                    predictive_vars = var.numpy()*stds[:3] + means[:3]  

            case "DGP3":

                mean, var = self.model.predict(test_x)
                std = np.sqrt(var)

                if self.kinematics == "FK":

                    predictive_means = mean.numpy()*stds[3:5] + means[3:5]
                    predictive_stds = std.numpy()*stds[3:5] + means[3:5]
                    predictive_vars = var.numpy()*stds[3:5] + means[3:5]

                else:

                    predictive_means = mean.numpy()*stds[:3] + means[:3]             
                    predictive_stds = std.numpy()*stds[:3] + means[:3] 
                    predictive_vars = var.numpy()*stds[:3] + means[:3]  

            case "ExactGP":

                predictions = self.likelihood(self.model(test_x))

                if self.kinematics == "FK":

                    predictive_means = predictions.mean.detach().cpu().numpy()*stds[3:5] + means[3:5]
                    predictive_stds = predictions.stddev.detach().cpu().numpy()*stds[3:5] + means[3:5]
                    predictive_vars = predictions.variance.detach().cpu().numpy()*stds[3:5] + means[3:5]

                else:

                    predictive_means = predictions.mean.detach().cpu().numpy()*stds[:3] + means[:3]             
                    predictive_stds = predictions.stddev.detach().cpu().numpy()*stds[:3] + means[:3] 
                    predictive_vars = predictions.variance.detach().cpu().numpy()*stds[:3] + means[:3] 

        return softGPInfo(mean=predictive_means,
                    deviation=predictive_stds,
                    variance=predictive_vars,
                    )