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




# [motor 1, motor 2, motor 3, pitch, yaw]
meansArm = np.array([-0.652753, -0.035599, 0.634844, -1.911363, 0.100071])
stdsArm = np.array([118.812632, 113.574488, 113.688108, 21.753780, 26.242482])

# [M1, M2 ,M3, Inclination, Orientation]
meansNeck = np.array([0.422745, 0.388243, 0.331195, 24.728547, 177.387627])
stdsNeck = np.array([2.635532, 2.595948, 2.598489, 12.534071, 106.165696])



class softGP:

    def __init__(self, kinematics: str, type: str, device: str) -> None:

        if kinematics not in ["IK", "FK"]:
            raise ValueError("kinematics must be FK or IK")

        if type not in ["AGP", "DGP1", "DGP2", "DGP3", "ExactGP"]:
            raise ValueError("model must be AGP, DGP1, DGP2, DGP3 or ExactGP")
        
        if device not in ["Arm", "Neck"]:
            raise ValueError("device must be Arm or Neck")

        self.kinematics = kinematics
        self.type = type
        self.device = device

        dir = 'soft_gp/softGP/models/' + device + '/'

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


        if self.device == "Arm":

            meanValues = meansArm
            stdValues = stdsArm

        else:

            meanValues = meansNeck
            stdValues = stdsNeck

        if input_data.ndim != 2:
            raise ValueError("input data must have 2 dimensions")

        num_rows, num_cols = input_data.shape

        if self.kinematics == "IK" and num_cols != 2:

            raise ValueError("number of columns must be 2")

        if self.kinematics == "FK" and num_cols != 3:

            raise ValueError("number of columns must be 3")
        
        if self.kinematics == "FK":

            normalized_data = (input_data - meanValues[:3])/stdValues[:3]

        else:

            normalized_data = (input_data - meanValues[3:5])/stdValues[3:5]

        test_x: torch.Tensor = torch.Tensor(normalized_data).cuda() if torch.cuda.is_available() else torch.Tensor(normalized_data)

        test_dataset = TensorDataset(test_x, test_x)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        means = []
        variances = []
        lowers = []
        uppers = []


        match self.type:

            case "AGP":

                with torch.no_grad():

                    for x_batch, y_batch in test_loader:

                        predictions = self.likelihood(self.model(x_batch))
                        mean = predictions.mean
                        lower, upper = predictions.confidence_region()  
                        var = predictions.variance

                        means.append(mean.cpu().numpy())
                        variances.append(var.cpu().numpy())
                        lowers.append(lower.cpu().numpy())
                        uppers.append(upper.cpu().numpy())


                means = np.concatenate(means, axis=0)
                variances = np.concatenate(variances, axis=0)
                lowers = np.concatenate(lowers, axis=0)
                uppers = np.concatenate(uppers, axis=0)
                stddev = np.sqrt(variances)

                if self.kinematics == "FK":

                    predictive_means = means*stdValues[3:5] + meanValues[3:5]
                    predictive_stds = stddev*stdValues[3:5] + meanValues[3:5]
                    predictive_vars = variances*stdValues[3:5] + meanValues[3:5]

                else:

                    predictive_means = means*stdValues[:3] + meanValues[:3]             
                    predictive_stds = stddev*stdValues[:3] + meanValues[:3] 
                    predictive_vars = variances*stdValues[:3] + meanValues[:3] 


            case "DGP1":

                with torch.no_grad():

                    for x_batch, y_batch in test_loader:

                        mean, var = self.model.predict(x_batch)
                        lower = mean - 2 * var.sqrt()
                        upper = mean + 2 * var.sqrt()

                        means.append(mean.cpu().numpy())
                        variances.append(var.cpu().numpy())
                        lowers.append(lower.cpu().numpy())
                        uppers.append(upper.cpu().numpy())

                means = np.concatenate(means, axis=0)
                variances = np.concatenate(variances, axis=0)
                lowers = np.concatenate(lowers, axis=0)
                uppers = np.concatenate(uppers, axis=0)
                stddev = np.sqrt(variances)

                if self.kinematics == "FK":

                    predictive_means = means*stdValues[3:5] + meanValues[3:5]
                    predictive_stds = stddev*stdValues[3:5] + meanValues[3:5]
                    predictive_vars = variances*stdValues[3:5] + meanValues[3:5]

                else:

                    predictive_means = means*stdValues[:3] + meanValues[:3]             
                    predictive_stds = stddev*stdValues[:3] + meanValues[:3] 
                    predictive_vars = variances*stdValues[:3] + meanValues[:3]                     

            case "DGP2":

                with torch.no_grad():

                    for x_batch, y_batch in test_loader:

                        mean, var = self.model.predict(x_batch)
                        lower = mean - 2 * var.sqrt()
                        upper = mean + 2 * var.sqrt()

                        means.append(mean.cpu().numpy())
                        variances.append(var.cpu().numpy())
                        lowers.append(lower.cpu().numpy())
                        uppers.append(upper.cpu().numpy())

                means = np.concatenate(means, axis=0)
                variances = np.concatenate(variances, axis=0)
                lowers = np.concatenate(lowers, axis=0)
                uppers = np.concatenate(uppers, axis=0)
                stddev = np.sqrt(variances)

                if self.kinematics == "FK":

                    predictive_means = means*stdValues[3:5] + meanValues[3:5]
                    predictive_stds = stddev*stdValues[3:5] + meanValues[3:5]
                    predictive_vars = variances*stdValues[3:5] + meanValues[3:5]

                else:

                    predictive_means = means*stdValues[:3] + meanValues[:3]             
                    predictive_stds = stddev*stdValues[:3] + meanValues[:3] 
                    predictive_vars = variances*stdValues[:3] + meanValues[:3]  

            case "DGP3":

                with torch.no_grad():

                    for x_batch, y_batch in test_loader:

                        mean, var = self.model.predict(x_batch)
                        lower = mean - 2 * var.sqrt()
                        upper = mean + 2 * var.sqrt()

                        means.append(mean.cpu().numpy())
                        variances.append(var.cpu().numpy())
                        lowers.append(lower.cpu().numpy())
                        uppers.append(upper.cpu().numpy())

                means = np.concatenate(means, axis=0)
                variances = np.concatenate(variances, axis=0)
                lowers = np.concatenate(lowers, axis=0)
                uppers = np.concatenate(uppers, axis=0)
                stddev = np.sqrt(variances)

                if self.kinematics == "FK":

                    predictive_means = means*stdValues[3:5] + meanValues[3:5]
                    predictive_stds = stddev*stdValues[3:5] + meanValues[3:5]
                    predictive_vars = variances*stdValues[3:5] + meanValues[3:5]

                else:

                    predictive_means = means*stdValues[:3] + meanValues[:3]             
                    predictive_stds = stddev*stdValues[:3] + meanValues[:3] 
                    predictive_vars = variances*stdValues[:3] + meanValues[:3]  

            case "ExactGP":

                with torch.no_grad():

                    for x_batch, y_batch in test_loader:

                        predictions = self.likelihood(self.model(test_x))
                        mean = predictions.mean
                        lower, upper = predictions.confidence_region()  
                        var = predictions.variance

                        means.append(mean.cpu().numpy())
                        variances.append(var.cpu().numpy())
                        lowers.append(lower.cpu().numpy())
                        uppers.append(upper.cpu().numpy())

                means = np.concatenate(means, axis=0)
                variances = np.concatenate(variances, axis=0)
                lowers = np.concatenate(lowers, axis=0)
                uppers = np.concatenate(uppers, axis=0)
                stddev = np.sqrt(variances)

                if self.kinematics == "FK":

                    predictive_means = means*stdValues[3:5] + meanValues[3:5]
                    predictive_stds = stddev*stdValues[3:5] + meanValues[3:5]
                    predictive_vars = variances*stdValues[3:5] + meanValues[3:5]

                else:

                    predictive_means = means*stdValues[:3] + meanValues[:3]             
                    predictive_stds = stddev*stdValues[:3] + meanValues[:3] 
                    predictive_vars = variances*stdValues[:3] + meanValues[:3] 

        return softGPInfo(mean=predictive_means,
                    deviation=predictive_stds,
                    variance=predictive_vars,
                    )