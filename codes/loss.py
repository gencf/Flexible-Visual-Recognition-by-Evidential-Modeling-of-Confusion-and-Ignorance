import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO: check if sum or mean is used in the loss functions


class custom_loss(nn.Module):
    def __init__(self,
                 max_epochs,
                 max_lambda_kl,
                 annealing_last_value,
                 n_classes,
                 lambda_reg=1):
        super(custom_loss, self).__init__()

        self.max_epochs = max_epochs
        self.max_lambda_kl = max_lambda_kl
        self.annealing_last_value = annealing_last_value
        self.n_classes = n_classes
        self.lambda_reg = lambda_reg

        self.lambda_kl = max_lambda_kl

        assert self.max_epochs > 0, "max_epochs must be greater than 0"
        assert self.max_lambda_kl >= 0, "max_lambda_kl must be greater than or equal to 0"
        assert self.annealing_last_value >= 0, "annealing_last_value must be greater than or equal to 0"
        assert self.lambda_reg >= 0, "lambda_reg must be greater than or equal to 0"



    def forward(self,
                plausibility,
                y_true,
                epoch=-1,
                return_losses_seperately=False,
                epsilon=1e-6):
        """
        Forward pass of the custom loss function
        """
        
        # calculate (1-plausibility) for each class
        inverse_plausibility = 1 - plausibility


        # calculate beliefs (class plausibility times product of inverse plausibilities of each other class)
        beliefs = torch.clone(plausibility)

        for i in range(1, self.n_classes):
            # shift the inverse plausibility tensor by i
            beliefs = beliefs * torch.roll(inverse_plausibility, shifts=i, dims=1)


        # calculate the uncertainity (1 - sum of beliefs)
        uncertainity = 1 - torch.sum(beliefs, dim=1)


        # calculate the ignorance (product of inverse plausibilities of each class)
        ignorance = torch.prod(inverse_plausibility, dim=1)


        # calculate the dirichlet parameters
        dirichlet_parameters = (beliefs * self.n_classes / (uncertainity + epsilon).repeat(self.n_classes, 1).T) + 1


        # calculate the EDL loss
        loss_EDL = self.calculate_EDL_loss(dirichlet_parameters, y_true)

        # calculate the REG loss
        loss_REG = self.calculate_REG_loss(plausibility, y_true, ignorance)

        # update the lambda kl
        if (epoch != -1):
            self.update_lambda_kl(epoch, self.max_epochs, self.max_lambda_kl)
        else:
            raise ValueError("Epoch and max_epochs must be provided to update lambda kl.")

        # calculate the KL loss
        loss_KL = self.calculate_KL_loss(dirichlet_parameters, y_true, plausibility)

        if return_losses_seperately:
            return torch.mean(loss_EDL), self.lambda_reg * torch.mean(loss_REG), self.lambda_kl * torch.mean(loss_KL)
        else:
            return torch.mean(loss_EDL) + self.lambda_reg * torch.mean(loss_REG) + self.lambda_kl * torch.mean(loss_KL)








    @staticmethod
    def calculate_EDL_loss(dirichlet_parameters, y_true):
        """
        Evidential Deep Learning (EDL) loss
        """

        # calculate the sum of the dirichlet parameters
        sum_dirichlet_parameters = torch.sum(dirichlet_parameters, dim=1, keepdim=True)


        # calculate the loss
        loss_EDL = y_true * (torch.log(sum_dirichlet_parameters) - torch.log(dirichlet_parameters))

        return torch.sum(loss_EDL, dim=1)

    @staticmethod
    def calculate_REG_loss(plausibilities, y_true, ignorance):
        """
        Regularization loss
        """
        # get the batch size and number of classes
        batch_size, n_classes = plausibilities.shape

        # calculate the loss
        loss_REG = y_true * torch.square(plausibilities - (1 - ignorance).repeat(n_classes, 1).T)

        return torch.sum(loss_REG, dim=1)
    
    @staticmethod
    def calculate_KL_loss(dirichlet_parameters, y_true, plausibilities):
        """
        Kullback-Leibler (KL) divergence loss

        As stated in the paper, it is taken from "Evidential Deep Learning to Quantify Classification uncertainity" (https://proceedings.neurips.cc/paper_files/paper/2018/file/a981f2b708044d6fb4a71a1463242520-Paper.pdf)
        """

        # change the true class label index of dirichlet parameters to 1
        dirichlet_parameters_for_KL = y_true + (1-y_true) * dirichlet_parameters



        # get the alpha sum 
        alpha_sum = torch.sum(dirichlet_parameters_for_KL, dim=1, keepdim=True)

        # create ones
        ones = torch.ones([1, dirichlet_parameters_for_KL.shape[-1]], dtype=torch.float32, device=dirichlet_parameters_for_KL.device)

        # calculate the log (...) part which includes gamma functions with logarithms
        loss_KL_log_gamma_part = torch.lgamma(alpha_sum) + \
                                 torch.lgamma(ones).sum(dim=1, keepdim=True) + \
                                 (- torch.lgamma(ones.sum(dim=1, keepdim=True))) + \
                                 (- torch.lgamma(dirichlet_parameters_for_KL).sum(dim=1, keepdim=True))
        
        
        # calculate the part with digamma functions
        loss_KL_digamma_part = ((dirichlet_parameters_for_KL - ones).mul(torch.digamma(dirichlet_parameters_for_KL) - torch.digamma(alpha_sum))).sum(dim=1, keepdim=True)


        return torch.sum(loss_KL_log_gamma_part + loss_KL_digamma_part, dim=1)


    def update_lambda_kl(self, epoch, max_epochs, max_lambda_kl=0.05, annealing_last_value=0.0):
        """
        lambda_kl anneals to 0 with epochs with the maximum coefficient of 0.05, and λreg is set to 1.
        """

        self.lambda_kl = max(max_lambda_kl * (1 - epoch / max_epochs), annealing_last_value)
        


def calculate_only_belief_uncertainity_ignorance_confusion(plausibilities):
    """
    Calculate only beliefs, uncertainity, ignorance, and confusion
    """
    # get the batch size and number of classes
    batch_size, n_classes = plausibilities.shape

    # calculate (1-plausibility) for each class
    inverse_plausibility = 1 - plausibilities


    # calculate beliefs (class plausibility times product of inverse plausibilities of each other class)
    beliefs = torch.clone(plausibilities)

    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                beliefs[:, i] = beliefs[:, i] * inverse_plausibility[:, j]


    # calculate the uncertainity (1 - sum of beliefs)
    uncertainity = 1 - torch.sum(beliefs, dim=1)


    # calculate the ignorance (product of inverse plausibilities of each class)
    ignorance = torch.prod(inverse_plausibility, dim=1)

    # calculate the confusion (uncertainity - ignorance)
    confusion = uncertainity - ignorance

    return beliefs, uncertainity, ignorance, confusion


if __name__ == "__main__":
    # create the custom loss
    loss = custom_loss(
        max_epochs=5,
        max_lambda_kl=0.05,
        annealing_last_value=0.0,
        lambda_reg=1
    )

    torch.set_printoptions(precision=4, sci_mode=False)

    # create a tensor of plausibility values (between 0 and 1)
    plausibilities = torch.tensor([[0.0, 0.9, 0.3], [0.0, 0.9, 0.2]], dtype=torch.float32)

    # create a tensor of true labels (one-hot)
    y_true = torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.float32)


    # calculate the loss
    for epoch in range(5):
        print(loss(plausibilities, y_true, epoch=epoch))
