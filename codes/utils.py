import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from codes.loss import custom_loss
from codes.loss import calculate_only_belief_uncertainity_ignorance_confusion

def get_n_classes_2d_toy_dataset(n_classes: int = 3,
                                 n_samples_per_class: int = 500,
                                 std: float = 4.0,
                                 distance: float = 9.0,
                                 random_seed: int = 42,
                                 shuffle: bool = True):
    """
    Get n classes 2D toy dataset

    Args:
        n_classes: int = 3: number of classes
        n_samples_per_class: int = 500: number of samples per class
        std: float = 4.0: standard deviation
        distance: float = 9.0: distance between classes

    Returns:
        X: np.ndarray: input data
        y: np.ndarray: target data (labels)
    """

    # set the random seed
    random.seed(random_seed)

    # define the X and y
    X = np.zeros((n_classes * n_samples_per_class, 2))
    y = np.zeros(n_classes * n_samples_per_class, dtype='uint8')

    # find the center of the classes (center of the center class is (0, 0), and the distance between classes is 9)

    centers = np.zeros((n_classes, 2))


    for i in range(n_classes):
        # the first class is in the upper
        angle = (np.pi * 2 / n_classes * i) + np.pi / 2
        centers[i] = (np.cos(angle) * distance, np.sin(angle) * distance)
    
    # sample from the normal distribution with the given standard deviation
    normal_dist_samples = np.random.randn(n_classes * n_samples_per_class, 2) * std

    # create the samples from the centers
    for i in range(n_classes):
        X[i * n_samples_per_class:(i + 1) * n_samples_per_class] = normal_dist_samples[i * n_samples_per_class:(i + 1) * n_samples_per_class] + centers[i]
        y[i * n_samples_per_class:(i + 1) * n_samples_per_class] = i
    
    # add the noise with normal distribution
    X += np.random.randn(n_classes * n_samples_per_class, 2) * 0.5

    if shuffle:
        # shuffle the data
        idx = np.arange(n_classes * n_samples_per_class)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

    # return the data
    return X, y


def plot_2d_toy_dataset(X, y, n_classes: int = 3, save_path=""):
    """
    Plot 2D toy dataset

    Args:
        X: np.ndarray: input data
        y: np.ndarray: target data (labels)
        n_classes: int = 3: number of classes
    """

    plt.figure()
    for i in range(n_classes):
        plt.scatter(X[y == i, 0], X[y == i, 1], label=f'class {i}')
    plt.legend()

    # save the plot
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_losses_and_accuracies(losses_train,
                               accuracies_train,
                               losses_validation,
                               accuracies_validation,
                               title,
                               save_path=""):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(losses_train, label='Train')
    ax[0].plot(losses_validation, label='Validation')
    ax[0].set_title('Loss')
    ax[0].legend()

    ax[1].plot(accuracies_train, label='Train')
    ax[1].plot(accuracies_validation, label='Validation')
    ax[1].set_title('Accuracy')
    ax[1].legend()

    fig.suptitle(title)

    # save the plot
    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_boundries(model,
                   device,
                   X,
                   y,
                   n_classes: int = 3,
                   plot_step: float = 0.02,
                   x_min: float = -20,
                   x_max: float = 20,
                   y_min: float = -20,
                   y_max: float = 20,
                   save_path=""):
    """
    Plot the decision boundaries of the model

    Args:
        model: nn.Module: the model
        X: np.ndarray: input data
        y: np.ndarray: target data (labels)
        n_classes: int = 3: number of classes
        plot_step: float = 0.02: plot step
        x_min: float = -20: minimum x value
        x_max: float = 20: maximum x value
        y_min: float = -20: minimum y value
        y_max: float = 20: maximum y value
    """


    # create the meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    
    # create the input data
    X_grid = np.c_[xx.ravel(), yy.ravel()]


    # set the model to evaluation mode
    model.eval()

    # get the predictions
    with torch.no_grad():
        y_pred = (model.to(device))(torch.tensor(X_grid, dtype=torch.float32).to(device)).cpu().numpy()
    
        # plot the decision boundaries
        plt.figure()


        # plot the decision boundaries
        y_pred = y_pred.argmax(axis=1).reshape(xx.shape)
        plt.contourf(xx, yy, y_pred, cmap=plt.cm.Spectral, alpha=0.8)

        # plot the data points
        for i in range(n_classes):
            plt.scatter(X.to("cpu")[y.to("cpu").argmax(axis=1) == i][:, 0],
                        X.to("cpu")[y.to("cpu").argmax(axis=1) == i][:, 1],
                        label=f"class {i}")

        plt.legend()
        # save the plot
        if save_path:
            plt.savefig(save_path)
        
        plt.show()

        # load back the model to the CPU
        model = model.to("cpu")



def plot_boundries_per_class(model,
                             device,
                             X,
                             y,
                             n_classes: int = 3,
                             plot_step: float = 0.02,
                             x_min: float = -20,
                             x_max: float = 20,
                             y_min: float = -20,
                             y_max: float = 20,
                             save_path=""):
    """
    Plot the decision boundaries of the model

    Args:
        model: nn.Module: the model
        X: np.ndarray: input data
        y: np.ndarray: target data (labels)
        n_classes: int = 3: number of classes
        plot_step: float = 0.02: plot step
        x_min: float = -20: minimum x value
        x_max: float = 20: maximum x value
        y_min: float = -20: minimum y value
        y_max: float = 20: maximum y value
    """


    # create the meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    
    # create the input data
    X_grid = np.c_[xx.ravel(), yy.ravel()]


    # set the model to evaluation mode
    model.eval()

    # get the predictions
    with torch.no_grad():

        # create figure
        plt.figure(figsize=(15, 5), layout = 'constrained')

        y_pred = (model.to(device))(torch.tensor(X_grid, dtype=torch.float32).to(device)).cpu().numpy()


        for i in range(n_classes):
            # plot the decision boundaries
            y_pred_class = y_pred[:, i].reshape(xx.shape)

            plt.subplot(1, n_classes, i+1)
            plt.contourf(xx, yy, y_pred_class, cmap=plt.cm.Spectral, alpha=0.8)

            plt.title(f"class {i}")

            # plot the data points
            for i in range(n_classes):
                plt.scatter(X[y.argmax(axis=1) == i][:, 0],
                            X[y.argmax(axis=1) == i][:, 1],
                            label=f"class {i}")



        plt.legend()
        # save the plot
        if save_path:
            plt.savefig(save_path)
        
        plt.show()

        # load back the model to the CPU
        model = model.to("cpu")



def plot_ignorance_uncertainity_confusion(model,
                                 device,
                                 X,
                                 y,
                                 n_classes: int = 3,
                                 plot_step: float = 0.02,
                                 x_min: float = -20,
                                 x_max: float = 20,
                                 y_min: float = -20,
                                 y_max: float = 20,
                                 save_path=""):
    """
    Plot the decision boundaries of the model

    Args:
        model: nn.Module: the model
        X: np.ndarray: input data
        y: np.ndarray: target data (labels)
        n_classes: int = 3: number of classes
        plot_step: float = 0.02: plot step
        x_min: float = -20: minimum x value
        x_max: float = 20: maximum x value
        y_min: float = -20: minimum y value
        y_max: float = 20: maximum y value
    """
    
    # create the meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    
    # create the input data
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # move the data and model to the device
    X_grid_tensor = torch.tensor(X_grid, dtype=torch.float32).to(device)
    model.to(device)

    # set the model to evaluation mode
    model.eval()

    # get the predictions
    with torch.no_grad():
        y_pred = model(X_grid_tensor)
        

        # get the beliefs, uncertainity, ignorance, confusion
        beliefs, uncertainity, ignorance, confusion = calculate_only_belief_uncertainity_ignorance_confusion(plausibilities=y_pred)



    
    # plot the decision boundaries for uncertainity, ignorance, and confusion
    plt.figure(figsize=(15, 5), layout = 'constrained')



    # plot the uncertainity
    plt.subplot(1, 3, 1)
    uncertainity = uncertainity.reshape(xx.shape)

    plt.imshow(uncertainity, cmap='viridis', alpha=0.8, extent=(x_min, x_max, y_min, y_max))

    # plot the data points
    for i in range(n_classes):
        plt.scatter(X[y.argmax(axis=1) == i][:, 0],
                    X[y.argmax(axis=1) == i][:, 1],
                    label=f"class {i}")
        
    plt.title("uncertainity")

    
    # plot the ignorance
    plt.subplot(1, 3, 2)
    ignorance = ignorance.reshape(xx.shape)

    plt.imshow(ignorance, cmap='viridis', alpha=0.8, extent=(x_min, x_max, y_min, y_max))

    # plot the data points
    for i in range(n_classes):
        plt.scatter(X[y.argmax(axis=1) == i][:, 0],
                    X[y.argmax(axis=1) == i][:, 1],
                    label=f"class {i}")
        
    plt.title("Ignorance")

    # plot the confusion
    plt.subplot(1, 3, 3)
    confusion = confusion.reshape(xx.shape)

    plt.imshow(confusion, cmap='viridis', alpha=0.8, extent=(x_min, x_max, y_min, y_max))

    # plot the data points
    for i in range(n_classes):
        plt.scatter(X[y.argmax(axis=1) == i][:, 0],
                    X[y.argmax(axis=1) == i][:, 1],
                    label=f"class {i}")
        
    plt.title("Confusion")

    # add the legend
    plt.legend()

    # add the colorbar to the plots (without making the plots smaller
    plt.colorbar()
    





    # save the plot
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

    # load back the model to the CPU
    model.to("cpu")




def train_model_for_toy_dataset(
    model,
    optimizer,
    loss_fn,
    device,
    X_train,
    y_train,
    X_validation,
    y_validation,
    n_epochs = 200000,
    verbose = True,
    verbose_every = 10000,
):
    # check the type of the loss function
    if isinstance(loss_fn, custom_loss):
        our_loss = True
    else:
        our_loss = False
    
    # create the loss function for the classic loss
    loss_fn = loss_fn.to(device)

    # train the MLP with the classic loss
    losses_train_classic = []
    accuracy_train_classic = []

    losses_validation_classic = []
    accuracy_validation_classic = []

    # move the data to the device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_validation = X_validation.to(device)
    y_validation = y_validation.to(device)


    # move the MLP to the device
    model.to(device)

    # set the MLP to training mode
    model.train()

    # get the train and test set sizes
    n_train = X_train.shape[0]
    n_validation = X_validation.shape[0]

    for epoch in range(n_epochs):

        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X_train)

        # calculate the loss
        if our_loss:
            loss = loss_fn(y_pred, y_train, epoch)
        else:
            loss = loss_fn(y_pred, y_train)

        # backward pass
        loss.backward()

        # update the weights
        optimizer.step()

        # store the loss
        losses_train_classic.append(loss.detach().cpu())

        # calculate the training accuracy
        accuracy_train = torch.sum(y_pred.argmax(dim=1) == y_train.argmax(dim=1)).detach().cpu() / n_train
        accuracy_train_classic.append(accuracy_train)

        # validate the model
        with torch.no_grad():
            # set the MLP to evaluation mode
            y_pred_validation = model(X_validation)

            # calculate the loss
            if our_loss:
                loss_validation = loss_fn(y_pred_validation, y_validation, epoch)
            else:
                loss_validation = loss_fn(y_pred_validation, y_validation)

            # store the loss
            losses_validation_classic.append(loss_validation.detach().cpu())

            # calculate the accuracy
            accuracy_validation = torch.sum(y_pred_validation.argmax(dim=1) == y_validation.argmax(dim=1)).detach().cpu() / n_validation

            accuracy_validation_classic.append(accuracy_validation)
            
        if verbose:
            if (epoch+1) % verbose_every == 0:
                # print the losses and accuracies
                print(f"Epoch {epoch}:")
                print(f"Train loss: {loss.item()}, Train accuracy: {accuracy_train}")
                print(f"Validation loss: {loss_validation.item()}, Validation accuracy: {accuracy_validation}")
                print("")

    # move everything back to the CPU
    X_train = X_train.to("cpu")
    y_train = y_train.to("cpu")
    X_validation = X_validation.to("cpu")
    y_validation = y_validation.to("cpu")
    model.to("cpu")
    y_pred = y_pred.to("cpu")
    y_pred_validation = y_pred_validation.to("cpu")
    loss_fn = loss_fn.to("cpu")
        
    return losses_train_classic, accuracy_train_classic, losses_validation_classic, accuracy_validation_classic




if __name__ == '__main__':
    # set the random seed
    random_seed = 42
    np.random.seed(random_seed)

    X, y = get_n_classes_2d_toy_dataset(n_classes=3,
                                        n_samples_per_class=500,
                                        std=2.0,
                                        distance=9.0,
                                        random_seed=random_seed,
                                        shuffle=True)
    plot_2d_toy_dataset(X, y)

