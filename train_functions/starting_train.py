import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchmetrics.classification import BinaryF1Score
from torchmetrics import ConfusionMatrix

def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    #loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.BCELoss() #we already have sigmoid

    step = 0
    for epoch in range(epochs):
        # print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        metric = BinaryF1Score()
        confmat = ConfusionMatrix(task="binary", num_classes=2)
        f1_scores, confmats = [], []
        cnt = 0 #how many batches processed
        acc = 0 # accuracy per batch, cumulative.
        loss = 0 # Loss per batch, cumulative.

        for batch in tqdm(train_loader):
            inputs, labels = batch
            # Forward propagate
            outputs = model(inputs).squeeze()

            
            # Backpropagation and gradient descent
            loss = loss_fn(outputs, labels.float())
            loss.backward()       # Compute gradients
            optimizer.step()      # Update all the weights with the gradients you just calculated
            optimizer.zero_grad() # Clear gradients before next iteration

            # Cumulate tickers.
            f1_scores.append(metric(outputs, torch.Tensor(labels)))
            confmats.append(confmat(outputs, torch.Tensor(labels)))
            acc += (compute_accuracy(outputs, labels))
            loss += (loss_fn(outputs, labels.float()))
            cnt += 1

            # Periodically evaluate our model + log to Tensorboard
            if (step+1) % n_eval == 0:
                # Print out model performance during training phase.
                print("Training statistics:\n===============================")
                print("Accuracy: " + str(acc / cnt))
                print("Loss: " + str(loss / cnt))
                print("F1:", sum(f1_scores) / len(f1_scores))
                print("Confusion matrix:")
                print(torch.stack(confmats).to(torch.float32).mean(axis=0)/32)
                cnt = 0; acc = 0; loss = 0; f1_scores.clear(); confmats.clear(); 
                print("Evaluating\n===============================")
                model.eval()

                # Print out model performance during evaluation phase.
                compute_accuracy(outputs, labels)
                ### Log results

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard. 
                # Don't forget to turn off gradient calculations!
                evaluate(val_loader, model, loss_fn)
                model.train()
            step += 1

        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.
    """
    metric = BinaryF1Score()
    confmat = ConfusionMatrix(task="binary", num_classes=2)
    f1_scores, confmats = [], []
    with torch.no_grad():
        cnt = 0
        acc = 0
        loss = 0
        running_corrects = 0
        for batch in tqdm(val_loader):
            cnt += 1
            inputs, labels = batch
            
            outputs = model(inputs).squeeze()

            acc += (compute_accuracy(outputs, labels))
            loss += (loss_fn(outputs, labels.float()))
            running_corrects += torch.sum(outputs == labels.data)
            
            #print(labels.data[0])
            #print(torch.round(outputs[0]))  
            f1_scores.append(metric(outputs, torch.Tensor(labels)))
            confmats.append(confmat(outputs, torch.Tensor(labels)))

        print("Accuracy: " + str(acc / cnt))
        print("Loss: " + str(loss / cnt))
        print("F1:", sum(f1_scores) / len(f1_scores))
        print("Confusion matrix:")
        print(torch.stack(confmats).to(torch.float32).mean(axis=0)/36)


