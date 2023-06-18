import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data import XORDataset
from modeling import XORClassifier
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler
from sklearn.model_selection import train_test_split



def train(model, train_loader, valid_loader, epochs, loss_func, optimizer, scheduler, scaler, device):
    model = model.to(device)
    for epoch in range(epochs):
        total_train_loss = 0 # initialzing the total losses to zero for each epoch
        total_validation_loss = 0
        for X, y in train_loader:  # going through data loader
            X, y = X.to(device), y.to(device).view(-1, 1)
            logits = model(X)  # forward pass

            loss = loss_func(logits, y)  # loss calculation
            total_train_loss += loss.item()  # adding to existing loss

            # calculating gradients
            scaler.scale(loss).backward()  # backward pass with scaled gradients
            with torch.no_grad():  # disabling gradients propagation
                scaler.step(optimizer)
                scaler.update()  # updating model parameters(weights)

        
        # printing the current situation for each epoch
        print(f"Epoch: {epoch + 1}/{epochs}\tTraining loss: {total_train_loss:.3f}\t\
              Validation loss: {total_validation_loss:.3f}\
              \tLearning rate: {optimizer.param_groups[0]['lr']}")
        
        if epoch % 10 == 0:  # evaluate the model at every 10 epochs
            model.eval()  # evaluating the model in eval mode
            total_validation_loss = validate(model, valid_loader, loss_func, device)
            print(f"\nValidation loss: {total_validation_loss}\n")
        else:
            model.train() # training mode
        
        scheduler.step()  # decay the learning rate
           


def validate(model, data_loader, loss_func, device):
    total_loss = 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device).view(-1, 1)
        logits = model(X)
        loss = loss_func(logits, y)
        total_loss += loss.item()

    return total_loss




if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    

    dataset = XORDataset(n_samples=100000)  # Generating dataset

    # splitting dataset into train, validation and test
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.4, shuffle=True, random_state=32)
    valid_dataset, test_dataset = train_test_split(dataset, test_size=0.15, shuffle=True, random_state=32)   # creating validation dataset

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=64, 
                              num_workers=4 if device == "cuda" else 1,
                              shuffle=True)
    
    valid_loader = DataLoader(dataset=valid_dataset, 
                              batch_size=64, 
                              num_workers=4 if device == "cuda" else 1,
                              shuffle=False)

    test_loader = DataLoader(dataset=valid_dataset, 
                              batch_size=64, 
                              num_workers=2 if device == "cuda" else 1,
                              shuffle=False)

    # for X, y in valid_loader:
    #     print(f"X: {X}\ny: {y}")
    #     break   

    # instantiating the model
    model = XORClassifier(2, 1)

    # defining the loss function
    loss_function = F.binary_cross_entropy()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.98)
    
    epochs = 30
    scaler = GradScaler()
    scheduler = StepLR(step_size=6, optimizer=optimizer, gamma=0.1)
    # training and evaluating the model
    print("\n\n")
    train(model, train_loader, valid_loader, epochs, loss_function, optimizer, scheduler, scaler, device)
    print("\n\n")
    # making predictions
    with torch.inference_mode():
        print(f"=================== Predictions/model inference =====================")
        print(f"\t1 XOR 1: {model.predict(torch.tensor([1., 1.]).to(device))}")
        print(f"\t0 XOR 1: {model.predict(torch.tensor([0., 1.]).to(device))}")
        print(f"\t1 XOR 0: {model.predict(torch.tensor([1., 0.]).to(device))}")
        print(f"\t0 XOR 0: {model.predict(torch.tensor([0., 0.]).to(device))}")
    
    print("\n\n")
    