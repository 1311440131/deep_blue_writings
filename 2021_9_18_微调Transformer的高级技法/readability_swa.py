def train_fn(data_loader, model, optimizer, ...., swa_step=False):
        
    model.train()                               # Put the model in training mode.   
    ....
    ....
    for batch in data_loader:                   # Loop over all batches.
        ....
        ....
        optimizer.zero_grad()                   # To zero out the gradients.        
        outputs = model(ids, masks).squeeze(-1) # Predictions from 1 batch of data.
        ....
        ....
        loss.backward()                         # To backpropagate the error (gradients are computed).
        optimizer.step()                        # To update parameters based on current gradients.
        ....
        ....
        if swa_step:            
            swa_model.update_parameters(model)  # To update parameters of the averaged model.
            swa_scheduler.step()                # Switch to SWALR.
        else:        
            scheduler.step()                    # To update learning rate.
               
    return train_losses, ....