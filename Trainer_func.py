def train_epoch(model, B, im2_train, im3_train, disp_train, optimizer, criterion, device, iters, ep_num, writer):
    """ Training a model for one epoch """
    
    loss_list = []
    im_L_iter = iter(im2_train)
    im_R_iter = iter(im3_train)
    disp_iter = iter(disp_train)
    counter = 0
    run_for = int(150/B)
    for counter in range(run_for):
        
        im_L = next(im_L_iter)
        im_R = next(im_R_iter)
        disp = next(disp_iter)
        
        left_im = im_L
        left_im = left_im.to(device)
        
        right_im = im_R
        right_im = right_im.to(device)
        
        disp_true = disp
        disp_true = disp_true.to(device)
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        
        # Forward pass
        recons = model(left_im,right_im)
        recons = recons[:,0,:,:] # We only need one channel
        
        disp_true = disp_true/256.0  #Because original values are too large to be compared to max_disparity
        
        loss = criterion(recons, disp_true)
        
        loss_list.append(loss.item())
  
        # Getting gradients w.r.t. parameters
        loss.backward()
        
        # Updating parameters
        optimizer.step()
   
        writer.add_scalar(f'Loss/Discriminator Loss', loss.item(), global_step=iters)
       
        iters += 1
    return loss_list, iters