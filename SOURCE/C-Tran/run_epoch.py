import argparse,math,time,warnings,copy, numpy as np, os.path as path 
import torch, torch.nn as nn, torch.nn.functional as F
from pdb import set_trace as stop
from tqdm import tqdm
from models.utils import custom_replace
import random
from losses import *

def run_epoch(args,model,data,optimizer,epoch,desc,train=False,warmup_scheduler=None):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    # pre-allocate full prediction and target tensors
    all_predictions = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_targets = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_masks = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_image_ids = []

    max_samples = args.max_samples

    batch_idx = 0
    loss_total = 0
    unk_loss_total = 0

    for batch in tqdm(data,mininterval=0.5,desc=desc,leave=False,ncols=50):
        if batch_idx == max_samples:
            break

        ## If use backbone2.
        # if train:
        #     label_vec_est = model.backbone.g_train(batch['idx'])
        # else:
        #     label_vec_est = model.backbone.g_val(batch['idx'])
            
        label_obs = batch['labels_obs'].float()  ###

        labels = batch['labels'].float()
        images = batch['image'].float()
        mask = batch['mask'].float()
        unk_mask = custom_replace(mask,1,0,0)
        all_image_ids += batch['imageIDs']
        
        mask_in = mask.clone()

        # Cuda is available
        if torch.cuda.is_available(): 
            if train:
                pred,int_pred,attns = model(images.cuda(),mask_in.cuda())
            else:
                with torch.no_grad():
                    pred,int_pred,attns = model(images.cuda(),mask_in.cuda())

            if args.dataset == 'cub':
                class_label = batch['class_label'].float()
                concept_certainty = batch['concept_certainty'].float()

                class_label_onehot = torch.zeros(class_label.size(0),200)
                class_label_onehot.scatter_(1,class_label.long(),1)

                labels = torch.cat((labels,class_label_onehot),1)
                loss =  F.binary_cross_entropy_with_logits(pred.view(labels.size(0),-1),labels.cuda(),reduction='none')
                loss = (unk_mask.cuda()*loss).sum()/unk_mask.detach().sum().item()

                aux_loss =  F.binary_cross_entropy_with_logits(int_pred.view(labels.size(0),-1),labels.cuda(),reduction='none')
                aux_loss = (unk_mask.cuda()*aux_loss).sum()/unk_mask.detach().sum().item()

                loss_out = 1.0*loss + float(args.aux_loss)*aux_loss
                loss = loss_out
            else:
                # Original Loss
                # loss =  F.binary_cross_entropy_with_logits(pred.view(labels.size(0),-1),labels.cuda(),reduction='none')

                # Softmax activation
                # pred = F.softmax(pred, dim=1)

                # Sigmoid activation
                pred = F.sigmoid(pred)

                reg_loss = 0
                loss = F.binary_cross_entropy(pred.view(labels.size(0),-1),labels.cuda(),reduction='none')

                # Combine model loss
                # loss, reg_loss = compute_batch_loss(preds = pred, observed_labels = label_obs.cuda(), loss = 'an_ls')
                # loss, reg_loss = compute_batch_loss(preds = pred, observed_labels = label_obs.cuda(), loss = 'hu')
                # loss, reg_loss = compute_batch_loss(preds = pred, observed_labels = label_obs.cuda(), loss = 'role', estimated_labels = label_vec_est)

                if args.loss_labels == 'unk': 
                    # only use unknown labels for loss
                    loss_out = (unk_mask.cuda()*loss).sum()
                else: 
                    # use all labels for loss
                    loss_out = loss.sum() 
                loss_out += reg_loss
        # Cuda isn't available
        else:
            if train:
                pred,int_pred,attns = model(images,mask_in)
            else:
                with torch.no_grad():
                    pred,int_pred,attns = model(images,mask_in)

            if args.dataset == 'cub':
                class_label = batch['class_label'].float()
                concept_certainty = batch['concept_certainty'].float()

                class_label_onehot = torch.zeros(class_label.size(0),200)
                class_label_onehot.scatter_(1,class_label.long(),1)

                labels = torch.cat((labels,class_label_onehot),1)
                loss =  F.binary_cross_entropy_with_logits(pred.view(labels.size(0),-1),labels,reduction='none')
                loss = (unk_mask*loss).sum()/unk_mask.detach().sum().item()

                aux_loss =  F.binary_cross_entropy_with_logits(int_pred.view(labels.size(0),-1),labels,reduction='none')
                aux_loss = (unk_mask*aux_loss).sum()/unk_mask.detach().sum().item()

                loss_out = 1.0*loss + float(args.aux_loss)*aux_loss
                loss = loss_out
            else:
                # Original Loss
                # loss =  F.binary_cross_entropy_with_logits(pred.view(labels.size(0),-1),labels.cuda(),reduction='none')

                # Softmax activation
                # pred = F.softmax(pred, dim=1)

                # Sigmoid activation
                pred = F.sigmoid(pred)

                reg_loss = 0
                loss = F.binary_cross_entropy(pred.view(labels.size(0),-1),labels.cuda(),reduction='none')

                # Combine model loss
                # loss, reg_loss = compute_batch_loss(preds = pred, observed_labels = label_obs.cuda(), loss = 'an_ls')
                # loss, reg_loss = compute_batch_loss(preds = pred, observed_labels = label_obs.cuda(), loss = 'hu')
                # loss, reg_loss = compute_batch_loss(preds = pred, observed_labels = label_obs.cuda(), loss = 'role', estimated_labels = label_vec_est)

                if args.loss_labels == 'unk': 
                    # only use unknown labels for loss
                    loss_out = (unk_mask.cuda()*loss).sum()
                else: 
                    # use all labels for loss
                    loss_out = loss.sum() 
                loss_out += reg_loss

        if train:
            loss_out.backward()
            # Grad Accumulation
            if ((batch_idx+1)%args.grad_ac_steps == 0):
                optimizer.step()
                optimizer.zero_grad()
                if warmup_scheduler is not None:
                    warmup_scheduler.step()

        ## Updates ##
        loss_total += loss_out.item()
        unk_loss_total += loss_out.item()
        start_idx,end_idx=(batch_idx*data.batch_size),((batch_idx+1)*data.batch_size)
        
        if pred.size(0) != all_predictions[start_idx:end_idx].size(0):
            pred = pred.view(labels.size(0),-1)
        
        all_predictions[start_idx:end_idx] = pred.data.cpu()
        all_targets[start_idx:end_idx] = labels.data.cpu()
        all_masks[start_idx:end_idx] = mask.data.cpu()
        batch_idx +=1

    loss_total = loss_total/float(all_predictions.size(0))
    unk_loss_total = unk_loss_total/float(all_predictions.size(0))

    return all_predictions,all_targets,all_masks,all_image_ids,loss_total,unk_loss_total


