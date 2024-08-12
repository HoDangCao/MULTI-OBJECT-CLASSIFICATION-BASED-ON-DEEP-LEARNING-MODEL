import torch

LOG_EPSILON = 1e-5
P = {}
P['ls_coef'] = 0.1
P['expected_num_pos'] = 1.72 
P['num_classes'] = 323 
'''
helper functions
'''

def neg_log(x):
    return - torch.log(x + LOG_EPSILON)


def expected_positive_regularizer(preds, expected_num_pos, norm='2'):
    # Assumes predictions in [0,1].
    if norm == '1':
        reg = torch.abs(preds.sum(1).mean(0) - expected_num_pos)
    elif norm == '2':
        reg = (preds.sum(1).mean(0) - expected_num_pos)**2
    else:
        raise NotImplementedError
    return reg
    
'''
loss functions
'''

def loss_huber(preds, true_labels, t = None):
    delta = 1.0
    # compute loss:
    huber_mse = 0.5*(true_labels - preds)**2
    huber_mae = delta * (torch.abs(true_labels - preds) - 0.5 * delta)
    loss_mtx = torch.where(torch.abs(true_labels - preds) <= delta, huber_mse, huber_mae)
    reg_loss = 0
    return loss_mtx, reg_loss

def loss_an_ls(preds, observed_labels, t = None):
    # input validation: 
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = (1.0 - P['ls_coef']) * neg_log(preds[observed_labels == 1]) + P['ls_coef'] * neg_log(1.0 - preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = (1.0 - P['ls_coef']) * neg_log(1.0 - preds[observed_labels == 0]) + P['ls_coef'] * neg_log(preds[observed_labels == 0])
    reg_loss = 0
    return loss_mtx, reg_loss


def loss_role(preds, observed_labels, estimated_labels):
    # input validation:
    assert torch.min(observed_labels) >= 0
    # (image classifier) compute loss w.r.t. observed positives:
    loss_mtx_pos_1 = torch.zeros_like(observed_labels)
    loss_mtx_pos_1[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    # (image classifier) compute loss w.r.t. label estimator outputs:
    estimated_labels_detached = estimated_labels#.detach()
    loss_mtx_cross_1 = estimated_labels_detached * neg_log(preds) + (1.0 - estimated_labels_detached) * neg_log(1.0 - preds)
    # (image classifier) compute regularizer: 
    reg_1 = expected_positive_regularizer(preds, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    # (label estimator) compute loss w.r.t. observed positives:
    loss_mtx_pos_2 = torch.zeros_like(observed_labels)
    loss_mtx_pos_2[observed_labels == 1] = neg_log(estimated_labels[observed_labels == 1])
    # (label estimator) compute loss w.r.t. image classifier outputs:
    preds_detached = preds#.detach()
    loss_mtx_cross_2 = preds_detached * neg_log(estimated_labels) + (1.0 - preds_detached) * neg_log(1.0 - estimated_labels)
    # (label estimator) compute regularizer:
    reg_2 = expected_positive_regularizer(estimated_labels, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    # compute final loss matrix:
    reg_loss = 0.5 * (reg_1 + reg_2)
    loss_mtx = 0.5 * (loss_mtx_pos_1 + loss_mtx_pos_2)
    loss_mtx += 0.5 * (loss_mtx_cross_1 + loss_mtx_cross_2)
    
    return loss_mtx, reg_loss

loss_functions = {
    'an_ls': loss_an_ls,
    'role': loss_role,
    'hu': loss_huber,
}

'''
top-level wrapper
'''

def compute_batch_loss(preds, observed_labels, loss, estimated_labels = None):
    assert preds.dim() == 2
    
    batch_size = int(preds.size(0))
    num_classes = int(preds.size(1))
        
    # input validation:
    assert torch.max(observed_labels) <= 1
    assert torch.min(observed_labels) >= -1
    assert preds.size() == observed_labels.size()
    # assert P['loss'] in loss_functions
    
    # validate predictions:
    assert torch.max(preds) <= 1
    assert torch.min(preds) >= 0
    
    # compute loss for each image and class:
    loss_mtx, reg_loss = loss_functions[loss](preds, observed_labels, estimated_labels)
    
    
    return loss_mtx, reg_loss