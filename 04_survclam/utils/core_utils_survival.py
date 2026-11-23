import numpy as np
import torch
import torch.nn.functional as F
from utils.utils_survival import *
import os
import torch.nn.functional as F
from dataset_modules.dataset_generic_survival import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam_survival import CLAM_SB_Surv
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sksurv.metrics import concordance_index_censored
from lifelines.utils import concordance_index 
import pandas as pd
import sys
#from utils.gpu_utils import gpu_profile, print_gpu_mem
#os.environ['GPU_DEBUG']='0'

def coxph_loss(y_pred, y_time, y_event):
    time = y_time
    event = y_event

    sort_time = torch.argsort(time, 0, descending=True)
    event = torch.gather(event, 0, sort_time)
    
    risk = torch.gather(y_pred, 0, sort_time)
    exp_risk = torch.exp(risk)
    log_risk = torch.log(torch.cumsum(exp_risk, 0))
    censored_likelihood = (risk - log_risk) * event
    censored_likelihood = torch.sum(censored_likelihood)
    censored_likelihood = censored_likelihood / y_time.shape[0]
    return -censored_likelihood
def nll_loss(hazards, S, Y, c, alpha=0.15, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss
class NegativeLogLikelihood(nn.Module):
    def __init__(self, config):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):
        mask = torch.ones(y.shape[0], y.shape[0])
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        l2_loss = self.reg(model)
        return neg_log_loss + l2_loss

def c_index(y_pred, y_time, y_event):
    time = y_time
    event = y_event

    return concordance_index(time, -np.exp(y_pred), event)

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif  self.best_score - score>0.0005:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience or epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """
    Train for a single fold and return validation/test c-index
    """
    print(f'\nTraining Fold {cur}!')
    writer_dir = os.path.join(args.results_dir, str(cur))
    os.makedirs(writer_dir, exist_ok=True)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    split_keys = ['train', 'val'] + (['test'] if test_split is not None else [])
    save_splits(datasets[:len(split_keys)], split_keys, os.path.join(args.results_dir, f'splits_{cur}.csv'))
    print('Done!')
    print(f"Training on {len(train_split)} samples")
    print(f"Validating on {len(val_split)} samples")
    if test_split is not None:
        print(f"Testing on {len(test_split)} samples")

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes).cuda() if device.type == 'cuda' else SmoothTop1SVM(n_classes=args.n_classes)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0]).to('cuda'))
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=False)
    val_loader = get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing) if test_split is not None else None
    print('Done!')

    print(f"\nUsing alpha={args.alpha}")

    print('\nInit Model...', end=' ')
    model_dict = {'embed_dim': args.embed_dim, 'dropout': args.drop_out, 'n_classes': args.n_classes, 'alpha': args.alpha}
    if args.model_type == 'clam' and args.subtyping:
        model_dict['subtyping'] = True
    if args.model_size and args.model_type != 'mil':
        model_dict['size_arg'] = args.model_size

    if args.model_type in ['clam_sb', 'clam_mb', 'clam_sb_surv']:
        if args.subtyping:
            model_dict['subtyping'] = True
        if args.B > 0:
            model_dict['k_sample'] = args.B

        if args.inst_loss == 'svm':
            from topk import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes=2).cuda() if device.type == 'cuda' else SmoothTop1SVM(n_classes=2)
        else:
            instance_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0]).to('cuda'))

        model = {
            'clam_sb': lambda: CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn),
            'clam_mb': lambda: CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn),
            'clam_sb_surv': lambda: CLAM_SB_Surv(**model_dict)
        }[args.model_type]()

    elif 'hipt' in args.model_type:
        model = {
            'hipt_n': lambda: HIPT_None_FC(**model_dict),
            'hipt_lgp': lambda: HIPT_LGP_FC(**model_dict, freeze_4k=args.freeze_4k, pretrain_4k=args.pretrain_4k, freeze_WSI=args.freeze_WSI, pretrain_WSI=args.pretrain_WSI),
            'hipt_gp': lambda: HIPT_GP_FC(**model_dict, freeze_WSI=args.freeze_WSI, pretrain_WSI=args.pretrain_WSI)
        }[args.model_type]()

    elif args.model_type == 'dsmil':
        model = MILNet(FCLayer(in_size=args.embed_dim, out_size=args.n_classes), BClassifier(input_size=args.embed_dim, output_class=args.n_classes, dropout_v=0.0))
    elif args.model_type == 'dgcn':
        model = DeepGraphConv(num_features=args.embed_dim, n_classes=args.n_classes)
    elif args.model_type == 'mi_fcn':
        model = MIL_Cluster_FC(embed_dim=args.embed_dim, n_classes=args.n_classes)
    else:
        model = MIL_fc_mc(**model_dict) if args.n_classes > 2 else MIL_fc(**model_dict)

    model = model.relocate() if hasattr(model, 'relocate') else model.to(torch.device('cuda'))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    early_stopping = EarlyStopping(patience=5, stop_epoch=20, verbose=True) if args.early_stopping else None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb', 'clam_sb_surv'] and not args.no_inst_cluster:
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, args.alpha, writer, loss_fn, dropinput=0.25)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, args.alpha, early_stopping, writer, loss_fn, args.results_dir)
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, early_stopping, writer, loss_fn, args.results_dir)
        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt"))

    val_index = summary(model, val_loader, args.n_classes, args.alpha)
    test_index = summary(model, test_loader, args.n_classes, args.alpha) if test_loader else float('nan')

    with open(os.path.join(args.results_dir, f'fold_{cur}.txt'), 'w') as f:
        f.write(f'Fold:{cur}, Alpha: {args.alpha}, Valid:  C-Index: {val_index:.4f}, Test:  C-Index: {test_index:.4f}\n')
    print(f'Fold:{cur}, Alpha: {args.alpha}, Valid:  C-Index: {val_index:.4f}, Test:  C-Index: {test_index:.4f}')

    if writer:
        writer.add_scalar('val_cindex', val_index, 0)
        if not np.isnan(test_index):
            writer.add_scalar('test_cindex', test_index, 0)
        writer.close()

    return val_index, test_index


def train_loop(epoch, model, loader, optimizer, n_classes, alpha, writer = None, loss_fn = None, gc=32):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, batch in enumerate(loader):

        if hasattr(model, "num_clusters"):
            data, cluster_id, label = batch
            data, cluster_id, label = data.to(device, non_blocking=True), cluster_id, label.to(device, non_blocking=True)
        else:
            data, label = batch
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            cluster_id = None

        # logits, Y_prob, Y_hat, _, _ = model(data, cluster_id=cluster_id)
        #logits, Y_prob, Y_hat, _, _ = model(x_path=data)
        if Y_hat is not None:
            acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        loss = loss / gc
        loss.backward()

        # step
        
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if hasattr(model, "num_clusters"):
                data, cluster_id, label = batch
                data, cluster_id, label = data.to(device, non_blocking=True), cluster_id, label.to(device, non_blocking=True)
            else:
                data, label = batch
                data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
                cluster_id = None
            logits, Y_prob, Y_hat, _, _ = model(data, cluster_id=cluster_id)
            #logits, Y_prob, Y_hat, _, _ = model(x_path=data)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, alpha, writer=None, loss_fn=None, dropinput=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    train_loss = 0.0
    step = 0
    gc = 0
    epochCoxloss = 0

    WSI_logits = []
    slide_time = []
    slide_event = []

    Cindex_logits = []
    Cindex_time = []
    Cindex_event = []

    for batch_idx, batch in enumerate(loader):
        if len(batch) == 2:
            data, survival = batch
            coords = None
        elif len(batch) == 3:
            data, survival, coords = batch
        else:
            raise ValueError(f"Unexpected batch format: {len(batch)} items")

        data = data.to(device)
        y_time = survival[:, 0].to(device)
        y_event = survival[:, 1].to(device)

        if dropinput > 0:
            data = F.dropout(data, p=dropinput)

        logits, Y_prob, Y_hat, _, instance_dict = model(h=data, cluster_id=None, label=None, instance_eval=True)

        WSI_logits.append(logits[0][0].clone())
        slide_time.append(y_time[0].item())
        slide_event.append(y_event[0].item())

        Cindex_logits.append(logits[0].item())
        Cindex_time.append(y_time[0].item())
        Cindex_event.append(y_event[0].item())

        step += 1
        if step % 10 == 0:
            WSI_logits = torch.stack(WSI_logits)
            slide_time_tensor = torch.tensor(slide_time).to(device)
            slide_event_tensor = torch.tensor(slide_event).to(device)

            # Cox loss
            coxloss = coxph_loss(WSI_logits, slide_time_tensor, slide_event_tensor)

            # === L2 Regularization ===
            l2_reg = torch.tensor(0.0, device=coxloss.device)
            for param in model.parameters():
                if param.requires_grad:
                    l2_reg += torch.norm(param, p=2) ** 2
            coxloss += alpha * l2_reg

            coxloss.backward()

            epochCoxloss += coxloss.item()

            WSI_logits = []
            slide_time = []
            slide_event = []

            gc += 1
            if gc % 20 == 0:
                optimizer.step()
                optimizer.zero_grad()

    epochCoxloss = epochCoxloss / gc if gc > 0 else 0
    cindex = c_index(Cindex_logits, Cindex_time, Cindex_event)

    print(f'Train: Epoch: {epoch}, coxloss: {epochCoxloss:.4f}, cindex: {cindex:.4f}')


def validate_clam(cur, epoch, model, loader, n_classes, alpha, early_stopping=None, writer=None, loss_fn=None, results_dir=None, args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    WSI_logits = []
    slide_time = []
    slide_event = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if hasattr(model, "num_clusters") and len(batch) == 3:
                data, cluster_id, survival = batch
            else:
                data, survival = batch
                cluster_id = None

            data = data.to(device)
            y_time = survival[:, 0].to(device)
            y_event = survival[:, 1].to(device)

            logits, Y_prob, Y_hat, _, instance_dict = model(h=data, cluster_id=cluster_id, label=None, instance_eval=True)

            WSI_logits.append(logits[0][0].detach().cpu())
            slide_time.append(y_time[0].item())
            slide_event.append(y_event[0].item())

    WSI_logits = torch.stack(WSI_logits)
    slide_time = torch.tensor(slide_time)
    slide_event = torch.tensor(slide_event)

    # === Cox Loss ===
    coxloss = coxph_loss(WSI_logits, slide_time, slide_event)

    # === L2 Regularization ===
    if alpha > 0:
        l2_reg = sum(torch.norm(param, p=2) ** 2 for param in model.parameters() if param.requires_grad)
        l2_reg = l2_reg.to(coxloss.device)
        coxloss += alpha * l2_reg

    # === C-index ===
    cindex = c_index(WSI_logits, slide_time, slide_event)

    print(f'Valid: Epoch: {epoch}, coxloss: {coxloss:.4f}, cindex: {cindex:.4f}')

    if writer:
        writer.add_scalar('val/coxloss', coxloss.item(), epoch)
        writer.add_scalar('val/cindex', cindex, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, coxloss, model, ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt"))
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes, alpha):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    slide_ids = loader.dataset.slide_data['slide_id']

    WSI_logits = []
    slide_time = []
    slide_event = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if hasattr(model, "num_clusters") and len(batch) == 3:
                data, cluster_id, survival = batch
            else:
                data, survival = batch
                cluster_id = None

            data = data.to(device)
            y_time, y_event = survival.squeeze().tolist()

            logits, Y_prob, Y_hat, _, _ = model(data, cluster_id=cluster_id)

            WSI_logits.append(logits[0][0].cpu())
            slide_time.append(y_time)
            slide_event.append(y_event)

    WSI_logits = torch.tensor(WSI_logits)
    slide_time = torch.tensor(slide_time)
    slide_event = torch.tensor(slide_event)

    coxloss = coxph_loss(WSI_logits, slide_time, slide_event)

    # === Add L2 regularization if alpha is provided ===
    if alpha > 0:
        l2_reg = sum(torch.norm(param, p=2) ** 2 for param in model.parameters() if param.requires_grad)
        l2_reg = l2_reg.to(coxloss.device)
        coxloss += alpha * l2_reg

    cindex = c_index(WSI_logits, slide_time, slide_event)

    print('Summary: coxloss: {:.4f}, cindex: {:.4f}'.format(coxloss, cindex))
    return cindex


def summary_score(fold, model, loader, n_classes, alpha):
    results_dir = n_classes.results_dir
    n_classes = n_classes.n_classes 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    slide_ids = loader.dataset.slide_data['slide_id']

    WSI_logits = []
    slide_time = []
    slide_event = []

    for batch_idx, batch in enumerate(loader):
        if hasattr(model, "num_clusters"):
            data, cluster_id, survival = batch
        else:
            data, survival = batch
            cluster_id = None

        data = data.to(device)
        y_time = survival[:, 0].to(device)
        y_event = survival[:, 1].to(device)

        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data, cluster_id=cluster_id)
            WSI_logits.append(logits[0][0].cpu())
            slide_time.append(y_time.item())
            slide_event.append(y_event.item())

    final_score = pd.DataFrame({'slide_ids': slide_ids, 'val_score': np.array(WSI_logits)})

    WSI_logits = torch.tensor(WSI_logits)
    slide_time = torch.tensor(slide_time)
    slide_event = torch.tensor(slide_event)

    coxloss = coxph_loss(WSI_logits, slide_time, slide_event)

    # === Add L2 regularization ===
    if alpha > 0:
        l2_reg = sum(torch.norm(param, p=2) ** 2 for param in model.parameters() if param.requires_grad)
        l2_reg = l2_reg.to(coxloss.device)
        coxloss += alpha * l2_reg

    cindex = c_index(WSI_logits, slide_time, slide_event)

    print('Valid:  coxloss: {:.4f} cindex: {:.4f}'.format(coxloss, cindex))
    final_score.to_csv(os.path.join(results_dir, f'fold_{fold}.csv'))
    return cindex
