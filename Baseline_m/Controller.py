import torch.optim as optim
from tensorboardX import SummaryWriter
import time
from .Utils import *
from .Metrics import *
from .Retrieval_Utils import i2t, t2i, evaluate_recall
from pathlib import Path
from torch.nn.utils.clip_grad import clip_grad_norm_ as clip_grad_norm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch.nn as nn
from torch.autograd import Variable
import itertools
import random
import os
import mlflow
from .Models import *


class Controller(nn.Module):
    def __init__(self, config):
        super(Controller, self).__init__()
        # CONFIG
        self.config = config
        self.ft_gcn = config['ft_gcn']
        self.ft_com = config['ft_com']
        self.ft_itm = config['ft_itm']
        self.ft_trans = config['ft_trans']
        self.type_gcn = config['type_gcn']
        self.skip = config['skip']
        self.act_func = config['act_func']
        self.batch_norm = config['batch_norm']
        self.dropout = config['dropout']
        self.n_heads = config['n_heads']
        self.l2_norm = config['l2_norm']
        
        self.alpha = config['alpha']
        self.distill = config['distill']
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        
        self.optimizer_choice = config['optimizer_choice']
        self.weight_decay = config['weight_decay']
        self.early_stop = config['early_stop']
        self.thres_loss = config['thres_loss']
        self.learning_rate = config['learning_rate']
        self.device = config['device']
        self.min_lr = config['min_lr']
        self.factor = config['factor']
        self.patience = config['patience']
        self.T0 = config['T0']
        self.Tmul = config['Tmul']
        self.Tmax = config['Tmax']
        self.margin_loss = config['margin_loss']
        weight_emb_loss = config['weight_emb_loss']
        weight_nll_loss = config['weight_nll_loss']
        weight_itm_loss = config['weight_itm_loss']
        total_weight = weight_emb_loss + weight_nll_loss + weight_itm_loss
        total_weight = 1
        self.weight_emb_loss = weight_emb_loss / total_weight
        self.weight_nll_loss = weight_nll_loss / total_weight
        self.weight_itm_loss = weight_itm_loss / total_weight
        current_folder = os.path.dirname(os.path.abspath(__file__)) 
        self.out_dir = f"{current_folder}/{config['out_dir']}"
        self.pretrained_path = config['pretrained_path']
        self.temp = config['temp']
        if self.temp > 0:
            self.temp_para = nn.Parameter(torch.ones([]) * self.temp)
            # self.temp_para = self.temp_para.to(self.device)
        else:
            self.temp_para = self.temp
            
        # MODEL
        self.img_encoder = MH(ft_in=1024, ft_out=self.ft_com,
                              batch_norm=self.batch_norm, dropout=self.dropout, act_func=self.act_func)
        self.cap_encoder = MH(ft_in=1024, ft_out=self.ft_com,
                              batch_norm=self.batch_norm, dropout=self.dropout, act_func=self.act_func)
        self.discriminator = Discriminator(ft_in=self.ft_com[-1], ft_out=self.ft_itm,
                                           batch_norm=self.batch_norm, dropout=self.dropout, act_func=self.act_func)
        
        # self.discriminator = Discriminator_2(ft_in=self.ft_com[-1], dropout=0.25)
        self.img_encoder_m = MH(ft_in=1024, ft_out=self.ft_com,
                                batch_norm=self.batch_norm, dropout=self.dropout, act_func=self.act_func)
        self.cap_encoder_m = MH(ft_in=1024, ft_out=self.ft_com,
                                batch_norm=self.batch_norm, dropout=self.dropout, act_func=self.act_func)
        
        self.img_encoder = self.img_encoder.to(self.device)
        self.cap_encoder = self.cap_encoder.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.img_encoder_m = self.img_encoder_m.to(self.device)
        self.cap_encoder_m = self.cap_encoder_m.to(self.device)
        
        self.model_pairs = [[self.img_encoder,self.img_encoder_m],
                            [self.cap_encoder,self.cap_encoder_m]]
        self.copy_params()
        
        ## PARAMS & OPTIMIZER
        self.params = []
        self.params += list(filter(lambda p: p.requires_grad, self.img_encoder.parameters()))
        self.params += list(filter(lambda p: p.requires_grad, self.cap_encoder.parameters()))
        self.params += list(filter(lambda p: p.requires_grad, self.discriminator.parameters()))
        self.params += list(filter(lambda p: p.requires_grad, self.img_encoder_m.parameters()))
        self.params += list(filter(lambda p: p.requires_grad, self.cap_encoder_m.parameters()))
        
        if self.temp > 0:
            self.params += [self.temp_para]
        if self.optimizer_choice.lower() == 'adam':                                                     
            self.optimizer = optim.Adam(self.params,
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
        if self.optimizer_choice.lower() == 'adamw':                                                     
            self.optimizer = optim.AdamW(self.params,
                                         lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
            
        # create the queue
        self.register_buffer("image_queue", torch.randn(self.ft_com[-1], self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.ft_com[-1], self.queue_size))
        self.register_buffer("idx_queue", torch.full((1,self.queue_size),-100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  

        self.image_queue = F.normalize(self.image_queue, dim=0).to(self.device)
        self.text_queue = F.normalize(self.text_queue, dim=0).to(self.device)
        
        ## Add MLFLOW
        self.model_type = config['model_type']
        self.config_path = config['config_path']
        self.config_name = self.config_path.split('/')[-1]
        experiment = mlflow.get_experiment_by_name(self.model_type)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(name = self.model_type)
        else: 
            self.experiment_id = experiment.experiment_id
        
    def train_mode(self):
        self.img_encoder.train()
        self.cap_encoder.train()
        self.img_encoder_m.train()
        self.cap_encoder_m.train()
        self.discriminator.train()
        
    def eval_mode(self):
        self.img_encoder.eval()
        self.cap_encoder.eval()
        self.img_encoder_m.eval()
        self.cap_encoder_m.eval()
        self.discriminator.eval()
        
    # ---------- LOAD TRAINED MODEL ---------
    def load_model(self, save_path=None):
        #---- Load checkpoint 
        if save_path is not None:
            modelCheckpoint = torch.load(save_path)
            self.img_encoder.load_state_dict(modelCheckpoint['img_encoder'])
            self.cap_encoder.load_state_dict(modelCheckpoint['cap_encoder'])
            self.discriminator.load_state_dict(modelCheckpoint['discriminator'])
            self.temp_para = modelCheckpoint['temp_para']
            if 'optimizer' in list(modelCheckpoint.keys()):
                self.optimizer.load_state_dict(modelCheckpoint['optimizer'])
            print(f"LOADED PRETRAINED MODEL AT {save_path}")
        else:
            print("TRAIN FROM SCRATCH")
            
    def save_model(self, loss, epochID=0, save_path='', optimizer=True):
        if optimizer:
            torch.save({'epoch': epochID,
                        'img_encoder': self.img_encoder.state_dict(),
                        'cap_encoder': self.cap_encoder.state_dict(),
                        'discriminator': self.discriminator.state_dict(),
                        'temp_para': self.temp_para,
                        'optimizer': self.optimizer.state_dict(),
                        'best_loss': loss}, f"{save_path}")
        else:
            torch.save({'epoch': epochID,
                        'img_encoder': self.img_encoder.state_dict(),
                        'cap_encoder': self.cap_encoder.state_dict(),
                        'discriminator': self.discriminator.state_dict(),
                        'temp_para': self.temp_para,
                        'best_loss': loss}, f"{save_path}")
    
    def emb_loss(self, imgs, caps, max_violation=True):
        scores = imgs @ caps.T / self.temp_para
        diagonal = scores.diag().view(len(imgs), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        cost_s = (self.margin_loss + scores - d1).clamp(min=0)
        cost_im = (self.margin_loss + scores - d2).clamp(min=0)
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask).to(imgs.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)
        if max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.mean() + cost_im.mean() 
    
    def itm_loss(self, imgs, cap, sim_i2t, sim_t2i, idx):
        # Find negative
        with torch.no_grad():
            bs = imgs.size(0)
            weights_i2t = F.softmax(sim_i2t[:,:bs]+1e-4,dim=1)
            weights_t2i = F.softmax(sim_t2i[:,:bs]+1e-4,dim=1)
            mask = torch.eq(idx, idx.T)
            mask = mask.to(self.device)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0) 
        # select a negative image for each text
        img_enc_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            img_enc_neg.append(imgs[neg_idx])
        img_enc_neg = torch.stack(img_enc_neg,dim=0) 

        # select a negative text for each image
        cap_enc_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            cap_enc_neg.append(cap[neg_idx])
        cap_enc_neg = torch.stack(cap_enc_neg,dim=0)   

        cap_enc_all = torch.cat([cap, cap, cap_enc_neg],dim=0)     
        img_enc_all = torch.cat([imgs, img_enc_neg, imgs],dim=0)
        itm_labels = torch.cat([torch.ones(bs,dtype=torch.float),torch.zeros(2*bs,dtype=torch.float)],
                               dim=0).view(-1,1).to(imgs.device)

        disc = self.discriminator(img_enc_all, cap_enc_all)
        loss_itm = F.binary_cross_entropy(disc, itm_labels)
        return loss_itm
    
    def forward_batch(self, batch):
        with torch.no_grad():
            self.temp_para.clamp_(0.01, 0.5)
        img_dict, cap_dict, img_id = batch
        for key, value in img_dict.items():
            img_dict[key] = value.to(self.device)
        for key, value in cap_dict.items():
            cap_dict[key] = value.to(self.device)
            
        idx = img_id.view(-1,1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
        pos_idx = torch.eq(idx, idx_all).float()     
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)
        sim_targets = sim_targets.to(self.device)
        
        img_enc = self.img_encoder(img_dict)
        cap_enc = self.cap_encoder(cap_dict)
        # Normalize ???
        if self.l2_norm:
            img_enc = F.normalize(img_enc, dim=1)
            cap_enc = F.normalize(cap_enc, dim=1)
        
        with torch.no_grad():
            self._momentum_update()
            img_enc_m = self.img_encoder_m(img_dict) 
            cap_enc_m = self.cap_encoder_m(cap_dict) 
            if self.l2_norm:
                img_enc_m = F.normalize(img_enc_m, dim=1)
                cap_enc_m = F.normalize(cap_enc_m, dim=1)
            
            img_enc_all = torch.cat([img_enc_m.t(),self.image_queue.clone().detach()],dim=1) 
            cap_enc_all = torch.cat([cap_enc_m.t(),self.text_queue.clone().detach()],dim=1)

            if self.distill:               
                sim_i2t_m = img_enc_m @ cap_enc_all / self.temp_para
                sim_t2i_m = cap_enc_m @ img_enc_all / self.temp_para   
                sim_i2t_targets = self.alpha * F.softmax(sim_i2t_m, dim=1) + (1 - self.alpha) * sim_targets
                sim_t2i_targets = self.alpha * F.softmax(sim_t2i_m, dim=1) + (1 - self.alpha) * sim_targets 
        
        sim_i2t = img_enc @ cap_enc_all / self.temp_para
        sim_t2i = cap_enc @ img_enc_all / self.temp_para 

        if self.distill:
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 
        else:
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()   

        loss_nll = 0.3*loss_i2t + 0.7*loss_t2i
        
        self._dequeue_and_enqueue(img_enc_m, cap_enc_m, idx)
        
        loss_emb = self.emb_loss(img_enc, cap_enc)
        loss_itm = self.itm_loss(img_enc, cap_enc, sim_i2t, sim_t2i, idx)
        result = {'img_enc':img_enc, 'cap_enc': cap_enc, 'img_id': img_id, 
                  'loss_nll': loss_nll, 'loss_emb': loss_emb, 'loss_itm': loss_itm}
        return result
    
    def train_epoch(self, dataloader, epochID=0, writer=None):
        self.train_mode()
        loss_all_report = 0
        loss_emb_report = 0
        loss_nll_report = 0
        loss_itm_report = 0
        numb_iter = len(dataloader)
        for idx, batch in enumerate(dataloader):
            # Forward
            if epochID > 0:
                self.alpha = self.config['alpha']
            else:
                self.alpha = self.config['alpha']*min(1,idx/len(dataloader))
            br = self.forward_batch(batch)
            loss_emb, loss_nll, loss_itm = br['loss_emb'], br['loss_nll'], br['loss_itm']
            all_loss = self.weight_emb_loss*loss_emb + self.weight_nll_loss*loss_nll + self.weight_itm_loss*loss_itm
            
            # Update
            self.optimizer.zero_grad()
            all_loss.backward()
            
            # Plot Gradient
            # ie_l, ie_mi, ie_av, ie_ma = plot_grad_flow(self.img_encoder.named_parameters())
            # ce_l, ce_mi, ce_av, ce_ma = plot_grad_flow(self.cap_encoder.named_parameters())
            # dc_l, dc_mi, dc_av, dc_ma = plot_grad_flow(self.discriminator.named_parameters())
            
            self.optimizer.step()
            
            loss_all_report += all_loss.item()
            loss_emb_report += loss_emb.item()
            loss_nll_report += loss_nll.item()
            loss_itm_report += loss_itm.item()
            
            if writer is not None:
                wit = int(self.n_iters / 100)
                if (idx+1) % wit == 0 or idx == 0:
                    if self.temp > 0:
                        writer.add_scalars('TempPara', {'t': self.temp_para.item()}, epochID * np.floor(numb_iter/wit) + np.floor((idx+1)/wit))
                    writer.add_scalars('Training Iter', {'All': loss_all_report/(idx+1)}, epochID * np.floor(numb_iter/wit) + np.floor((idx+1)/wit))
                    writer.add_scalars('Training Iter', {'NLL': loss_nll_report/(idx+1)}, epochID * np.floor(numb_iter/wit) + np.floor((idx+1)/wit))
                    writer.add_scalars('Training Iter', {'EMB': loss_emb_report/(idx+1)}, epochID * np.floor(numb_iter/wit) + np.floor((idx+1)/wit))
                    writer.add_scalars('Training Iter', {'ITM': loss_itm_report/(idx+1)}, epochID * np.floor(numb_iter/wit) + np.floor((idx+1)/wit))
                    writer.add_scalars('Training Iter', {'CEMB': loss_emb.item()}, epochID * np.floor(numb_iter/wit) + np.floor((idx+1)/wit))
                    writer.add_scalars('Training Iter', {'CNLL': loss_nll.item()}, epochID * np.floor(numb_iter/wit) + np.floor((idx+1)/wit))
                    writer.add_scalars('Training Iter', {'CITM': loss_itm.item()}, epochID * np.floor(numb_iter/wit) + np.floor((idx+1)/wit))

                    # for inn, n in enumerate(ie_l):
                    #     writer.add_scalars('GF-AvgIMG', {n: ie_av[inn]}, epochID * np.floor(numb_iter/wit) + np.floor((idx+1)/wit))
                    # for inn, n in enumerate(ce_l):
                    #     writer.add_scalars('GF-AvgCAP', {n: ce_av[inn]}, epochID * np.floor(numb_iter/wit) + np.floor((idx+1)/wit))
                    # for inn, n in enumerate(dc_l):
                    #     writer.add_scalars('GF-AvgDC', {n: dc_av[inn]}, epochID * np.floor(numb_iter/wit) + np.floor((idx+1)/wit))
            
                if self.Tmax > 0 or self.T0 > 0:
                    current_lr = self.scheduler.get_last_lr()
                    writer.add_scalars('Learning Rate', {'lr': current_lr}, epochID * np.floor(numb_iter/wit) + np.floor((idx+1)/wit))
                    
            if self.Tmax > 0: 
                self.scheduler.step()
            if self.T0 > 0: # cosine scheduler
                self.scheduler.step(epochID + idx / self.n_iters)       
        loss_all_report = round(loss_all_report/(idx+1), 6)
        loss_emb_report = round(loss_emb_report/(idx+1), 6)
        loss_nll_report = round(loss_nll_report/(idx+1), 6)
        loss_itm_report = round(loss_itm_report/(idx+1), 6)
        loss_dict = {'all': loss_all_report, 'emb': loss_emb_report, 'nll': loss_nll_report, 'itm': loss_itm_report}
        return loss_dict
    
    def evaluate_with_list_id(self, list_img_id, list_cap_id, para_standardize=None, apply_temp=True, with_itm=False):
        dataset_img = Image_Dataset(list_img_id=list_img_id, config=self.config,
                                    para_standardize=para_standardize)
        dataset_txt = Caption_Dataset(list_img_id_cap_id=list_cap_id, config=self.config,
                                      para_standardize=para_standardize)
        dataloader_img = make_dataloader(dataset_img, branch='img', batch_size=int(self.config['batch_size']/2), shuffle=False)
        dataloader_txt = make_dataloader(dataset_txt, branch='txt', batch_size=int(self.config['batch_size']/2), shuffle=False)
        img_enc = self.eval_encode(dataloader_img, branch='img', with_itm=with_itm)
        cap_enc = self.eval_encode(dataloader_txt, branch='txt', with_itm=with_itm)
        if apply_temp:
            sims = img_enc @ cap_enc.T / self.temp_para
        else:
            sims = img_enc @ cap_enc.T 
        sims = sims.cpu().numpy()
        r1i, r5i, r10i, r1t, r5t, r10t = evaluate_recall(sims, mode='both')
        rall = r1i + r5i + r10i + r1t + r5t + r10t
        loss = 6 - rall
        return (r1i, r5i, r10i, r1t, r5t, r10t), loss
            
    def train(self, train_img_id, train_cap_id,  para_standardize, train_img_aug_cap_id=None, val_img_id=None, val_cap_id=None, num_epoch=10, model_name='best'):
        if train_img_aug_cap_id is None:
            dataset_train = Image_Caption_Dataset(list_img_id_cap_id=train_cap_id, 
                                                  config=self.config, para_standardize=para_standardize)
        else:
            dataset_train = Image_Caption_Dataset_Augment(list_img_id_cap_id=train_img_aug_cap_id, 
                                                          config=self.config, para_standardize=para_standardize)
            
        save_dir = f"{self.out_dir}/{model_name}"
        Path(f"{save_dir}").mkdir(parents=True, exist_ok=True)
        
        self.load_model(save_path=self.pretrained_path)
        
         ## REPORT ##
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        timestampLaunch = timestampDate + '-' + timestampTime
        writer = SummaryWriter(f"{save_dir}/{timestampLaunch}/")
        
        count_change_loss = 0
        
        loss_best = 100000
        if self.T0 <= 0 and self.Tmax <= 0: # plateau scheduler
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor = self.factor, patience=self.patience, 
                                               mode = 'min', verbose=True, min_lr=self.min_lr)
        elif self.Tmax <= 0: # cosine scheduler
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=self.T0, T_mult=self.Tmul, eta_min=self.min_lr) 
        elif self.T0 <=0:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.Tmax, eta_min=self.min_lr) 
        
        with mlflow.start_run(experiment_id = self.experiment_id, 
                              run_name = self.config_name):
            mlflow.log_artifact(self.config_path)
            # mlflow.log_dict(self.config, self.config_name)
            params = {'Numb_Para': self.count_parameters()}
            mlflow.log_params(params)
            best_epoch = 0
            
            for idx_epoch in range(num_epoch):
                dataset_train.shuffle(seed=1509+idx_epoch)
                dataloader_train = make_dataloader(dataset_train, branch='both', batch_size=self.config['batch_size'], shuffle=True)
                self.n_iters = len(dataloader_train)
                loss_tr_dict = self.train_epoch(dataloader_train, idx_epoch, writer)
                loss_tr_all, loss_tr_emb, loss_tr_nll, loss_tr_itm = loss_tr_dict['all'], loss_tr_dict['emb'], loss_tr_dict['nll'], loss_tr_dict['itm']

                with torch.no_grad():
                    timestampTime = time.strftime("%H%M%S")
                    timestampDate = time.strftime("%d%m%Y")
                    timestampEND = timestampDate + '-' + timestampTime
                    apply_temp = True if self.temp > 0 else False

                    if val_img_id is not None and val_cap_id is not None: 
                        r, loss_rall = self.evaluate_with_list_id(val_img_id, val_cap_id, para_standardize, apply_temp)
                        r1i, r5i, r10i, r1t, r5t, r10t = r
                        loss_update = loss_rall
                    else:
                        loss_update = loss_tr_all

                    random.seed(1509+idx_epoch)
                    idx_img = random.sample(range(0, len(train_img_id)), 1014)
                    train_img_id_sl = [train_img_id[x] for x in idx_img]
                    idx_cap = [5*x for x in idx_img]
                    train_cap_id_sl = [train_cap_id[x:x+5] for x in idx_cap]
                    train_cap_id_sl = list(itertools.chain.from_iterable(train_cap_id_sl))
                    tr, loss_trall = self.evaluate_with_list_id(train_img_id_sl, train_cap_id_sl, para_standardize, apply_temp)
                    tr1i, tr5i, tr10i, tr1t, tr5t, tr10t = tr

                if self.T0 <= 0 and self.Tmax <= 0:
                    self.scheduler.step(loss_update)

                if loss_update <= (loss_best-self.thres_loss):
                    count_change_loss = 0
                    best_epoch = idx_epoch
                    print(f"[SAVE MODEL]")
                    self.save_model(loss=loss_update, epochID=idx_epoch, save_path=f"{save_dir}/best.pth.tar", optimizer=False)
                    loss_best = loss_update
                    info_txt = f"Epoch {idx_epoch}/{num_epoch-1} [{timestampEND} --- SAVE MODEL]\n"
                    metrics = {'temp_para': self.temp_para.item(),
                               'best_epoch':best_epoch, 
                               'current_epoch':idx_epoch, 
                               'Ri': np.round(r1i+r5i+r10i, 6),
                               'Rt': np.round(r1t+r5t+r10t, 6),
                               'Rall': np.round(r1i+r5i+r10i+r1t+r5t+r10t, 6),
                               'TRi': np.round(tr1i+tr5i+tr10i, 6),
                               'TRt': np.round(tr1t+tr5t+tr10t, 6),
                               'TRall': np.round(tr1i+tr5i+tr10i+tr1t+tr5t+tr10t, 6),
                               'Ctemp_para': self.temp_para.item(),
                               'CTRi': np.round(tr1i+tr5i+tr10i, 6),
                               'CTRt': np.round(tr1t+tr5t+tr10t, 6),
                               'CTRall': np.round(tr1i+tr5i+tr10i+tr1t+tr5t+tr10t, 6)
                               }
                    mlflow.log_metrics(metrics)

                else:
                    metrics = {'Ctemp_para': self.temp_para.item(),
                               'current_epoch':idx_epoch,
                               'CTRi': np.round(tr1i+tr5i+tr10i, 6),
                               'CTRt': np.round(tr1t+tr5t+tr10t, 6),
                               'CTRall': np.round(tr1i+tr5i+tr10i+tr1t+tr5t+tr10t, 6)
                               }
                    mlflow.log_metrics(metrics)
                    
                    count_change_loss += 1
                    info_txt = f"Epoch {idx_epoch}/{num_epoch-1} [{timestampEND}]\n"

                info_txt += f"Loss Train: {loss_tr_all}\nLoss Train Emb: {loss_tr_emb}\nLoss Train Nll: {loss_tr_nll}\nLoss Train Itm: {loss_tr_itm}\n"

                if val_img_id is not None and val_cap_id is not None:
                    info_txt += f"R1i: {np.round(r1i,4)}\nR5i: {np.round(r5i,4)}\nR10i: {np.round(r10i,4)}\n"
                    info_txt += f"R1t: {np.round(r1t,4)}\nR5t: {np.round(r5t,4)}\nR10t: {np.round(r10t,4)}\n"
                    info_txt += f"Ri: {np.round(r1i+r5i+r10i,4)}\nRt: {np.round(r1t+r5t+r10t,4)}\n"
                    info_txt += f"Rall: {np.round(r1i+r5i+r10i+r1t+r5t+r10t,4)}\n"
                    writer.add_scalars('Recall Epoch', {'R1i': r1i}, idx_epoch)
                    writer.add_scalars('Recall Epoch', {'R5i': r5i}, idx_epoch)
                    writer.add_scalars('Recall Epoch', {'R10i': r10i}, idx_epoch)
                    writer.add_scalars('Recall Epoch', {'R1t': r1t}, idx_epoch)
                    writer.add_scalars('Recall Epoch', {'R5t': r5t}, idx_epoch)
                    writer.add_scalars('Recall Epoch', {'R10t': r10t}, idx_epoch)
                    writer.add_scalars('Recall Epoch', {'LoRe': loss_rall}, idx_epoch)

                info_txt += f"TR1i: {np.round(tr1i,4)}\nTR5i: {np.round(tr5i,4)}\nTR10i: {np.round(tr10i,4)}\n"
                info_txt += f"TR1t: {np.round(tr1t,4)}\nTR5t: {np.round(tr5t,4)}\nTR10t: {np.round(tr10t,4)}\n"
                info_txt += f"TRi: {np.round(tr1i+tr5i+tr10i,4)}\nTRt: {np.round(tr1t+tr5t+tr10t,4)}\n"
                info_txt += f"TRall: {np.round(tr1i+tr5i+tr10i+tr1t+tr5t+tr10t,4)}\n"
                writer.add_scalars('Recall Epoch', {'TR1i': tr1i}, idx_epoch)
                writer.add_scalars('Recall Epoch', {'TR5i': tr5i}, idx_epoch)
                writer.add_scalars('Recall Epoch', {'TR10i': tr10i}, idx_epoch)
                writer.add_scalars('Recall Epoch', {'TR1t': tr1t}, idx_epoch)
                writer.add_scalars('Recall Epoch', {'TR5t': tr5t}, idx_epoch)
                writer.add_scalars('Recall Epoch', {'TR10t': tr10t}, idx_epoch)
                writer.add_scalars('Recall Epoch', {'TLoRe': loss_trall}, idx_epoch)

                info_txt += f"--------\n"

                writer.add_scalars('Loss Epoch', {'TrAll': loss_tr_all}, idx_epoch)
                writer.add_scalars('Loss Epoch', {'TrEmb': loss_tr_emb}, idx_epoch)
                writer.add_scalars('Loss Epoch', {'TrNLL': loss_tr_nll}, idx_epoch)
                writer.add_scalars('Loss Epoch', {'TrITM': loss_tr_itm}, idx_epoch)

                if count_change_loss >= self.early_stop:
                    print(f'Early stopping: {count_change_loss} epoch not decrease the loss')
                    info_txt += "[EARLY STOPPING]\n"
                    break
                write_to_file(f"{save_dir}/TrainReport.log", info_txt)
                print(info_txt)

        writer.close()
        
    def count_parameters(self, trainable=True):
        total = 0
        if trainable:
            total += sum(p.numel() for p in self.img_encoder.parameters() if p.requires_grad)
            total += sum(p.numel() for p in self.cap_encoder.parameters() if p.requires_grad)
            total += sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
            total += sum(p.numel() for p in self.img_encoder_m.parameters() if p.requires_grad)
            total += sum(p.numel() for p in self.cap_encoder_m.parameters() if p.requires_grad)
        else:
            total += sum(p.numel() for p in self.img_encoder.parameters())
            total += sum(p.numel() for p in self.cap_encoder.parameters())
            total += sum(p.numel() for p in self.discriminator.parameters())
            total += sum(p.numel() for p in self.img_encoder_m.parameters())
            total += sum(p.numel() for p in self.cap_encoder_m.parameters())
        return total
    
    def eval_encode(self, dataloader, branch='img', with_itm=False):
        self.eval_mode()
        list_enc = torch.tensor(())
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                x_dict,_ = batch
                for key, value in x_dict.items():
                    x_dict[key] = value.to(self.device)
                if branch == 'img':
                    x_enc = self.img_encoder(x_dict)
                else:
                    x_enc = self.cap_encoder(x_dict)
                # Normalize ???
                if self.l2_norm:
                    x_enc = F.normalize(x_enc, dim=1)
                if with_itm:
                    if branch == 'img':
                        x_enc = self.discriminator.p1 * x_enc
                    else:
                        x_enc = self.discriminator.p2 * x_enc
                    x_enc = F.normalize(x_enc, dim=1)
                list_enc = torch.cat((list_enc, x_enc.detach().cpu()), 0)
        return list_enc
    
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        batch_size = image_feat.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feat.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feat.T
        self.idx_queue[:, ptr:ptr + batch_size] = idx.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr  