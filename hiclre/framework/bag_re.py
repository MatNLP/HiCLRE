import torch
from torch import nn, optim
import json
from .data_loader import SentenceRELoader, BagRELoader
from .utils import AverageMeter
from tqdm import tqdm
import sys
sys.path.append("..")
import HiCLRE.hiclre.framework.data_loader as data_load_py
from transformers import BertModel, BertTokenizer
from scipy.spatial.distance import cdist
import torch.nn.functional as F

pretrain_path='/pythonCode/HiCLRE/pretrain/bert-base-uncased'
class BagRE(nn.Module):

    def __init__(self, 
                 model,
                 train_path, 
                 val_path,
                 test_path,
                 ckpt,
                 batch_size=32, 
                 max_epoch=100, 
                 lr=0.1, 
                 weight_decay=1e-5, 
                 opt='sgd',
                 bag_size=0,
                 tau=0.5,
                 lambda_1=0.4,
                 lambda_2=0.4,
                 lambda_3=0.1,
                 lambda_4=0.1,
                 loss_weight=False):
    
        super().__init__()
        self.max_epoch = max_epoch
        self.bag_size = bag_size
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.fc = nn.Linear(model.sentence_encoder.hidden_size, model.num_class)
        self.diag = nn.Parameter(torch.ones(model.sentence_encoder.hidden_size))
        self.softmax = nn.Softmax(-1)
        self.drop = nn.Dropout()
        self.fc = nn.Linear(model.sentence_encoder.hidden_size, len(model.rel2id))
        self.max_length=128
        self.tau = tau
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        # Load data
        if train_path != None:
            self.train_loader = BagRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True,
                # False,
                bag_size=bag_size,
                entpair_as_bag=False)

        if val_path != None:
            self.val_loader = BagRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                bag_size=bag_size,
                entpair_as_bag=True)

        if test_path != None:
            self.test_loader = BagRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                bag_size=bag_size,
                entpair_as_bag=True
            )
        # Model
        self.model = nn.DataParallel(model)   
        # Criterion
        if loss_weight:
            self.criterion = nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight)
        else:
            self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.model.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw':
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'bert_adam'.")
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt


    def train_model(self, metric='auc'):
        best_metric = 0      
        sentence_rep_memory = {}
        entity_rep_memory = {}
        align_list = []
        uniform_list = []

        for epoch in range(self.max_epoch):
            # Train
            self.train()
            print("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            avg_pos_acc = AverageMeter()

            # inner_bag_rep_memory = {}
            inner_sentence_rep_memory = {}
            inner_entity_rep_memory = {}
            num_no_key = 0

            t = tqdm(self.train_loader)
            t_len= t.__len__()
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]  
                bag_name = data[1] 
                scope = data[2]  
                args = data[3:7] 
                head_end = data[7]
                tail_end = data[8]
                sentence_index = data[9]
                sentence_index_list = (torch.squeeze(sentence_index.view(self.batch_size * self.bag_size,-1))).detach().cpu().numpy()

                logits, rep, rep_head, rep_tail, bag_perturb, sent_perturb, entity_perturb = self.model( head_end, tail_end, label, scope, *args, bag_size=self.bag_size) 

                #---------------------store sentence rep to memory in sentIndex order---------------------------
                rep_list = ((rep.view(self.batch_size * self.bag_size,-1)).detach().cpu().numpy()).tolist()
                sentenceIndex_rep_dic = dict(map(lambda x, y: [x, y], sentence_index_list, rep_list))
                inner_sentence_rep_memory.update(sentenceIndex_rep_dic)
                # ---------------------store sentence rep to memory in sentIndex order---------------------------

                # ---------------------store entity rep to memory in sentIndex order---------------------------
                rep_entity = torch.cat([rep_head, rep_tail], 1)
                rep_entity_list = (rep_entity.detach().cpu().numpy()).tolist()
                sentenceIndex_rep_entity_dic = dict(map(lambda x, y: [x, y], sentence_index_list, rep_entity_list))
                inner_entity_rep_memory.update(sentenceIndex_rep_entity_dic)
                # ---------------------store entity rep to memory in sentIndex order---------------------------


                loss_crossEnropy = self.criterion(logits, label)
                loss_crossEnropy.detach()


                # --------------------sentence-level ---------------------

                rep = rep.view(self.batch_size * self.bag_size, -1) 
                sent_perturb = sent_perturb.view(self.batch_size * self.bag_size, -1)  
                if epoch != 0:
                    last_epoch_rep_list = []
                    for i in range(len(sentence_index_list)):
                        
                        if sentence_index_list[i] in sentence_rep_memory.keys() :
                            last_epoch_rep_list.append(sentence_rep_memory[sentence_index_list[i]])
                        else:
                            num_no_key += 1
                            last_epoch_rep_list.append(torch.zeros(1536).detach().cpu().numpy().tolist())
                    last_epoch_rep_tensor = torch.tensor(last_epoch_rep_list)

                    sentence_simi = torch.cosine_similarity( rep, last_epoch_rep_tensor.to('cuda'))
                    sentence_simi = torch.unsqueeze(sentence_simi, dim=1)
                    sent_sim_expand = ((self.max_epoch - epoch) / self.max_epoch) * sentence_simi / torch.norm(sentence_simi, 2)
                    sent_perturb = sent_perturb + sent_sim_expand

                rep_after_perturb = rep + sent_perturb
                sim_cos = torch.cosine_similarity(rep, rep_after_perturb)

                sim_self = torch.cosine_similarity(rep.unsqueeze(1), rep.unsqueeze(0), dim=2)
                loss_sentence = - torch.log((torch.exp(sim_cos / self.tau)) / torch.exp(sim_self / self.tau).sum(0))
                # --------------------sentence-level ---------------------

                #--------------------bag-level ---------------------
                if epoch != 0:
                    batch_size = label.size(0) 
                    query = label.unsqueeze(1) 
                    att_mat = self.fc.weight[query]  
                    att_mat = att_mat * self.diag.unsqueeze(0) 
                    last_epoch_rep_tensor = last_epoch_rep_tensor.view(self.batch_size, self.bag_size, -1).to('cuda') 
                    att_score = (last_epoch_rep_tensor * att_mat).sum(-1)  
                    softmax_att_score = self.softmax(att_score)  
                    last_bag_rep_3dim = (softmax_att_score.unsqueeze(-1) * last_epoch_rep_tensor) 
                    last_bag_rep = last_bag_rep_3dim.sum(1)
                    last_bag_rep = self.drop(last_bag_rep)  
                    last_bag_logits = self.fc(last_bag_rep) 

                    bag_simi = torch.cosine_similarity( logits, last_bag_logits )  
                    bag_simi = torch.unsqueeze(bag_simi, dim=1)
                   
                    bag_sim_expand = ((self.max_epoch - epoch) / self.max_epoch ) * bag_simi / torch.norm(bag_simi, 2)
                    bag_perturb = bag_perturb + bag_sim_expand

                logits_after_perturb = logits + bag_perturb  
                simcse_adv_cos = torch.cosine_similarity(logits, logits_after_perturb)  
                adv_sim_self = torch.cosine_similarity(logits.unsqueeze(1), logits.unsqueeze(0), dim=2)
                loss_bag = - torch.log((torch.exp(simcse_adv_cos / self.tau)) / torch.exp(adv_sim_self / self.tau).sum(0))

                #----------------align and uniform-------------------------
                align = (sim_cos) / (self.batch_size * self.bag_size)
                align = align.detach().cpu().numpy().tolist()
                align_list = align_list + align
                uniform = (-  2 * sim_self).sum(0) / (self.batch_size * self.bag_size)
                uniform = uniform.detach().cpu().numpy().tolist()
                uniform_list = uniform_list + uniform
                # ----------------align and uniform------------------------

                # --------------------bag-level ---------------------

                # --------------------entity-level ---------------------
                rep_entity = rep_entity.view(self.batch_size * self.bag_size, -1)  
                entity_perturb = entity_perturb.view(self.batch_size * self.bag_size, -1)  
                if epoch != 0:
                    last_epoch_rep_entity_list = []
                    for i in range(len(sentence_index_list)):
                        
                        if sentence_index_list[i] in entity_rep_memory.keys():
                            last_epoch_rep_entity_list.append(entity_rep_memory[sentence_index_list[i]])
                        else:
                            last_epoch_rep_entity_list.append(torch.zeros(1536).detach().cpu().numpy().tolist())
                    last_epoch_rep_entity_tensor = torch.tensor(last_epoch_rep_entity_list)

                    entity_simi = torch.cosine_similarity(rep_entity, last_epoch_rep_entity_tensor.to('cuda'))
                    entity_simi = torch.unsqueeze(entity_simi, dim=1)
                    entity_sim_expand = ((self.max_epoch - epoch) / self.max_epoch) * entity_simi / torch.norm(entity_simi, 2)
                    entity_perturb = entity_perturb + entity_sim_expand
                entity_rep_after_perturb = rep_entity + entity_perturb
                entity_sim_cos = torch.cosine_similarity(rep_entity, entity_rep_after_perturb)
                entity_sim_self = torch.cosine_similarity(rep_entity.unsqueeze(1), rep_entity.unsqueeze(0), dim=2)
                loss_entity = - torch.log((torch.exp(entity_sim_cos / self.tau)) / torch.exp(entity_sim_self / self.tau).sum(0))
                # --------------------entity-level ---------------------

                mean_loss_bag = (torch.sum(loss_bag, dim=0)) / (self.batch_size * self.bag_size)
                mean_loss_sentence = (torch.sum(loss_sentence, dim=0)) / (self.batch_size * self.bag_size) / 4
                mean_loss_entity = (torch.sum(loss_entity, dim=0)) / (self.batch_size * self.bag_size) / 4

                total_loss = self.lambda_1 *loss_crossEnropy + self.lambda_2*mean_loss_bag + self.lambda_3*mean_loss_sentence + self.lambda_4*mean_loss_entity

                score, pred = logits.max(-1) 
                acc = float((pred == label).long().sum()) / label.size(0)

                pos_total = (label != 0).long().sum()
                pos_correct = ((pred == label).long() * (label != 0).long()).sum()
                if pos_total > 0:
                    pos_acc = float(pos_correct) / float(pos_total)
                else:
                    pos_acc = 0

                avg_loss.update(total_loss.item(), 1)
                avg_acc.update(acc, 1)
                avg_pos_acc.update(pos_acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, pos_acc=avg_pos_acc.avg)

                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                #--------------------add more test operation in each epoch------------------------
                if iter % (int((t_len / 3) + 1)) == 0 and iter > 0:
                    val_index = (iter // (int((t_len / 3) + 1))) - 1
                    print("=== Epoch %d-%d val === " % (epoch,  val_index))
                    result = self.eval_model(self.val_loader)
                    print("AUC: %.4f" % result['auc'])
                    print("Micro F1: %.4f" % (result['max_micro_f1']))
                    if result[metric] > best_metric:
                        print("Best ckpt and saved.")
                        torch.save({'state_dict': self.model.module.state_dict()}, self.ckpt)
                        best_metric = result[metric]
                #--------------------add more test operation in each epoch------------------------
            sentence_rep_memory.update( inner_sentence_rep_memory )
            entity_rep_memory.update( inner_entity_rep_memory )

            # Val
            print("=== Epoch %d-2 val ===" % epoch)
            result = self.eval_model(self.val_loader)
            print("AUC: %.4f" % result['auc'])
            print("Micro F1: %.4f" % (result['max_micro_f1']))
            if result[metric] > best_metric:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.model.module.state_dict()}, self.ckpt)
                best_metric = result[metric]

        print("Best %s on val set: %f" % (metric, best_metric))
        # return align_list, uniform_list

    def eval_model(self, eval_loader):
        self.model.eval()
        with torch.no_grad():
            t = tqdm(eval_loader)
            pred_result = []
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                bag_name = data[1]
                scope = data[2]
                args = data[3:7]
                head_end = data[7]
                tail_end = data[8]
                sentence_index = data[9]
                logits, rep, rep_head, rep_tail, bag_perturb, sent_perturb, entity_perturb = self.model(head_end, tail_end,  None, scope, *args, train=False, bag_size=self.bag_size) # results after softmax

                logits = logits.cpu().numpy()
                for i in range(len(logits)):
                    for relid in range(self.model.module.num_class):
                        if self.model.module.id2rel[relid] != 'NA':
                            pred_result.append({
                                'entpair': bag_name[i][:2], 
                                'relation': self.model.module.id2rel[relid], 
                                'score': logits[i][relid],
                                'sentence_index': sentence_index[i]
                            })
            result = eval_loader.dataset.eval(pred_result)  
        return result

    def load_state_dict(self, state_dict):
        self.model.module.load_state_dict(state_dict)
