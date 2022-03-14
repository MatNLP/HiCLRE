import torch
from torch import nn, optim
from .base_model import BagRE
import numpy
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

class BagAttention(BagRE):
    """
    Instance attention for bag-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id, use_diag=True):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel
        if use_diag:
            self.use_diag = True
            self.diag = nn.Parameter(torch.ones(self.sentence_encoder.hidden_size)) 
        else:
            self.use_diag = False

    def infer(self, bag):
        """
        Args:
            bag: bag of sentences with the same entity pair
                [{
                  'text' or 'token': ..., 
                  'h': {'pos': [start, end], ...}, 
                  't': {'pos': [start, end], ...}
                }]
        Return:
            (relation, score)
        """
        self.eval()
        tokens = []
        pos1s = []
        pos2s = []
        masks = []
        for item in bag:
            token, pos1, pos2, mask = self.sentence_encoder.tokenize(item)
            tokens.append(token)
            pos1s.append(pos1)
            pos2s.append(pos2)
            masks.append(mask)
        tokens = torch.cat(tokens, 0).unsqueeze(0) 
        pos1s = torch.cat(pos1s, 0).unsqueeze(0)
        pos2s = torch.cat(pos2s, 0).unsqueeze(0)
        masks = torch.cat(masks, 0).unsqueeze(0)
        scope = torch.tensor([[0, len(bag)]]).long() 
        bag_logits = self.forward(None, scope, tokens, pos1s, pos2s, masks, train=False).squeeze(0) 
        score, pred = bag_logits.max(0)
        score = score.item()
        pred = pred.item()
        rel = self.id2rel[pred]
        return (rel, score)


    def forward(self,  head_end, tail_end, label,  scope, token, pos1, pos2, mask=None, train=True, bag_size=0):
        """
        Args:
            label: (B), label of the bag  =torch.Size([8])
            scope: (B), scope for each bag   =16
            token: (nsum, L), index of tokens   =torch.Size([8, 4, 128])
            pos1: (nsum, L), relative position to head entity  =torch.Size([32, 1])
            pos2: (nsum, L), relative position to tail entity
            mask: (nsum, L), used for piece-wise CNN
        Return:
            logits, (B, N)

        Dirty hack:
            When the encoder is BERT, the input is actually token, att_mask, pos1, pos2, but
            since the arguments are then fed into BERT encoder with the original order,
            the encoder can actually work out correclty.
        """
        if bag_size > 0:
            token = token.view(-1, token.size(-1))    
            pos1 = pos1.view(-1, pos1.size(-1))   
            pos2 = pos2.view(-1, pos2.size(-1))
            head_end = head_end.view(-1, head_end.size(-1))  
            tail_end = tail_end.view(-1, tail_end.size(-1))
            if mask is not None:
                mask = mask.view(-1, mask.size(-1))
        else:
            begin, end = scope[0][0], scope[-1][1]
            token = token[:, begin:end, :].view(-1, token.size(-1))
            pos1 = pos1[:, begin:end, :].view(-1, pos1.size(-1))
            pos2 = pos2[:, begin:end, :].view(-1, pos2.size(-1))
            head_end = head_end[:, begin:end, :].view(-1, head_end.size(-1))
            tail_end = tail_end[:, begin:end, :].view(-1, tail_end.size(-1))
            if mask is not None:
                mask = mask[:, begin:end, :].view(-1, mask.size(-1))
            scope = torch.sub(scope, torch.zeros_like(scope).fill_(begin))
        # Attention
        if train:
            if mask is not None:
                rep, rep_head, rep_tail, head_start_hidden, tail_start_hidden = self.sentence_encoder(head_end, tail_end, token, pos1, pos2, mask, train=train ) 
            else:
                rep, rep_head, rep_tail, head_start_hidden, tail_start_hidden = self.sentence_encoder(head_end, tail_end,token, pos1, pos2, mask, train=train)
            if bag_size == 0:
                bag_rep = []
                query = torch.zeros((rep.size(0))).long()
                if torch.cuda.is_available():
                    query = query.cuda()
                for i in range(len(scope)):
                    query[scope[i][0]:scope[i][1]] = label[i]
                att_mat = self.fc.weight[query] # (nsum, H)
                if self.use_diag:
                    att_mat = att_mat * self.diag.unsqueeze(0)
                att_score = (rep * att_mat).sum(-1) # (nsum)

                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]]
                    softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]]) 
                    bag_rep.append((softmax_att_score.unsqueeze(-1) * bag_mat).sum(0)) 
                bag_rep = torch.stack(bag_rep, 0)
            else:
                batch_size = label.size(0)    
                query = label.unsqueeze(1)
                att_mat = self.fc.weight[query]
                if self.use_diag:
                    att_mat = att_mat * self.diag.unsqueeze(0)   
                rep = rep.view(batch_size, bag_size, -1)   
                att_score = (rep * att_mat).sum(-1) 
                softmax_att_score = self.softmax(att_score) 
                bag_rep_3dim = (softmax_att_score.unsqueeze(-1) * rep) 
                bag_rep = bag_rep_3dim.sum(1) 

            #----------------add attention to every level---------------------
            entity_rep_3dim = torch.cat([rep_head, rep_tail], 1).view(batch_size, bag_size, -1)
            bag_focus_afterAttn = self.attn(rep, entity_rep_3dim, bag_rep_3dim)
            sentence_focus_afterAttn = self.attn(bag_rep_3dim, entity_rep_3dim, rep)
            entity_focus_afterAttn = self.attn(bag_rep_3dim, rep, entity_rep_3dim)

            entityRep_after_add_attRep = entity_rep_3dim + entity_focus_afterAttn
            sentenceRep_after_add_attRep = rep + sentence_focus_afterAttn
            bagRep_after_add_attRep =  bag_rep_3dim + bag_focus_afterAttn

            bag_rep_3dim_after_some_operation = self.some_operation_after_addAttnRep(label, scope, bag_size, entityRep_after_add_attRep, sentenceRep_after_add_attRep, bagRep_after_add_attRep)
            bag_rep = (bagRep_after_add_attRep + bag_rep_3dim_after_some_operation) / 2
            bag_rep = bag_rep.sum(1)

            #----------------add attention to every level---------------------



            bag_rep = self.drop(bag_rep) 
            bag_logits = self.fc(bag_rep) 


            self.criterion = nn.CrossEntropyLoss()
            loss_crossEnropy = self.criterion(bag_logits, label)
            adv_epsilon = 2
            loss_crossEnropy.detach()

            # ------------bag-level perturb--------------
            g = torch.autograd.grad(loss_crossEnropy, rep, create_graph=True, allow_unused=True) 
            g_tensor = self.fc(g[0].sum(1))
            bag_perturb = adv_epsilon / torch.norm(g[0], 2) * g_tensor  
            # ------------bag-level perturb--------------

            # ------------sentence-level perturb--------------
            g_sent_head = torch.autograd.grad(loss_crossEnropy, rep_head, create_graph=True, allow_unused=True) 
            g_sent_tail = torch.autograd.grad(loss_crossEnropy, rep_tail, create_graph=True, allow_unused=True)  
            g_sent = torch.cat([g_sent_head[0], g_sent_tail[0]], 1)
            sentence_adv_epsilon = 2
            sent_perturb = sentence_adv_epsilon / torch.norm(g_sent, 2) * g_sent 
            # ------------sentence-level perturb--------------

            # ------------entity-level perturb--------------
            g_entity_head = torch.autograd.grad(loss_crossEnropy, head_start_hidden, create_graph=True, allow_unused=True) 
            g_entity_tail = torch.autograd.grad(loss_crossEnropy, tail_start_hidden, create_graph=True, allow_unused=True) 
            g_entity = torch.cat([g_entity_head[0], g_entity_tail[0]], 1)
            entity_adv_epsilon = 2
            entity_perturb = entity_adv_epsilon / torch.norm(g_entity, 2) * g_entity 
            # ------------entity-level perturb--------------

        else:
            # val
            if bag_size == 0:
                rep = []
                bs = 256
                total_bs = len(token) // bs + (1 if len(token) % bs != 0 else 0)
                for b in range(total_bs):
                    with torch.no_grad():
                        left = bs * b
                        right = min(bs * (b + 1), len(token))
                        if mask is not None:        
                            rep.append(self.sentence_encoder(token[left:right], pos1[left:right], pos2[left:right], mask[left:right]).detach()) # (nsum, H) 
                        else:
                            rep.append(self.sentence_encoder(token[left:right], pos1[left:right], pos2[left:right]).detach()) # (nsum, H) 
                rep = torch.cat(rep, 0)

                bag_logits = []
                att_mat = self.fc.weight.transpose(0, 1)
                if self.use_diag:
                    att_mat = att_mat * self.diag.unsqueeze(1)
                att_score = torch.matmul(rep, att_mat) 
                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]] 
                    softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]].transpose(0, 1)) 
                    rep_for_each_rel = torch.matmul(softmax_att_score, bag_mat)
                    logit_for_each_rel = self.softmax(self.fc(rep_for_each_rel)) 
                    logit_for_each_rel = logit_for_each_rel.diag()
                    bag_logits.append(logit_for_each_rel)
                bag_logits = torch.stack(bag_logits, 0) # after **softmax**
            else:
                if mask is not None:
                    rep, rep_head, rep_tail, head_start_hidden, tail_start_hidden = self.sentence_encoder(head_end, tail_end, token, pos1, pos2, mask, train=train)
                else:
                    rep, rep_head, rep_tail, head_start_hidden, tail_start_hidden = self.sentence_encoder(head_end, tail_end, token, pos1, pos2, mask, train=train)

                batch_size = rep.size(0) // bag_size
                att_mat = self.fc.weight.transpose(0, 1)
                if self.use_diag:
                    att_mat = att_mat * self.diag.unsqueeze(1) 
                att_score = torch.matmul(rep, att_mat) 
                att_score = att_score.view(batch_size, bag_size, -1)
                rep = rep.view(batch_size, bag_size, -1) 
                softmax_att_score = self.softmax(att_score.transpose(1, 2)) 
                rep_for_each_rel = torch.matmul(softmax_att_score, rep) 
                bag_logits = self.softmax(self.fc(rep_for_each_rel)).diagonal(dim1=1, dim2=2) 
                bag_perturb = torch.zeros_like(bag_logits).cuda()
                sent_perturb = torch.zeros_like(rep).cuda()
                entity_perturb = torch.zeros_like(rep).cuda()
        return bag_logits, rep, rep_head, rep_tail, bag_perturb, sent_perturb, entity_perturb

    def attn(self, Q, K, V):

        K = K.permute(0, 2, 1)

        alpha = torch.matmul(Q, K)
        alpha = F.softmax(alpha, dim=2)

        attn_out = torch.matmul(alpha, V)

        return attn_out
    def some_operation_after_addAttnRep(self, label, scope,  bag_size, entityRep_after_add_attRep, sentenceRep_after_add_attRep, bagRep_after_add_attRep ):
        if bag_size == 0:
            bag_rep = []
            query = torch.zeros((sentenceRep_after_add_attRep.size(0))).long()
            if torch.cuda.is_available():
                query = query.cuda()
            for i in range(len(scope)):
                query[scope[i][0]:scope[i][1]] = label[i]
            att_mat = self.fc.weight[query] 
            if self.use_diag:
                att_mat = att_mat * self.diag.unsqueeze(0)
            att_score = (sentenceRep_after_add_attRep * att_mat).sum(-1)  # (nsum)

            for i in range(len(scope)):
                bag_mat = sentenceRep_after_add_attRep[scope[i][0]:scope[i][1]]  
                softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]])  
                bag_rep.append((softmax_att_score.unsqueeze(-1) * bag_mat).sum(0)) 
            bag_rep = torch.stack(bag_rep, 0)   
        else:
            batch_size = label.size(0) 
            query = label.unsqueeze(1) 
            att_mat = self.fc.weight[query] 
            if self.use_diag:
                att_mat = att_mat * self.diag.unsqueeze(0)  
            rep = sentenceRep_after_add_attRep.view(batch_size, bag_size, -1) 
            att_score = (rep * att_mat).sum(-1)  
            softmax_att_score = self.softmax(att_score)  
            bag_rep_3dim = (softmax_att_score.unsqueeze(-1) * rep)  

        return bag_rep_3dim
