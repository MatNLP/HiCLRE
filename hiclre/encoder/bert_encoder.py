import logging
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from .base_encoder import BaseEncoder
import random
from torch.nn import functional as F
# from textda.data_expansion import *




class BERTEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768 * 2
        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, token, att_mask, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens torch.Size([96, 128])
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)  torch.Size([96, 128])
        Return:
            (B, H), representations for sentences
        """

        hidden = self.bert(input_ids=token, attention_mask=att_mask).last_hidden_state   #torch.Size([96, 128, 768])
        return hidden.sum(1)

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']



        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
        
        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        if self.mask_entity:
            ent0 = ['[unused4]'] if not rev else ['[unused5]']
            ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']

        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()


        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2


class BERTEntityEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768 * 2
        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
    def encoder_entity_description(self):
        pass
    def forward(self, head_end, tail_end, token, att_mask, pos1, pos2, train=True):
        """
        Args:
            token: (B, L), index of tokens  torch.Size([24*2=48, 128])
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter   torch.Size([32, 1])
        Return:
            (B, 2H), representations for sentences
        """
        if train :
            hidden = self.bert(input_ids=token, attention_mask=att_mask).last_hidden_state

            # Get entity start hidden state
            onehot_head_start = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)   torch.Size([32, 128])
            onehot_tail_start = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
            onehot_head_end = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)   torch.Size([32, 128])
            onehot_tail_end = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
            onehot_head_start = onehot_head_start.scatter_(1, pos1, 1)  # torch.Size([32, 128])
            onehot_tail_start = onehot_tail_start.scatter_(1, pos2, 1)
            onehot_head_end = onehot_head_end.scatter_(1, head_end, 1)
            onehot_tail_end = onehot_tail_end.scatter_(1, tail_end, 1)

            head_start_hidden = (onehot_head_start.unsqueeze(2) * hidden).sum(1)   
            tail_start_hidden = (onehot_tail_start.unsqueeze(2) * hidden).sum(1) 
            head_end_hidden = (onehot_head_end.unsqueeze(2) * hidden).sum(1)  
            tail_end_hidden = (onehot_tail_end.unsqueeze(2) * hidden).sum(1)
            head_hidden = (head_start_hidden + head_end_hidden) / 2
            tail_hidden = (tail_start_hidden + tail_end_hidden) / 2
            x = torch.cat([head_hidden, tail_hidden], 1)  
            x = self.linear(x) 
            return x, head_hidden, tail_hidden, head_start_hidden, tail_start_hidden
        else:
            hidden = self.bert(input_ids=token, attention_mask=att_mask).last_hidden_state

            # Get entity start hidden state
            onehot_head_start = torch.zeros(hidden.size()[:2]).float().to(
                hidden.device)  # (B, L)   torch.Size([32, 128])
            onehot_tail_start = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  
            onehot_head_end = torch.zeros(hidden.size()[:2]).float().to(hidden.device) 
            onehot_tail_end = torch.zeros(hidden.size()[:2]).float().to(hidden.device) 
            onehot_head_start = onehot_head_start.scatter_(1, pos1, 1)  
            onehot_tail_start = onehot_tail_start.scatter_(1, pos2, 1)
            onehot_head_end = onehot_head_end.scatter_(1, head_end, 1)
            onehot_tail_end = onehot_tail_end.scatter_(1, tail_end, 1)

            head_start_hidden = (onehot_head_start.unsqueeze(2) * hidden).sum(
                1)  
            tail_start_hidden = (onehot_tail_start.unsqueeze(2) * hidden).sum(1)  
            head_end_hidden = (onehot_head_end.unsqueeze(2) * hidden).sum(
                1) 
            tail_end_hidden = (onehot_tail_end.unsqueeze(2) * hidden).sum(1)
            head_hidden = (head_start_hidden + head_end_hidden) / 2
            tail_hidden = (tail_start_hidden + tail_end_hidden) / 2 

            x = torch.cat([head_hidden, tail_hidden], 1) 
            x = self.linear(x)  # torch.Size([32, 1536])
            return x, head_hidden, tail_hidden, head_start_hidden, tail_start_hidden



    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True


        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        sentence_index = item['sentence_index']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False


        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        if self.mask_entity:
            ent0 = ['[unused4]'] if not rev else ['[unused5]']
            ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']

        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]'] #['[CLS]', 'less', 'radical', 'than', 'chase', 'and', 'more', 'firmly', 'anti', '##sl', '##aver', '##y', 'than', '[unused2]', 'bates', '[unused3]', ',',
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        head_token_end = len(sent0 + ent0) if not rev else len(sent0 + ent0 + sent1 + ent1)
        tail_token_end = len(sent0 + ent0 + sent1 + ent1) if not rev else len(sent0 + ent0)
        head_token_end = min(self.max_length - 1, head_token_end)
        tail_token_end = min(self.max_length - 1, tail_token_end)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()
        head_token_end = torch.tensor([[head_token_end]]).long()
        tail_token_end = torch.tensor([[tail_token_end]]).long()
        sentence_index = torch.tensor([[sentence_index]]).long()


        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2, head_token_end, tail_token_end, sentence_index
