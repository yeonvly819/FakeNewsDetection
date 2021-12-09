import warnings
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer
import torch
from transformers import AdamW
import argparse
import time
import datetime
import os

from transformers.utils.dummy_pt_objects import get_linear_schedule_with_warmup
from smart_batch import make_smart_batches

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=4, type=int,
                    help='model batch size')
parser.add_argument('--epochs', default=20, type=int, help='training epochs')
parser.add_argument('--mode', default='normal', type=str, help='normal / smart_batch')
args = parser.parse_args()

with open('no_split_news.txt', 'r') as f:
        news_data = f.readlines()

bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
# model = AutoModel.from_pretrained(bert_model_name)
model = BertForMaskedLM.from_pretrained(bert_model_name)

# print(tokenizer.mask_token_id)
# print(tokenizer.pad_token_id)

# print(model)

print()
max_len = model.bert.embeddings.position_embeddings.state_dict()['weight'].shape[0]
print('max_len:', max_len)

if args.mode == 'normal':
    class NewsDataset(torch.utils.data.Dataset):
            def __init__(self, full_text, tokenizer, max_len):
                self.texts = full_text
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                content = self.texts[idx]
                encoding = self.tokenizer.encode_plus(
                    text = content,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    truncation=True,
                    return_token_type_ids=True,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                encoding['labels'] = encoding.input_ids.detach().clone()

                rand = torch.rand(encoding['input_ids'].shape)
                mask_arr = (rand < 0.15) * (encoding.input_ids != 101) * (encoding.input_ids != 102) * (encoding.input_ids != 0)
                selection = []
                for i in range(mask_arr.shape[0]):
                        selection.append(
                            torch.flatten(mask_arr[i].nonzero()).tolist()
                        )

                for i in range(mask_arr.shape[0]):
                        encoding.input_ids[i, selection[i]] = 103

                return {
                'document_text': content,
                'input_ids': encoding['input_ids'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': encoding['labels'].flatten()
                }

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.train()
    dataset = NewsDataset(news_data, tokenizer, max_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # data = next(iter(dataloader))
    # print(data['input_ids'][2])
    # print('==============================================================================')
    # print(tokenizer.decode(data['input_ids'][2]))

    # optimizer
    optim = AdamW(model.parameters(), lr=1e-5)

    tot_loss = 0
    best_loss = 1000000
    
    # train
    epochs = args.epochs
    for epoch in range(epochs):
        print('===== Epoch {:} / {:} ====='.format(epoch, args.epochs))
        
        start = time.time()
        
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            optim.zero_grad()

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            
            tot_loss += loss.item()
        
        epoch_loss = tot_loss / len(input_ids)
        print(f'epoch.{epoch} \ntrain loss: {epoch_loss}')
        
        sec = time.time() - start
        times = str(datetime.timedelta(seconds=sec)).split('.')
        times = times[0]
        
        print('Elapsed time: {}'.format(times))
        
        if not os.path.exists('./normal_models_BERT'):
            os.mkdir('./normal_models_BERT')
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            
            model_dir = './normal_models_BERT'
            state = {'model':model.state_dict()}
            now = time.strftime('%m_%d_%H_%M')
            
            torch.save(state, os.path.join(model_dir, '_'.join([now, 'EPOCH', str(epoch), 'loss', str(round(epoch_loss, 4))]) + '.pth'))
            
            print('model saved')

        tot_loss = 0

elif args.mode=='smart_batch':
    # use smart_batch
    # tokenizing
    print()
    print('Tokenizing whole data...')
    input_ids = []
    for idx, text in enumerate(news_data):
        encoding = tokenizer.encode_plus(
            text = text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=False,
            return_tensors='pt'
        )
        input_ids.append(encoding['input_ids'])
        
        if idx % 10000 == 0:
            print(f'{idx}/{len(news_data)}')


    input_ids = torch.cat(input_ids)
    print(input_ids.shape)
    
    labels = input_ids.detach().clone()
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < 0.15) * (input_ids != 101) * (input_ids != 102) * (input_ids != 0)
    selection = []
    for i in range(mask_arr.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )

    for i in range(mask_arr.shape[0]):
            input_ids[i, selection[i]] = 103

    group_ids = torch.arange(len(news_data))
    
    # make smart batch
    print()
    print('===== Making Smart Batches =====')
    batched_token_ids, batched_attn_masks, batched_labels, batch_ordered_group_ids = make_smart_batches(input_ids.tolist(),
                                                                                                        labels.tolist(),
                                                                                                        group_ids.tolist(),
                                                                                                        args.batch_size,
                                                                                                        max_len)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    optim = AdamW(model.parameters(), lr=1e-5)

    print()
    print('Training...')
    print()
    
    tot_loss = 0
    best_loss = 1
    
    for epoch in range(1, args.epochs+1) :
        
        print('===== Epoch {:} / {:} ====='.format(epoch, args.epochs))
        
        start = time.time()
        
        model.train()

        # train
        trn_outputs_lst = []; trn_labels_lst = []
        for batch_token_ids, batch_attn_masks, batch_labels in zip(batched_token_ids, batched_attn_masks, batched_labels):
            
            batch_token_ids = batch_token_ids.to(device, dtype=torch.long)
            batch_attn_masks = batch_attn_masks.to(device, dtype=torch.long)
            batch_labels = torch.Tensor(batch_labels).to(device, dtype=torch.long)

            outputs = model(batch_token_ids, attention_mask=batch_attn_masks, labels=batch_labels)
            
            loss = outputs.loss
            loss.backward()
            
            optim.step()
            optim.zero_grad()
            
            tot_loss += loss.item()
        
        epoch_loss = tot_loss / len(batch_token_ids)
        print(f'epoch.{epoch} \ntrain loss: {epoch_loss}')
        
        sec = time.time() - start
        times = str(datetime.timedelta(seconds=sec)).split('.')
        times = times[0]
        
        print('Elapsed time: {}'.format(times))
        
        if not os.path.exists('./models_BERT'):
            os.mkdir('./models_BERT')
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            
            model_dir = './models_BERT'
            state = {'model':model.state_dict()}
            now = time.strftime('%m_%d_%H_%M')
            
            torch.save(state, os.path.join(model_dir, '_'.join([now, 'EPOCH', str(epoch), 'loss', str(round(epoch_loss, 4))]) + '.pth'))
            
            print('model saved')

        tot_loss = 0
