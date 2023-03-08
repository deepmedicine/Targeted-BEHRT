import pytorch_pretrained_bert as Bert

def VAEadam(params, config=None):
    if config is None:
        config = {
            'lr': 3e-5,
            'warmup_proportion': 0.1
        }
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'Eps']
    vae= ['VAE']
    no_decayfull = no_decay+vae
#     print( {'params': [n for n, p in params if not any(nd in n for nd in no_decayfull)], 'weight_decay': 0.01, 'lr': config['lr']},
#         {'params': [n for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': config['lr']},
#         {'params': [n for n, p in params if any(nd in n for nd in vae)], 'weight_decay': 0.0, 'lr':1e-3 }
# )
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decayfull)], 'weight_decay': 0.01, 'lr': config['lr']},
        {'params': [p for n, p in params if (any(nd in n for nd in no_decay) and 'VAE' not in n)], 'weight_decay': 0.0, 'lr': config['lr']},
        {'params': [p for n, p in params if any(nd in n for nd in vae)], 'weight_decay': 0.0, 'lr':1e-3 }

    ]

    optim = Bert.optimization.BertAdam(optimizer_grouped_parameters,
                                       warmup=config['warmup_proportion'])
    return optim


def adam(params, config=None):
    if config is None:
        config = {
            'lr': 3e-5,
            'warmup_proportion': 0.1
        }
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'Eps','VAE']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optim = Bert.optimization.BertAdam(optimizer_grouped_parameters,
                                       lr=config['lr'],
                                       warmup=config['warmup_proportion'])
    return optim

def GPadam(params, gpLR, config=None):
    if config is None:
        config = {
            'lr': 3e-5,
            'warmup_proportion': 0.1
        }
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight','Eps']
    gp = ['GP']



    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay) and not any(nd in n for nd in gp)], 'weight_decay': 0.01 , 'lr': config['lr'], 'warmup_proportion': 0.1},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay) and not any(nd in n for nd in gp)], 'weight_decay': 0.0, 'lr': config['lr'], 'warmup_proportion': 0.1},
        {'params': [p for n, p in params if any(nd in n for nd in gp)], 'lr': gpLR}
    ]

    print([
        {'params': [n for n, p in params if not any(nd in n for nd in no_decay) and not any(nd in n for nd in gp)], 'weight_decay': 0.01 , 'lr': config['lr'], 'warmup_proportion': 0.1},
        {'params': [n for n, p in params if any(nd in n for nd in no_decay) and not any(nd in n for nd in gp)], 'weight_decay': 0.0, 'lr': config['lr'], 'warmup_proportion': 0.1},
        {'params': [n for n, p in params if any(nd in n for nd in gp)], 'lr': gpLR}
    ])
    optim = Bert.optimization.BertAdam(optimizer_grouped_parameters)
    return optim
