#thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
thresholds = [0.5]
init_lr = 1e-4
max_grad_norm = 0.1
optimizer = dict(type='AdamW',
    lr=init_lr, 
    weight_decay=0.0001,
)

backprop = True
track = False
viz = False
subset = 'train'

loss_weights = {
    'det_nll_loss': 1,
    'track_nll_loss': 1,
    'giou_loss': 1,
    'l1_loss': 1,
    'ce_loss': 1,
}

batch_size = 8
coco_batch_size = 0
num_iters = 1
subseq_len = -1 #entire sequence
save_fps = 15
train_fps = 1
bg_weight = 1


# num_steps = 2**13
# checkpoint_every = 1e8

# max_lr = init_lr
# min_lr = init_lr / 100

# scheduler = dict(type='OneCycleLR', 
    # eta_max=max_lr, 
    # pct_start=0.8, 
    # anneal_strategy='cos',
    # div_factor=max_lr / init_lr, 
    # final_div_factor=init_lr / min_lr,
    # total_steps=num_steps
# )
# print_every = 1

# num_workers = 0
# trainloader = dict(
    # batch_size=batch_size, 
    # shuffle=False,
    # num_workers=num_workers,
    # pin_memory=False,
    # drop_last=True
# )

# testloader = dict(
    # batch_size=1,
    # shuffle=False,
    # num_workers=num_workers,
    # pin_memory=False,
    # drop_last=False
# )
