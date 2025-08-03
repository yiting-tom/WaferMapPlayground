# Mixed Wafer Map 38k
[google drive]("https://drive.google.com/file/d/1M59pX-lPqL9APBIbp2AKQRTvngeUK8Va/view")

## Experiments
### MobileNetV3Large Sparse 1920 e50 sz224 0.0001
```bash
/home/W20862/miniconda3/envs/wmc/lib/python3.12/site-packages/lightning/pytorch/utilities/parsing.py:209: Attribute 'model' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['model'])`.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
You are using a CUDA device ('NVIDIA GeForce RTX 4090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name    | Type              | Params | Mode
------------------------------------------------------
0 | loss_fn | BCEWithLogitsLoss | 0      | train
1 | model   | MobileNetV3Large  | 9.8 M  | train
2 | metrics | ModuleList        | 0      | train
------------------------------------------------------
6.8 M     Trainable params
3.0 M     Non-trainable params
9.8 M     Total params
39.098    Total estimated model params size (MB)
287       Modules in train mode
0         Modules in eval mode
/home/W20862/miniconda3/envs/wmc/lib/python3.12/site-packages/lightning/pytorch/loops/fit_loop.py:310: The number of training batches (12) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
Epoch 49: 100%|█| 12/12 [00:02<00:00,  4.93it/s, v_num=0, train_loss_step=0.197, val_loss`Trainer.fit` stopped: `max_epochs=50` reached.
Epoch 49: 100%|█| 12/12 [00:03<00:00,  3.90it/s, v_num=0, train_loss_step=0.197, val_loss
Training time: 191.93421363830566 seconds
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Testing DataLoader 0: 100%|████████████████████████████████| 2/2 [00:00<00:00, 48.40it/s]
─────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
─────────────────────────────────────────────────────────────────────────────────────────
        test_loss           0.1949172168970108
 test_multilabelaccuracy        0.923828125
 test_multilabelf1score     0.8726876974105835
test_multilabelprecision    0.8406708836555481
  test_multilabelrecall     0.9072397947311401
─────────────────────────────────────────────────────────────────────────────────────────
Testing time: 3.087925910949707 seconds
```

### TinyViT Sparse 1920 e50 sz224 0.0001