# traintest_regression.py
import os
import time
import json
import datetime
import numpy as np
import torch
from torch import nn


from common import AverageMeter
from stats_regression import calculate_regression_stats


def train(audio_model, train_loader, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on', device)

    if device.type == "cuda" and not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)

    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total params: {:.3f} M'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Trainable params: {:.3f} M'.format(sum(p.numel() for p in trainables) / 1e6))

    optimizer = torch.optim.Adam(trainables, lr=args.lr, weight_decay=args.weight_decay, betas=(0.95, 0.999))
    clip_grad = float(getattr(args, 'clip_grad', 0.0) or 0.0)
    if clip_grad > 0:
        print('Gradient clipping enabled: max_norm={:.4f}'.format(clip_grad))

    if args.loss == 'MSE':
        loss_fn = nn.MSELoss()
    elif args.loss == 'L1':
        loss_fn = nn.L1Loss()
    elif args.loss == 'SmoothL1':
        loss_fn = nn.SmoothL1Loss(beta=1.0)
    else:
        raise ValueError('loss must be one of [MSE, L1, SmoothL1]')

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
        gamma=args.lrscheduler_decay
    )

    best_epoch = 0
    best_srcc = -np.inf
    best_rmse = np.inf
    global_step = 0
    exp_dir = args.exp_dir

    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'predictions'), exist_ok=True)

    result_rows = []

    for epoch in range(1, args.n_epochs + 1):
        audio_model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter()

        print('---------------')
        print(datetime.datetime.now())
        print(f'Epoch {epoch}/{args.n_epochs}')

        end_time = time.time()

        for i, (audio_input, mos) in enumerate(train_loader):
            data_time.update(time.time() - end_time)

            audio_input = audio_input.to(device, non_blocking=True)
            mos = mos.to(device, non_blocking=True)

            pred = audio_model(audio_input)      # [B]
            loss = loss_fn(pred, mos)

            optimizer.zero_grad()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss; check features, normalization and learning rate.')
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(audio_model.parameters(), clip_grad)
            optimizer.step()




            B = audio_input.size(0)
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            global_step += 1

            if global_step % args.n_print_steps == 0:
                print(
                    f'Epoch [{epoch}][{i}/{len(train_loader)}]  '
                    f'BatchTime {batch_time.avg:.4f}  '
                    f'DataTime {data_time.avg:.4f}  '
                    f'TrainLoss {loss_meter.avg:.6f}'
                )

        val_stats, val_loss, pred_arr, target_arr = validate(audio_model, val_loader, loss_fn, device)

        print('Validation:')
        print('  PLCC : {:.6f}'.format(val_stats['PLCC']))
        print('  SRCC : {:.6f}'.format(val_stats['SRCC']))
        print('  RMSE : {:.6f}'.format(val_stats['RMSE']))
        print('  MAE  : {:.6f}'.format(val_stats['MAE']))
        print('  Loss : {:.6f}'.format(val_loss))
        print('  LR   : {:.8f}'.format(optimizer.param_groups[0]['lr']))

        np.savetxt(os.path.join(exp_dir, 'predictions', f'pred_epoch_{epoch}.csv'), pred_arr, delimiter=',')
        np.savetxt(os.path.join(exp_dir, 'predictions', f'target_epoch_{epoch}.csv'), target_arr, delimiter=',')

        improved = False
        if val_stats['SRCC'] > best_srcc:
            improved = True
        elif np.isclose(val_stats['SRCC'], best_srcc) and val_stats['RMSE'] < best_rmse:
            improved = True

        if improved:
            best_srcc = val_stats['SRCC']
            best_rmse = val_stats['RMSE']
            best_epoch = epoch
            torch.save(audio_model.state_dict(), os.path.join(exp_dir, 'models', 'best_audio_model.pth'))

            best_info = {
                'best_epoch': best_epoch,
                'best_SRCC': best_srcc,
                'best_RMSE': best_rmse,
                'best_PLCC': val_stats['PLCC'],
                'best_MAE': val_stats['MAE'],
            }
            with open(os.path.join(exp_dir, 'best_result.json'), 'w', encoding='utf-8') as f:
                json.dump(best_info, f, indent=2)

        torch.save(audio_model.state_dict(), os.path.join(exp_dir, 'models', f'audio_model.{epoch}.pth'))

        result_rows.append([
            epoch,
            loss_meter.avg,
            val_loss,
            val_stats['PLCC'],
            val_stats['SRCC'],
            val_stats['RMSE'],
            val_stats['MAE'],
            optimizer.param_groups[0]['lr'],
        ])
        np.savetxt(
            os.path.join(exp_dir, 'result.csv'),
            np.asarray(result_rows, dtype=np.float64),
            delimiter=',',
            header='epoch,train_loss,val_loss,PLCC,SRCC,RMSE,MAE,lr',
            comments=''
        )

        scheduler.step()

    print('--------------- Training Finished ---------------')
    print(f'Best epoch: {best_epoch}')
    print(f'Best SRCC : {best_srcc:.6f}')
    print(f'Best RMSE : {best_rmse:.6f}')


def validate(audio_model, val_loader, loss_fn, device):
    audio_model.eval()

    preds = []
    targets = []
    loss_meter = AverageMeter()

    with torch.no_grad():
        for audio_input, mos in val_loader:
            audio_input = audio_input.to(device)
            mos = mos.to(device)

            pred = audio_model(audio_input)  # [B]
            loss = loss_fn(pred, mos)

            preds.append(pred.detach().cpu().numpy())
            targets.append(mos.detach().cpu().numpy())
            loss_meter.update(loss.item(), audio_input.size(0))

    pred_arr = np.concatenate(preds, axis=0).reshape(-1)
    target_arr = np.concatenate(targets, axis=0).reshape(-1)

    stats = calculate_regression_stats(pred_arr, target_arr)
    val_loss = float(loss_meter.avg)
    return stats, val_loss, pred_arr, target_arr
