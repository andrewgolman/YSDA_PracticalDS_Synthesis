import os
import time

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
import synthesise
from dataset import TrainDataset
from loss import FastSpeech2Loss
from optimizer import ScheduledOptim

device = torch.device('cuda')


def train(cfg, model_cfg, data_cfg, preprocessed_path, model):
    torch.manual_seed(0)

    model = nn.DataParallel(model).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(), betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay
    )
    scheduled_optim = ScheduledOptim(
        optimizer,
        model_cfg.decoder_hidden,
        cfg.n_warm_up_step,
        cfg.aneal_steps,
        cfg.aneal_rate,
        cfg.restore_step,
    )
    Loss = FastSpeech2Loss().to(device)
    print("Optimizer and Loss Function Defined.")

    # Load checkpoint if exists
    checkpoint_path = "./ckpt"
    try:
        checkpoint = torch.load(
            os.path.join(
                checkpoint_path, "checkpoint_{}.pth.tar".format(cfg.restore_step)
            )
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("\n---Model Restored at Step {}---\n".format(cfg.restore_step))
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    # Get dataset
    speaker_map_path = None
    if os.path.exists(checkpoint_path):
        speaker_map_path = os.path.join(checkpoint_path, "speakers.json")
        if not os.path.exists(speaker_map_path):
            speaker_map_path = None

    # Get dataset
    dataset = TrainDataset(
      preprocessed_path,
      speaker_map_path=speaker_map_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        num_workers=0,
    )

    # Init logger
    log_path = "./log"
    os.makedirs(log_path, exist_ok=True)
    train_logger = SummaryWriter(log_path)

    # Init synthesis directory
    synth_path = "./synth"
    os.makedirs(synth_path, exist_ok=True)

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()

    # Training
    model = model.train()
    total_step = cfg.epochs * len(loader) * cfg.batch_size
    current_step = cfg.restore_step
    for epoch in range(cfg.epochs):
        for i, batchs in enumerate(loader):
            for j, data_of_batch in enumerate(batchs):
                start_time = time.perf_counter()
                current_step += 1

                # Get Data
                id_ = data_of_batch["id"]
                speaker = torch.from_numpy(data_of_batch["speaker"]).long().to(device)
                mel_target = (
                    torch.from_numpy(data_of_batch["mel_target"]).float().to(device)
                )
                text = torch.from_numpy(data_of_batch["text"]).long().to(device)
                D = torch.from_numpy(data_of_batch["D"]).long().to(device)
                log_D = torch.from_numpy(np.log(data_of_batch["D"] + model_cfg.log_offset)).float().to(device)
                f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
                energy = torch.from_numpy(data_of_batch["energy"]).float().to(device)
                src_len = torch.from_numpy(data_of_batch["src_len"]).long().to(device)
                mel_len = torch.from_numpy(data_of_batch["mel_len"]).long().to(device)
                max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
                max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)

                # Forward
                output = model(
                    text,
                    src_len,
                    mel_len,
                    D,
                    f0,
                    energy,
                    max_src_len,
                    max_mel_len,
                    speaker=speaker,
                )
                (
                    mel_output,
                    mel_postnet_output,
                    log_duration_output,
                    _,
                    f0_output,
                    energy_output,
                    src_mask,
                    mel_mask,
                    _,
                ) = output[:9]

                # Cal Loss
                mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = Loss(
                    log_duration_output,
                    log_D,
                    f0_output,
                    f0,
                    energy_output,
                    energy,
                    mel_output,
                    mel_postnet_output,
                    mel_target,
                    ~src_mask,
                    ~mel_mask,
                )
                total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss

                # Logger
                t_l = total_loss.item()
                m_l = mel_loss.item()
                m_p_l = mel_postnet_loss.item()
                d_l = d_loss.item()
                f_l = f_loss.item()
                e_l = e_loss.item()

                # Backward
                total_loss = total_loss / cfg.acc_steps
                total_loss.backward()
                if current_step % cfg.acc_steps != 0:
                    continue
                if current_step > 900000:
                    break

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_thresh)

                # Update weights
                scheduled_optim.step_and_update_lr()
                scheduled_optim.zero_grad()

                # Print
                if current_step % cfg.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch + 1, cfg.epochs, current_step, total_step
                    )
                    str2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, F0 Loss: {:.4f}, Energy Loss: {:.4f};".format(
                        t_l, m_l, m_p_l, d_l, f_l, e_l
                    )
                    str3 = (
                        "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                            (Now - Start), (total_step - current_step) * np.mean(Time)
                        )
                    )

                    print("\n" + str1)
                    print(str2)
                    print(str3)

                    with open(os.path.join(log_path, "log.txt"), "a") as f_log:
                        f_log.write(str1 + "\n")
                        f_log.write(str2 + "\n")
                        f_log.write(str3 + "\n")
                        f_log.write("\n")

                    train_logger.add_scalar("Loss/total_loss", t_l, current_step)
                    train_logger.add_scalar("Loss/mel_loss", m_l, current_step)
                    train_logger.add_scalar(
                        "Loss/mel_postnet_loss", m_p_l, current_step
                    )
                    train_logger.add_scalar("Loss/duration_loss", d_l, current_step)
                    train_logger.add_scalar("Loss/F0_loss", f_l, current_step)
                    train_logger.add_scalar("Loss/energy_loss", e_l, current_step)

                if current_step % cfg.save_step == 0:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(
                            checkpoint_path,
                            "checkpoint_{}.pth.tar".format(current_step),
                        ),
                    )
                    print("save model at step {} ...".format(current_step))

                if current_step % cfg.synth_step == 0:
                    basename = id_[0]
                    mel_length = mel_len[0].item()
                    mel_postnet = (
                        mel_postnet_output[0:1, :mel_length].detach().transpose(1, 2)
                    )
                    synthesise.vocoder_infer(
                        mel_postnet,
                        [
                            os.path.join(
                                synth_path,
                                "step_{}_with_postnet_{}.wav".format(
                                    current_step, basename
                                ),
                            )
                        ],
                        data_cfg.sampling_rate,
                    )

                if len(Time) == cfg.clear_Time:
                    Time = Time[:-1]
                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
