import os
import sys
import shutil

import time
import datetime
import numpy as np
import torch
import wandb

import rl_credit
import rl_credit.script_utils as utils
from rl_credit.model import ACModel


def train(env_id,
          model_dir_stem,
          wandb_dir,
          seed=1,
          num_procs=16,
          save_interval=20,
          total_frames=3*10**6,
          log_interval=1,
          algo_name='a2c',
          algo_kwargs={},
          recurrence=10,
):
    """
    env_id (str) : id of registered gym environment

    model_dir_stem (str) : name of dir under 'storage' folder 
        containing the model status dict and local logs

    wandb_dir (str) : wandb.run.dir, the location of wandb run output

    seed (int) : random seed (default: 1)

    num_procs (int) : number of processes (default: 16)

    save_interval (int) : number of updates between two saves
        (default: 20, 0 means no saving)

    total_frames (int) : total number of frames of training
        (default: 3e7)

    log_interval (int) : number of updates between two logs
        (default: 1)

    algo_name (str) : name of algorithm, only a2c supported

    algo_kwargs (dict) : kwargs to pass into algorithm

    recurrence (int) : number of time-steps gradient is backpropagated 
        (default: 10). If > 1, a LSTM is added to the model to have memory.
    """
    # set run dir
    model_dir = utils.get_model_dir(model_dir_stem)

    # set plots output dir, only for attention model
    plots_dir = model_dir

    # Load loggers
    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(sys.argv[0]))

    # Set seed for all randomness sources
    utils.seed(seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_logger.info(f"Device: {device}\n")

    # Load environments
    envs = []
    for i in range(num_procs):
        envs.append(utils.make_env(env_id, seed + 10000 * i))
    txt_logger.info("Environments loaded\n")

    # Load training status
    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    use_mem = recurrence > 1
    if algo_name in ('a2c', 'tvt'):
        acmodel = ACModel(obs_space, envs[0].action_space, use_mem, False)
    else:
        raise ValueError("Unrecognized algo")

    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo
    algo_kwargs = algo_kwargs.copy()
    algo_kwargs.update({'acmodel': acmodel,
                        'envs': envs,
                        'device': device,
                        'recurrence': recurrence,
                        'preprocess_obss': preprocess_obss})
    if algo_name == 'a2c':
        algo = rl_credit.A2CAlgo(**algo_kwargs)
    elif algo_name == 'tvt':
        algo = rl_credit.AttentionQAlgo(**algo_kwargs, plots_dir=plots_dir)

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Main training loop

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < total_frames:

        update_start_time = time.time()

        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)

        logs = {**logs1, **logs2}

        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs
        if update % log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
            last_reward_per_episode = utils.synthesize(logs["last_reward_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger_output = "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"

            if "hca_loss" in logs:
                txt_logger_output += " | hca {:.2f}"
                header += ["hca_loss"]
                data += [logs["hca_loss"]]
            txt_logger.info(txt_logger_output.format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            header += ["last_reward_" + key for key in last_reward_per_episode.keys()]
            data += last_reward_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

        if save_interval > 0 and update % save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(),
                      "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

        # Wandb logging
        logs['returns_per_episode_std'] = np.std(logs.pop('return_per_episode'))
        logs['num_frames_per_episode_std'] = np.std(logs.pop('num_frames_per_episode'))
        logs['last_reward_per_episode_std'] = np.std(logs.pop('last_reward_per_episode'))
        logs.pop('reshaped_return_per_episode')
        for x in ('mean', 'min', 'max'):
            logs[f'return_per_episode_{x}'] = return_per_episode[x]
            logs[f'frames_per_episode_{x}'] = num_frames_per_episode[x]
            logs[f'last_reward_per_episode_{x}'] = last_reward_per_episode[x]
        logs['update_number'] = update
        logs['elapsed_time'] = duration
        wandb.log(logs, step=num_frames)

    # At end of run, copy model dir into wandb dir (includes csv logs and model dict)
    shutil.copytree(model_dir, os.path.join(wandb_dir, 'model'))
    return algo  # for interactive debugging
