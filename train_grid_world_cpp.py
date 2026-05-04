#
# PPO/RecurrentPPO training for the Coverage Path Planning environment.
#
# Examples:
#   python train_grid_world_cpp.py train 5 3 200 1000000
#   python train_grid_world_cpp.py curriculum 10 12 400 500000 --model data/model_5x5.zip
#   python train_grid_world_cpp.py test 5 3 --model data/model_5x5.zip
#   python train_grid_world_cpp.py run 10 12 --model data/model_10x10.zip
#

import argparse
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_schedule_fn


ENTROPY_COEF = 0.05
DEFAULT_EPISODES = 100


def print_action(action: int) -> str:
    return {
        0: "right",
        1: "up",
        2: "left",
        3: "down",
    }.get(action, "unknown")


def default_max_steps(dim: int) -> int:
    if dim <= 5:
        return 200
    if dim <= 10:
        return 400
    return 800


def register_env():
    try:
        gym.register(
            id="gymnasium_env/GridWorldCPP-v0",
            entry_point=GridWorldCPPEnv,
        )
    except Exception:
        pass


def make_env(dim: int, obstacles: int, max_steps: int, render_mode: str):
    return gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=dim,
        obs_quantity=obstacles,
        max_steps=max_steps,
        render_mode=render_mode,
    )


def local_action_mask(env):
    neighbors = env.unwrapped._neighbors
    center = neighbors.shape[0] // 2
    mask = np.array(
        [
            neighbors[center][center + 1] != 1,  # right
            neighbors[center - 1][center] != 1,  # up
            neighbors[center][center - 1] != 1,  # left
            neighbors[center + 1][center] != 1,  # down
        ],
        dtype=bool,
    )
    return mask if mask.any() else np.ones(4, dtype=bool)


def make_masked_env(dim: int, obstacles: int, max_steps: int, render_mode: str):
    return ActionMasker(
        make_env(dim, obstacles, max_steps, render_mode),
        local_action_mask,
    )


def build_recurrent_model(env, args):
    policy_kwargs = {
        "lstm_hidden_size": 64,
        "n_lstm_layers": 1,
        "shared_lstm": False,
        "enable_critic_lstm": True,
        "net_arch": {"pi": [32, 32], "vf": [32, 32]},
    }
    return RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        verbose=1,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        policy_kwargs=policy_kwargs,
        device="cpu",
    )


def build_maskable_model(env, args):
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        device="cpu",
    )


def load_model(path: str, env=None, algo: str = "maskable"):
    model_path = normalize_model_path(path)
    model_class = {
        "maskable": MaskablePPO,
        "recurrent": RecurrentPPO,
        "ppo": PPO,
    }[algo]
    return model_class.load(model_path, env=env, device="cpu")


def set_learning_rate(model, learning_rate: float):
    model.learning_rate = learning_rate
    model.lr_schedule = get_schedule_fn(learning_rate)
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = learning_rate


def normalize_model_path(path: str) -> str:
    if path.endswith(".zip"):
        return path
    return f"{path}.zip"


def configure_model_logger(model, algo, dim, obstacles, max_steps, learning_rate, ent_coef, suffix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{algo}_cpp_{dim}_{obstacles}_{max_steps}_lr{learning_rate}_ent{ent_coef}_{timestamp}{suffix}"
    log_dir = Path("log") / run_name
    model_path = Path("data") / f"{run_name}.zip"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))
    return model_path, log_dir


def build_eval_callback(algo, dim, obstacles, max_steps, log_dir, eval_freq):
    env_factory = make_masked_env if algo == "maskable" else make_env
    eval_env = env_factory(dim, obstacles, max_steps, "rgb_array")
    best_model_dir = Path(log_dir) / "best_model"
    callback_class = MaskableEvalCallback if algo == "maskable" else EvalCallback
    return callback_class(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=20,
        deterministic=algo != "maskable",
        render=False,
    )


def predict_action(model, obs, algo: str, lstm_states, episode_start, action_masks=None, deterministic=True):
    if algo == "recurrent":
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        return int(action.item()), lstm_states

    if algo == "maskable":
        action, _ = model.predict(
            obs,
            action_masks=action_masks,
            deterministic=deterministic,
        )
        return int(action.item()), lstm_states

    action, _ = model.predict(obs, deterministic=deterministic)
    return int(action.item()), lstm_states


def evaluate(model, env, episodes: int, algo: str, deterministic: bool, verbose: bool):
    full_coverage_count = 0
    total_coverages = []
    total_steps_list = []

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)

        while not done and not truncated:
            action_masks = get_action_masks(env) if algo == "maskable" else None
            action, lstm_states = predict_action(
                model,
                obs,
                algo,
                lstm_states,
                episode_start,
                action_masks=action_masks,
                deterministic=deterministic,
            )
            obs, reward, done, truncated, info = env.step(action)
            episode_start = np.array([done or truncated], dtype=bool)
            steps += 1

            if verbose:
                print(
                    f"Step: {steps}, Action: {print_action(action)}, "
                    f"Reward: {reward:.2f}, Coverage: {info['coverage']:.1%}, "
                    f"Done: {done}, Truncated: {truncated}"
                )

        total_coverages.append(info["coverage"])
        total_steps_list.append(steps)

        if done and not truncated:
            full_coverage_count += 1
            print(f"Episode {episode + 1}: Full coverage in {steps} steps.")
        else:
            print(f"Episode {episode + 1}: Coverage {info['coverage']:.1%} in {steps} steps.")

    full_coverage_rate = (full_coverage_count / episodes) * 100
    avg_coverage = np.mean(total_coverages) * 100
    coverage_std = np.std(total_coverages) * 100
    avg_steps = np.mean(total_steps_list)
    steps_std = np.std(total_steps_list)

    print("\n--- Test Finished ---")
    print(f"Full Coverage Rate: {full_coverage_rate:.2f}% ({full_coverage_count}/{episodes})")
    print(
        "Average Coverage: "
        f"{avg_coverage:.2f}% Standard Deviation: {coverage_std:.2f}% "
        f"Min Coverage: {np.min(total_coverages) * 100:.2f}% "
        f"Max Coverage: {np.max(total_coverages) * 100:.2f}%"
    )
    print(
        "Average Steps: "
        f"{avg_steps:.1f} Standard Deviation: {steps_std:.1f} "
        f"Min Steps: {np.min(total_steps_list)} Max Steps: {np.max(total_steps_list)}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test", "run", "curriculum"])
    parser.add_argument("dim", type=int)
    parser.add_argument("obstacles", type=int)
    parser.add_argument("max_steps", type=int, nargs="?")
    parser.add_argument("total_timesteps", type=int, nargs="?")
    parser.add_argument("--model", help="Model path or filename. The .zip suffix is optional.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--algo", choices=["maskable", "recurrent", "ppo"], default="maskable")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions during test/run.")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=6)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Evaluation interval in timesteps.")
    return parser.parse_args()


def main():
    args = parse_args()
    register_env()

    algo = args.algo
    max_steps = args.max_steps if args.max_steps is not None else default_max_steps(args.dim)

    if args.mode in ["train", "curriculum"] and args.total_timesteps is None:
        raise SystemExit("Training modes require total_timesteps.")

    if args.mode == "train":
        print(f"--- Starting CPP Training with {algo} ---")
        check_env(make_env(args.dim, args.obstacles, max_steps, "rgb_array"))
        env_factory = make_masked_env if algo == "maskable" else make_env
        env = make_vec_env(
            lambda: env_factory(args.dim, args.obstacles, max_steps, "rgb_array"),
            n_envs=args.n_envs,
        )

        if algo == "maskable":
            model = build_maskable_model(env, args)
        elif algo == "recurrent":
            model = build_recurrent_model(env, args)
        else:
            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                ent_coef=args.ent_coef,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                device="cpu",
            )

        model_path, log_dir = configure_model_logger(
            model,
            algo,
            args.dim,
            args.obstacles,
            max_steps,
            args.learning_rate,
            args.ent_coef,
            "",
        )

        print(f"Starting learning with {args.total_timesteps} timesteps...")
        eval_freq = max(args.eval_freq // args.n_envs, 1)
        callback = build_eval_callback(algo, args.dim, args.obstacles, max_steps, log_dir, eval_freq)
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
        model.save(model_path)
        print(f"Model trained and saved to {model_path}")
        print(f"Logs saved to {log_dir}")
        print(f"Best model saved under {Path(log_dir) / 'best_model'}")

    elif args.mode == "curriculum":
        if not args.model:
            raise SystemExit("Curriculum mode requires --model with the pretrained model path.")

        print("--- Starting CPP Curriculum Learning Training ---")
        env_factory = make_masked_env if algo == "maskable" else make_env
        env = make_vec_env(
            lambda: env_factory(args.dim, args.obstacles, max_steps, "rgb_array"),
            n_envs=args.n_envs,
        )
        model = load_model(args.model, env=env, algo=algo)
        set_learning_rate(model, args.learning_rate)
        model_path, log_dir = configure_model_logger(
            model,
            algo,
            args.dim,
            args.obstacles,
            max_steps,
            args.learning_rate,
            args.ent_coef,
            "_curriculum",
        )

        print(f"Continuing learning with {args.total_timesteps} timesteps...")
        eval_freq = max(args.eval_freq // args.n_envs, 1)
        callback = build_eval_callback(algo, args.dim, args.obstacles, max_steps, log_dir, eval_freq)
        model.learn(
            total_timesteps=args.total_timesteps,
            reset_num_timesteps=False,
            callback=callback,
        )
        model.save(model_path)
        print(f"Model trained and saved to {model_path}")
        print(f"Logs saved to {log_dir}")
        print(f"Best model saved under {Path(log_dir) / 'best_model'}")

    elif args.mode in ["test", "run"]:
        if not args.model:
            raise SystemExit("Test/run mode requires --model with the trained model path.")

        env_factory = make_masked_env if algo == "maskable" else make_env
        env = env_factory(
            args.dim,
            args.obstacles,
            max_steps,
            "human" if args.mode == "run" else "rgb_array",
        )
        model = load_model(args.model, algo=algo)
        evaluate(
            model,
            env,
            episodes=1 if args.mode == "run" else args.episodes,
            algo=algo,
            deterministic=not args.stochastic,
            verbose=args.mode == "run",
        )


if __name__ == "__main__":
    main()
