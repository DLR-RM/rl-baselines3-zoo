import argparse
import os
import re
import shutil
import subprocess
from copy import deepcopy

from utils.utils import ALGOS, get_latest_run_id

if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("-o", "--output-folder", help="Output folder", type=str)
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("-g", "--gif", action="store_true", default=False, help="Convert final video to gif")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    args = parser.parse_args()

    env_id = args.env
    algo = args.algo
    folder = args.folder
    n_timesteps = args.n_timesteps
    n_envs = args.n_envs
    video_folder = args.output_folder
    seed = args.seed
    deterministic = args.deterministic
    convert_to_gif = args.gif

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print(f"Loading latest experiment, id={args.exp_id}")
    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    if video_folder is None:
        video_folder = os.path.abspath(os.path.join(log_path, "videos"))
    shutil.rmtree(video_folder, ignore_errors=True)
    os.makedirs(video_folder, exist_ok=True)

    # record a video of every model
    models_dir_entries = [dir_ent.name for dir_ent in os.scandir(log_path) if dir_ent.is_file()]
    checkpoints = list(filter(lambda x: x.startswith("rl_model_"), models_dir_entries))
    checkpoints = list(map(lambda x: int(re.findall(r"\d+", x)[0]), checkpoints))
    checkpoints.sort()

    args_final_model = [
        "--env",
        env_id,
        "--algo",
        algo,
        "--exp-id",
        str(args.exp_id),
        "-f",
        folder,
        "-o",
        video_folder,
        "--n-timesteps",
        str(n_timesteps),
        "--n-envs",
        str(n_envs),
        "--seed",
        str(seed),
        # Disable rendering to generate videos faster
        "--no-render",
    ]
    if deterministic is not None:
        args_final_model.append("--deterministic")

    if os.path.exists(os.path.join(log_path, f"{env_id}.zip")):
        return_code = subprocess.call(["python", "-m", "utils.record_video"] + args_final_model)
        assert return_code == 0, "Failed to record the final model"

    if os.path.exists(os.path.join(log_path, "best_model.zip")):
        args_best_model = args_final_model + ["--load-best"]
        return_code = subprocess.call(["python", "-m", "utils.record_video"] + args_best_model)
        assert return_code == 0, "Failed to record the best model"

    args_checkpoint = args_final_model + ["--load-checkpoint"]
    args_checkpoint.append("0")
    for checkpoint in checkpoints:
        args_checkpoint[-1] = str(checkpoint)
        return_code = subprocess.call(["python", "-m", "utils.record_video"] + args_checkpoint)
        assert return_code == 0, f"Failed to record the {checkpoint} checkpoint model"

    # add text to each video
    episode_videos_names = [dir_ent.name for dir_ent in os.scandir(video_folder) if dir_ent.name.endswith(".mp4")]

    checkpoints_videos_names = list(filter(lambda x: x.startswith("checkpoint"), episode_videos_names))

    # sort checkpoints by the number of steps
    def get_number_from_checkpoint_filename(filename: str) -> int:
        match = re.search("checkpoint-(.*?)-", filename)
        number = 0
        if match is not None:
            number = match.group(1)

        return int(number)

    if checkpoints_videos_names is not None:
        checkpoints_videos_names.sort(key=get_number_from_checkpoint_filename)

    final_model_video_name = list(filter(lambda x: x.startswith("final-model"), episode_videos_names))
    best_model_video_name = list(filter(lambda x: x.startswith("best-model"), episode_videos_names))
    episode_videos_names = checkpoints_videos_names + final_model_video_name + best_model_video_name
    episode_videos_path = [os.path.join(video_folder, video) for video in episode_videos_names]

    # the text displayed will be the first two words of the file
    def get_text_from_video_filename(filename: str) -> str:
        match = re.search(r"^(\w+)-(\w+)", filename)
        text = ""
        if match is not None:
            text = f"{match.group(1)} {match.group(2)}"

        return text

    episode_videos_names = list(map(get_text_from_video_filename, episode_videos_names))

    # In some cases, ffmpeg needs a tmp file
    # https://stackoverflow.com/questions/28877049/issue-with-overwriting-file-while-using-ffmpeg-for-converting
    tmp_videos_path = deepcopy(episode_videos_path)
    tmp_videos_path = [path_[:-4] + "_with_text" + ".mp4" for path_ in tmp_videos_path]

    for i in range(len(episode_videos_path)):
        ffmpeg_command_to_add_text = (
            f'ffmpeg -i {episode_videos_path[i]} -vf drawtext="'
            f"text='{episode_videos_names[i]}': fontcolor=white: fontsize=24: box=1: boxcolor=black@0.5:"
            f'boxborderw=5: x=(w-text_w)/2: y=12" -codec:a copy {tmp_videos_path[i]} -y -hide_banner -loglevel error'
        )
        os.system(ffmpeg_command_to_add_text)

    # join videos together and convert to gif if needed
    ffmpeg_text_file = os.path.join(video_folder, "tmp.txt")
    with open(ffmpeg_text_file, "a") as file:
        for video_path in tmp_videos_path:
            file.write(f"file {video_path}\n")

    final_video_path = os.path.abspath(os.path.join(video_folder, "training.mp4"))
    os.system(f"ffmpeg -f concat -safe 0 -i {ffmpeg_text_file} -c copy {final_video_path} -hide_banner -loglevel error")
    os.remove(ffmpeg_text_file)
    print(f"Saving video to {final_video_path}")

    if convert_to_gif:
        final_gif_path = os.path.abspath(os.path.join(video_folder, "training.gif"))
        os.system(f"ffmpeg -i {final_video_path} -vf fps=10 {final_gif_path} -hide_banner -loglevel error")
        print(f"Saving gif to {final_gif_path}")

    # Remove tmp video files
    for video_path in tmp_videos_path:
        os.remove(video_path)
