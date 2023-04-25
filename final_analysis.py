import os
import json
import wandb
import subprocess
import re
from tqdm import tqdm

from functools import lru_cache

# Set up output directories and paths
eval_dir = "eval_dir"
configs_dir = os.path.join(eval_dir, "configs")
results_dir = os.path.join(eval_dir, "results")
analysis_dir = os.path.join(eval_dir, "analysis")
os.makedirs(configs_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(analysis_dir, exist_ok=True)

# Get the run from wandb
@lru_cache
def get_run(run_id):
    api = wandb.Api()
    run = api.run(run_id)
    return run

# Get config from run
def get_run_config(run_id):
    run = get_run(run_id)
    return run.config

# Get name of run
def get_run_name(run_id):
    run = get_run(run_id)
    return run.name

# Get tags of run
def get_run_tags(run_id):
    run = get_run(run_id)
    return run.tags

def get_run_gpu_ids(run_id):
    run = get_run(run_id)
    return run.config.get('gpu_ids', [1])

def get_run_runtime_hours(run_id):
    run = get_run(run_id)
    return run.summary['_wandb']['runtime']/3600

# Function to download the checkpoint using W&B API
def download_checkpoint(run_path, artifact_name, checkpoint_path):
    if os.path.exists(checkpoint_path):
        return

    api = wandb.Api()
    run = get_run(run_path)
    artifact = api.artifact(f'{run_path.rsplit("/", 1)[0]}/{artifact_name}')
    artifact_dir = artifact.download()
    checkpoint_file = [f'{artifact_dir}/{f}' for f in os.listdir(artifact_dir) if f.endswith('.pth')][0] # Replace with the correct file name in the artifact

    # Move the downloaded checkpoint to the desired path
    os.rename(checkpoint_file, checkpoint_path)


# Function to recursively convert JSON values to Python values
def json_to_python(obj):
    if isinstance(obj, dict):
        # special case for 'img_scale' key
        if 'img_scale' in obj:
            img_scale = obj['img_scale']
            obj['img_scale'] = [tuple(img_scale),]  # Convert to list[tuple[int, int]] format
        # special case for renamed models
        if obj.get('type') == 'VAN':
            obj['type'] = 'VAN_Official'
        for k, v in obj.items():
            if isinstance(v, str) and v.startswith('/content/'):
                obj[k] = obj[k].replace('/content/', '')
                obj[k] = obj[k].replace('test2017', 'val2017')
        return {k: json_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_to_python(elem) for elem in obj]
    else:
        return obj

# Function to write the config file
def write_config_file(config_dict, config_path):
    config_dict = json_to_python(config_dict)

    with open(config_path, "w") as f:
        for key, value in config_dict.items():
            if isinstance(value, str):
                f.write(f"{key} = '{value}'\n")
            else:
                f.write(f"{key} = {value}\n")

# Function to run the test script and store the results
def run_test_script(config_path, checkpoint_path, result_path, eval=False):
    print('Running test.py')
    if eval:
        out = [
            "--out",
            result_path + '.pkl',
        ]
    else:
        out = [
            f"--json-prefix={result_path}",
        ]
    with open(result_path + '_test_stdout.txt', 'w') as f:
        subprocess.run([
            "python",
            "tools/test.py",
            config_path,
            checkpoint_path,
            *out,
        ], stdout=f)

# Function to test for robustness
def run_robustness_test(config_path, checkpoint_path, result_path, img_dir):
    print('Running test_robustness.py')
    with open(result_path + '_robustness.txt', 'w') as f:
        subprocess.run([
            "python",
            "tools/analysis_tools/test_robustness.py",
            config_path,
            checkpoint_path,
            "--out",
            result_path + "_test_robustness.pkl",
            "--eval",
            "bbox",
            "--show-dir",
            img_dir,
        ], stdout=f)


# Function to run further analysis and store the results
def run_analysis(config_path, result_path, analysis_output_dir):
    os.makedirs(analysis_output_dir, exist_ok=True)
    tool_path = 'tools/analysis_tools'
    
    # Get FLOPs
    print('Calculating Complexity')
    with open(os.path.join(analysis_output_dir, "get_flops.txt"), "w") as f:
        subprocess.run(["python", f"{tool_path}/get_flops.py", config_path, '--shape', '1280,800'], stdout=f)

    # COCO error analysis
    print('Analyzing Errors')
    with open(os.path.join(analysis_output_dir, "confusion_matrix.txt"), "w") as f:
        subprocess.run(["python", f"{tool_path}/confusion_matrix.py", config_path, result_path + '.pkl', analysis_output_dir], stdout=f)

    # Robustness
    # print('Evaluating robustness')
    # with open(os.path.join(analysis_output_dir, "robustness_eval.txt"), "w") as f:
        # subprocess.run(["python", f"{tool_path}/robustness_eval.py", result_path + '_test_robustness.pkl'], stdout=f)

    # Benchmark
    print('Benchmarking')
    os.environ['LOCAL_RANK'] = "0"
    with open(os.path.join(analysis_output_dir, "benchmark.txt"), "w") as f:
        print(' '.join(["python", "-m", "torch.distributed.launch", "--nproc_per_node=1", "--master_port=29500",
                        f"{tool_path}/benchmark.py", config_path, checkpoint_path, '--size', '1280,800']))
        subprocess.run(["python", "-m", "torch.distributed.launch", "--nproc_per_node=1", "--master_port=29500",
                        f"{tool_path}/benchmark.py", config_path, checkpoint_path, '--size', '1280,800', '--launcher', 'pytorch'], stdout=f)

def analyze_and_summarize(run_name, analysis_dir):
    summary = {'run_name': run_name}
    
    # Read benchmark.txt
    with open(os.path.join(analysis_dir, run_name, "benchmark.txt"), "r") as f:
        content = f.read()
        overall_fps = re.search(r'Overall fps: (.+?) img', content)
        if overall_fps:
            summary['overall_fps'] = float(overall_fps.group(1))
    
    # Read get_flops.txt
    with open(os.path.join(analysis_dir, run_name, "get_flops.txt"), "r") as f:
        content = f.read()
        # Extract totals
        flops = re.search(r'Flops: (.+?) GFLOPs', content)
        params = re.search(r'Params: (.+?) M', content)
        if flops:
            summary['total_flops'] = float(flops.group(1))
        if params:
            summary['total_params'] = float(params.group(1))
            
        # Extract by part
        flops_pattern = re.compile(r'\s\((backbone|neck|head])\):\s(\w+)\(\s+(\d+\.\d+) M,\s+(\d+\.\d+)% Params,\s+(\d+\.\d+) GFLOPs,\s+(\d+\.\d+)% FLOPs,')
        flops_data = flops_pattern.findall(content)

        flops_dict = {}
        for part, part_type, params, params_percentage, flops, flops_percentage in flops_data:
            flops_dict[part] = {
                "type": part_type,
                "flops": float(flops),
                "params": float(params),
                "flops_percentage": float(flops_percentage),
                "params_percentage": float(params_percentage),
            }

        summary['complexity'] = flops_dict
    
    # Read the results file
    result_file = os.path.join("eval_dir", "results", f"{run_name}_test_stdout.txt")
    with open(result_file, "r") as f:
        content = f.read()
        bbox_mAP = re.search(r'\(\'bbox_mAP\', (.+?)\)', content)
        if bbox_mAP:
            summary['bbox_mAP'] = float(bbox_mAP.group(1))
    
    return summary


# Iterate through the list of run ids
run_list = [
        "gxcrib1t",
        ]

# First, load all checkpoints
print('Preparing checkpoints and configs')
for run_id in tqdm(run_list):
    run_path = f'nkoch-aitastic/mmyolo/{run_id}'
    run_name = get_run_name(run_path)
    config = get_run_config(run_path)
    checkpoint_path = os.path.join(configs_dir, f"{run_name}.pth")
    config_path = os.path.join(configs_dir, f"{run_name}.py")

    # Download checkpoint
    try:
        download_checkpoint(run_path, f'run_{run_id}_model:latest', checkpoint_path)
        # Write config file
        write_config_file(config, config_path)
    except:
        checkpoint_path = 'work_dirs/yolov8_n_syncbn_fast_8xb16-500e_coco/epoch_500.pth'
        config_path = 'work_dirs/yolov8_n_syncbn_fast_8xb16-500e_coco/yolov8_n_syncbn_fast_8xb16-500e_coco.py'

all_summaries = {}
# Then evaluate
print('Running evaluation')
for run_id in tqdm(run_list):
    run_path = f'nkoch-aitastic/mmyolo/{run_id}'
    run_name = get_run_name(run_path)
    config = get_run_config(run_path)

    # config_path = os.path.join(configs_dir, f"{run_name}.py")
    # checkpoint_path = os.path.join(configs_dir, f"{run_name}.pth")
    result_path = os.path.join(results_dir, f"{run_name}")
    analysis_output_dir = os.path.join(analysis_dir, run_name)
    robustness_dir = os.path.join(analysis_output_dir, 'robustness')
    checkpoint_path = 'work_dirs/yolov8_n_syncbn_fast_8xb16-500e_coco/epoch_500.pth'
    config_path = 'work_dirs/yolov8_n_syncbn_fast_8xb16-500e_coco/yolov8_n_syncbn_fast_8xb16-500e_coco.py'

    # Run the test script and store the results
    # if not os.path.exists(result_path):
        # only run test script if we don't have results already, it's expensive
    run_test_script(config_path, checkpoint_path, result_path)
    run_test_script(config_path, checkpoint_path, result_path, eval=True)

    # Test robustness, results are analysed later
    # takes a long time, do whenever
    # run_robustness_test(config_path, checkpoint_path, result_path, robustness_dir)

    # Run further analysis and store the results
    run_analysis(config_path, result_path, analysis_output_dir)

    # Analyze and summarize
    summary = analyze_and_summarize(run_name, "eval_dir/analysis")
    all_summaries[run_name] = {
            'run_id': run_id,
            'summary': summary,
            'tags': get_run_tags(run_path),
            'runtime_hours': get_run_runtime_hours(run_path),
            'gpus': get_run_gpu_ids(run_path),
            'train_gpu_hours': get_run_runtime_hours(run_path) * len(get_run_gpu_ids(run_path)),
            }

# Print or store the summaries
with open(results_dir + '/summary.json', 'w') as f:
    json.dump(all_summaries, f)

