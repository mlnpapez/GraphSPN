import os
import time
import json
import torch
import utils
import subprocess
import gridsearch_hyperpars
import graphspn
import datasets

CHECKPOINT_DIR = 'results/gridsearch/model_checkpoint/'
TRAINEPOCH_DIR = 'results/gridsearch/model_trainepoch/'
EVALUATION_DIR = 'results/gridsearch/model_evaluation/'
OUTPUTLOGS_DIR = 'results/gridsearch/model_outputlogs/'


def unsupervised(dataset, name, par_buffer):
    torch.set_float32_matmul_precision('medium')

    hyperpars = par_buffer[int(os.environ["SLURM_ARRAY_TASK_ID"])]

    if 'deq' in name:
        loader_trn, loader_val, loader_tst = datasets.load_dataset(dataset, hyperpars['batch_size'], ohe=True)
    else:
        loader_trn, loader_val, loader_tst = datasets.load_dataset(dataset, hyperpars['batch_size'], ohe=False)

    x_trn, _, _ = datasets.load_dataset(dataset, 0, raw=True)
    smiles_trn = [x['s'] for x in x_trn]

    model = graphspn.MODELS[name](**hyperpars['model_hyperpars'])

    print(dataset)
    print(json.dumps(hyperpars, indent=4))
    print(model)
    print(f'The number of parameters is {utils.count_parameters(model)}.')

    path = utils.train(model, loader_trn, loader_val, smiles_trn, hyperpars, CHECKPOINT_DIR, TRAINEPOCH_DIR, verbose=True)
    model = torch.load(path)
    metric = utils.evaluate(model, loader_trn, loader_val, loader_tst, smiles_trn, hyperpars, EVALUATION_DIR)

    print("\n".join(f'{key:<20}{value:>10.4f}' for key, value in metric.items()))


def submit_job(dataset, model, par_buffer, device, max_sub):
    outputlogs_dir = OUTPUTLOGS_DIR + f'{dataset}/'
    par_buffer_str = str(par_buffer).replace("'", '"')
    cmd_python = "from gridsearch import unsupervised\n" + f'unsupervised("{dataset}", "{model}", {par_buffer_str})'
    cmd_sbatch = "source activate graphspn\n" + f"python -c '{cmd_python}'"

    while True:
        run_squeue = subprocess.run(['squeue', f'--user={os.environ["USER"]}', '-h', '-r'], stdout=subprocess.PIPE)
        run_wcount = subprocess.run(['wc', '-l'], input=run_squeue.stdout, capture_output=True)
        num_queued = int(run_wcount.stdout)

        if len(par_buffer) <= max_sub - num_queued:
            if device == 'cuda':
                subprocess.run(['sbatch',
                                f'--job-name={model.replace("graphspn_naive","")}',
                                f'--output={outputlogs_dir}/{model}/%A_%a.out',
                                '--partition=amdgpu',
                                '--ntasks=1',
                                f'--gres=gpu:1',
                                f'--array=0-{len(par_buffer)-1}',
                                f'--wrap={cmd_sbatch}'])
            elif device == 'cpu':
                subprocess.run(['sbatch',
                                f'--job-name={model.replace("graphspn_naive","")}',
                                f'--output={outputlogs_dir}/{model}/%A_%a.out',
                                '--partition=amd',
                                '--ntasks=1',
                                '--ntasks-per-node=1',
                                '--cpus-per-task=1',
                                '--mem-per-cpu=64000',
                                f'--array=0-{len(par_buffer)-1}',
                                f'--wrap={cmd_sbatch}'])
            else:
                os.error('Unknown device.')

            break
        else:
            time.sleep(20)


if __name__ == "__main__":
    max_jobs_to_submit = 25
    par_buffer = []
    gpu_models = []

    run_maxjob = subprocess.run(['sacctmgr', 'show', 'qos', 'format=MaxJobsPU', 'where', 'collaborator', '-n'], capture_output=True)
    max_job = int(run_maxjob.stdout)
    max_sub = 500

    if max_jobs_to_submit < max_job:
        for dataset, attributes in datasets.MOLECULAR_DATASETS.items():
            print(dataset)
            for model in graphspn.MODELS.keys():
                print(model)
                if model in gpu_models:
                    device = 'cuda'
                else:
                    device = 'cpu'

                for hyperpars in gridsearch_hyperpars.GRIDS[model](attributes):
                    hyperpars['model_hyperpars']['device'] = device

                    path = EVALUATION_DIR + f'{dataset}/{model}/' + utils.dict2str(utils.flatten_dict(hyperpars)) + '.csv'
                    if not os.path.isfile(path):
                        par_buffer.append(hyperpars)

                    if len(par_buffer) == max_jobs_to_submit:
                        submit_job(dataset, model, par_buffer, device, max_sub)
                        par_buffer = []

                if len(par_buffer) > 1:
                    submit_job(dataset, model, par_buffer, device, max_sub)
                    par_buffer = []
    else:
        os.error('Too many jobs to submit in a single batch.')
