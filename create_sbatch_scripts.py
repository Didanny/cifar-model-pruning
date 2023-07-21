lines = [
    '#!/bin/sh \n' +
    '#SBATCH -n 1 \n' +              # Number of tasks to run (equal to 1 cpu/core per task)
    '#SBATCH -N 1 \n' +              # Ensure that all cores are on one machine
    '#SBATCH -p openlab.p \n' +      # Partition to submit to
    '#SBATCH --mem-per-cpu=8192 \n' +
    '#SBATCH -o log_',               # File to which STDOUT will be written, %j inserts jobid
    '#SBATCH -e err_',
]

model_list = [
    # 'cifar100_vgg11_bn',
    # 'cifar100_vgg13_bn',
    # 'cifar100_vgg16_bn',
    # 'cifar100_vgg19_bn',
    # 'cifar100_resnet20',
    # 'cifar100_resnet32',
    # 'cifar100_resnet44',
    # 'cifar100_resnet56',
    # 'cifar100_mobilenetv2_x0_5',
    'cifar100_mobilenetv2_x0_75',
    'cifar100_mobilenetv2_x1_0',
    'cifar100_mobilenetv2_x1_4',
]

mode_list = ['1', '2', 'mean', 'inf']

import os

if __name__ == '__main__':
    for model in model_list:
        for mode in mode_list:
            with open(f'{model}_{mode}.sh', 'w') as f:
                for line in lines:
                    f.write(f'{line}_{model}_{mode}.out \n')
                f.write(f'\npython pruning.py --model {model} --mode {mode}')
                
    # Run the scripts
    os.system('conda activate torch2')
    for model in model_list:
        for mode in mode_list:
            os.system(f'sbatch {model}_{mode}.sh')