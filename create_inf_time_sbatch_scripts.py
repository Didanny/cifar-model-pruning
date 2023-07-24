lines = [
    '#!/bin/sh \n' +
    '#SBATCH -n 1 \n' +              # Number of tasks to run (equal to 1 cpu/core per task)
    '#SBATCH -N 1 \n' +              # Ensure that all cores are on one machine
    '#SBATCH -p openlab.p \n' +      # Partition to submit to
    # '#SBATCH --mem-per-cpu=8192 \n' +
    '#SBATCH -o log_',               # File to which STDOUT will be written, %j inserts jobid
    '#SBATCH -e err_',
]

arg_list = [
    ('cifar100_vgg11_bn', 'inf', 0.175),
    ('cifar100_vgg13_bn', 'inf', 0.21),
    ('cifar100_vgg16_bn', 'inf', 0.255),
    ('cifar100_vgg19_bn', 'inf', 0.24),
    # 'cifar100_resnet20',
    # 'cifar100_resnet32',
    # 'cifar100_resnet44',
    # 'cifar100_resnet56',
    # 'cifar100_mobilenetv2_x0_5',
    # 'cifar100_mobilenetv2_x0_75',
    # 'cifar100_mobilenetv2_x1_0',
    # 'cifar100_mobilenetv2_x1_4',
]

mode_list = ['inf']

import os

if __name__ == '__main__':
    for arg in arg_list:
        with open(f'inftimes_{arg[0]}_{arg[1]}_{arg[2]}.sh', 'w') as f:
            for line in lines:
                f.write(f'{line}inftimes_{arg[0]}_{arg[1]}_{arg[2]}.out \n')
            f.write(f'\npython comp_inf_time.py --model {arg[0]} --mode {arg[1]} --thresh {arg[2]}')
            
    # Run the scripts
    os.system('conda activate torch2')
    for arg in arg_list:
        for mode in mode_list:
            os.system(f'sbatch inftimes_{arg[0]}_{arg[1]}_{arg[2]}.sh')