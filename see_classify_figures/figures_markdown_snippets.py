from IPython.display import display, Markdown

data_creation_instructions = """
To generate data, run:
- for sklearn
```bash
sbatch generate_sklearn_data.sb -n 100 -p 100 -t 100
```
- for dhahri
```bash
sbatch generate_dhahri_data.sb -n 100 -p 100 -t 100
```
These commands generate files for running genetic search using 100 generations 
and population size of 100 for 100 trials for the sklearn and the
Breast Cancer Wisconsin (Diagnostic) Dataset respectively.

The `-n`, `-p`, `-t` flags control number of generations, population sizes, 
and number of trials respectively.

The data generated tracks the top 10 individuals and the population by
each generation and is stored in the corresponding output 
(i.e. the slurm_\[id\].out file).

To extract this data to a csv file, run:
- for the top 10 individuals:
```bash
grep "# GEN HOF_index" slurm_[id].out | cut -d '|' -f2 > "filename_1.csv"
```
- for the population:
```bash
grep "# GEN population_index" slurm_[id].out | cut -d '|' -f2 > "filename_2.csv"
```
    
If the jobs are taking too long to finish, one can batch multiple jobs
and reduce the number of trials. To batch multiple jobs simultaneously
modify the array range on line two of the job scripts. For example,
to generate 100 trials only could submit 25 jobs, each of which would
run 4 jobs. To submit 25 jobs simultaneously, modify line two such that:

- before:
```bash
#SBATCH --array=0
```
- after:
```bash
#SBATCH --array=0-24
```

Then, to run each jobs with 4 trials, enter the following into
the command line:
```bash
sbatch generate_dhahri_data.sb -n 100 -p 100 -t 4
```

In this case, one would have multiple output files (`.out`) that contain different trials 
of the same GA run. One could try the moving all the relevant files into 
one directory (`mv`), changing to that directory (`cd`), and running the following commands instead:
- for the top 10 individuals:
```bash
grep "# GEN HOF_index" *.out | cut -d '|' -f2 > "filename_1.csv"
```
- for the population:
```bash
grep "# GEN population_index" *.out | cut -d '|' -f2 > "filename_2.csv"
```
"""

def show_data_instructions():
    display(Markdown(data_creation_instructions))