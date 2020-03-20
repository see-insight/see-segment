import sys
import os
from shutil import copyfile
import subprocess

def check_dir(path):
		directory = os.path.dirname(path)
		if not os.path.exists(path):
			return False
		return True

def check_and_create(path):
		if (check_dir(path) == False):
			os.makedirs(path)

if __name__=='__main__':
#Need to error check

	#What should we call the file?
	fileName = input("sb file name: ")
	fileName = fileName + ".sb"
	#How much time should the script run?
	runTime = input("Input run time (HH:MM:SS): ")
	#Number of CPUS
	CPUs = input("Number of CPUs: ")
	#Memory per CPU
	memory = input("Memory per CPU (GB): ")
	#Job name
	jobName = input("Name of job: ")


	#Let's write
	f = open(fileName, "a+")
	f.write("#!/bin/bash --login\n")

	f.write("#SBATCH --time="+ str(runTime) + "\n")
	f.write("#SBATCH -n "+ str(CPUs) +"\n")
	f.write("#SBATCH -c 1\n")
	f.write("#SBATCH --mem-per-cpu=" +str(memory) +"G\n")
	f.write("SBATCH --job-name " + str(jobName) + "\n")

	f.write("env\n")
	f.write("conda activate opencv35\n")

	f.write("export LD_LIBRARY_PATH=/mnt/home/f0008668/lib/libsdp.so:$LD_LIBRARY_PATH\n")
	f.write("cd Binary-Image-Segmentation/\n")
	f.write("HOSTFILE=$SLURM_JOB_ID.hostfile\n")
	f.write("srun hostname -s > $HOSTFILE\n")
	f.write("srun python -m scoop --hostfile $HOSTFILE -n "+ str(CPUs) + " main.py\n")

	f.write("module load powertools\n")
	f.write("js $SLURM_JOB_ID\n")

	f.close()

	#Now let's edit the script
	#Mutation chance
	mutProb = input("Specify the mutation chance (e.g. 0.75): ")
	#flipping chance (Each value of mutated value)
	flipProb = input ("Specify the flipping chance (e.g. 0.76): ")
	#Crossover  chance
	crProb = input("Specify the crossover chance: ")
	#Generations
	gens = input("Total number of generations: ")
	#Population
	pop = input("Total population: ")
	name = input("Name of new python script: ")

	#copyfile(src, dst)

	
	foundPop = False
	foundGen = False
	foundMut = False
	foundFlip = False
	foundCross = False
	

	prog = open("main.py", "r")
	

	popFolder = "POP="+str(pop)
	genFolder = "GEN="+str(gens)
	endFileName = "MUT="+str(mutProb)+"_FLIP="+str(flipProb)+"_CROSS="+str(crProb)+".txt"
	filePath = popFolder + "/" + genFolder
	check_and_create(filePath)
	exeFile = filePath + "/" + name
	dest = open(exeFile, "w+")
	for line in prog:
		#We have a few keywords to look for
		if "POPULATION" in line and foundPop == False:
			popString = "POPULATION = "+ pop + "\n"
			dest.write(popString)
			foundPop = True
			continue
		elif "GENERATIONS" in line and foundGen == False:
			genString = "GENERATIONS = "+gens+ "\n"
			dest.write(genString)
			foundGen = True
			continue
		elif "MUTATION" in line and foundMut == False:
			mutString = "MUTATION = "+mutProb + "\n"
			dest.write(mutString)
			foundMut = True
			continue
		elif "FLIPPROB" in line and foundFlip == False:
			flipString = "FLIPPROB = "+flipProb + "\n"
			dest.write(flipString)
			foundFlip = True
			continue
		elif "CROSSOVER" in line and foundCross == False:
			crosString = "CROSSOVER = "+crProb + "\n"
			dest.write(crosString)
			foundCross = True 
			continue
		else:
			dest.write(line)
	prog.close()

	dest.close()
	chdir = "cd "+filePath
	runString = "python "+name +">> "+filePath + "/" + endFileName
	subprocess.call(chdir)
	subprocess.call(runString)

	

	#Sort by population first
	




