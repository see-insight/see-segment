import os

if __name__ == '__main__':
	print ("ID before forking: {}".format(os.getpid()))

	try:
		pid = os.fork()
	except OSError:
		exit("Child Process could not be created")
	if pid == 0:
		print("Child has PID: {}".format(os.getpid()))
		exit()
	print("Parent afer fork: {}".format(pid))
	finished = os.waitpid(0,0)
	print(finished)