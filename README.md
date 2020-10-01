Lab Assignment #5
Amjad Aryan and Everett Tsang
-----------------------
This purpose of this lab is to utilize the MPI library to efficiently multiply a 14k x 14k matrix using multiple nodes to distribute the computations.

Compilation Instructions
1. Before compiling, execute the following commands
    module purge
    module load gcc/8.3.0
    module load openmpi/4.0.2
    module load pmix
2. To compile, execute the command: make
3. To clean the directory, execute the command: make clean

Run Instructions
1. Execute the command: sbatch run.job
2. Observe the output at the "slurm-{job_id}.out" file.

Changing Number of Nodes Allocated (Optional)
1. In the file "run.job" change the following lines
    Line 2:  #SBATCH --ntasks = {New Number of Nodes}
    Line 10: mpirun -n {New Number of Nodes} ./matrix
