EXECS=matrix
MPICC?=mpicc

all: ${EXECS}

matrix: matrix.c
	${MPICC} -o matrix matrix.c -lm -O2

clean:
	rm -f ${EXECS}
	rm slurm-*

run:
	sbatch run.job

queue:
	squeue -u everettt
