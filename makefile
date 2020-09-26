EXECS=matrix
MPICC?=mpicc

all: ${EXECS}

mpi_performance_test: matrix.c
	${MPICC} -o matrix matrix.c -lm

clean:
	rm -f ${EXECS}
