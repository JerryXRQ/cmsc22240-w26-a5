SHELL = /bin/sh

CC = mpicc
CFLAG = -Wall -g 

all: newton

newton:
	${CC} ${CFLAG} newton.c -o newton -lm

clean:
	rm newton