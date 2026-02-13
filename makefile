SHELL = /bin/sh

CC = mpicc
CFLAG = -Wall -g

all: mandelbrot

mandelbrot:
	${CC} ${CFLAG} mandelbrot.c -o mandelbrot

clean:
	rm mandelbrot
