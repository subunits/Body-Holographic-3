    CC=gcc
    CFLAGS=-O2 -Wall
    LIBS=-lfftw3 -lpng -lm

    all: holo_full

    holo_full: holo_full.c
	$(CC) $(CFLAGS) holo_full.c -o holo_full $(LIBS)

    clean:
	rm -f holo_full
