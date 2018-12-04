#
# Makefile
#
# lattice_gas
#
# Author: Bruno Beltran <brunobeltran0@gmail.com>
#

# # replaced rm **.o with more portable find commands
# # and regex search =~ in distclean with manual check for Y/y
SHELL:=/bin/bash
# .SHELLFLAGS="-O extglob -c"


CC=gcc
PEDANTICFLAGS = -std=gnu99 -ggdb -Jsrc ${INCLUDE_DIRS} -Wall -pedantic -Warray-bounds -Wattributes -Wextra
CFLAGS = ${PEDANTICFLAGS}

# include dirs
INCLUDE_DIRS = -I. -I./SFMT-src-1.5.1

# link flags
LDFLAGS = -lm

SRC := lattice_gas.c # ./SFMT-src-1.5.1/SFMT.c
OBJ := $(addsuffix .o,$(basename $(SRC)))

# program name
PROGRAM = mcgas

# by default, compile only
all: $(PROGRAM) Makefile

# a target to just run the main program
run: $(PROGRAM)
	./$(PROGRAM)

# %.o: %.c
# 	$(CC) -c $(CFLAGS) -o $@ $^ $(INCLUDE_DIRS)

$(PROGRAM): $(SRC)
	$(CC) $(CFLAGS) -o $(PROGRAM) $^ $(INCLUDE_DIRS) $(LDFLAGS)

clean:
	rm -f $(OBJ) $(PROGRAM) a.out core
