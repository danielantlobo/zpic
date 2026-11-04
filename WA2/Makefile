CC = gcc
NVCC = nvcc
CUDA_HOME ?= $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_INC = -I$(CUDA_HOME)/include
CUDA_LIB = -L$(CUDA_HOME)/lib64 -lcudart

CFLAGS = -Ofast -g -std=c99 -Wall -fopenmp -march=native -Isrc $(CUDA_INC) -D_XOPEN_SOURCE=600 -Wno-pedantic

# these are for search k20 GPU
#NCCFLAGS = -Wno-deprecated-gpu-targets -arch=sm_35 -O2 -g -std=c++11

NVCCFLAGS = -O3 -arch=sm_80 -Isrc

# Optional: particles per cell for the twostream deck (default 500), e.g. make PPC=4000
ifdef PPC
CFLAGS += -DZPIC_PPC=$(PPC)
endif
LDFLAGS = -lm -fopenmp -Wl,--allow-multiple-definition $(CUDA_LIB)
SRCDIR = src
INPUTDIR = src/input
INPUT_SOURCE = $(INPUTDIR)/twostream.c
C_SOURCES = $(SRCDIR)/current.c \
	        $(SRCDIR)/emf.c \
	        $(SRCDIR)/random.c \
	        $(SRCDIR)/timer.c \
	        $(SRCDIR)/main.c \
	        $(SRCDIR)/simulation.c \
	        $(SRCDIR)/zdf.c

CU_SOURCES = $(SRCDIR)/particles.cu

C_OBJS = $(C_SOURCES:.c=.o)
CU_OBJS = $(CU_SOURCES:.cu=.o)
INPUT_OBJ = $(INPUT_SOURCE:.c=.o)

TARGET = zpic
all: $(TARGET)
$(TARGET): $(C_OBJS) $(CU_OBJS) $(INPUT_OBJ)
	$(CC) $(C_OBJS) $(CU_OBJS) $(INPUT_OBJ) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET) $(SRCDIR)/*.o $(INPUTDIR)/*.o

run: $(TARGET)
	 sbatch run_zpic.sh
