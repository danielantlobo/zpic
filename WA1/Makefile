# GCC options optimized for A64FX
CC = gcc

# Optimization Flags:
# -Ofast: Max optimization (fast math, vectorization)
# -fopenmp: Enable OpenMP threading
# -mcpu=a64fx: Generate SVE instructions specific to the A64FX chip
# -march=native: Auto-detect architecture features
# -flto: Link Time Optimization (inlines across files)
CFLAGS = -Ofast -std=c99 -pedantic -Wall -fopenmp -march=native -mcpu=a64fx -flto
LDFLAGS = -lm -fopenmp -flto

SRC_DIR = src
SOURCE  = $(SRC_DIR)/current.c \
          $(SRC_DIR)/emf.c \
          $(SRC_DIR)/particles.c \
          $(SRC_DIR)/random.c \
          $(SRC_DIR)/timer.c \
          $(SRC_DIR)/main.c \
          $(SRC_DIR)/simulation.c \
          $(SRC_DIR)/zdf.c
TARGET = zpic
DOCSBASE = docs
DOCS = $(DOCSBASE)/html/index.html
OBJ = $(SOURCE:.c=.o)

# Default target
all : $(SOURCE) $(TARGET)

# Build the executable
$(TARGET) : $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $@

# Compile object files
.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

# Documentation
docs : $(DOCS)
$(DOCS) : $(SOURCE)
	@doxygen ./Doxyfile

# Clean build files
clean:
	@touch $(TARGET) $(OBJ)
	rm -f $(TARGET) $(OBJ)
	rm -rf $(DOCSBASE)

# --- RUN TARGET ---
# Sets the optimized environment variables and executes the code.
# The backslashes (\) force these commands to run in a single shell instance.
run: $(TARGET)
	export OMP_NUM_THREADS=12; \
	export OMP_PROC_BIND=close; \
	export OMP_PLACES=cores; \
	export OMP_WAIT_POLICY=active; \
	./$(TARGET)
