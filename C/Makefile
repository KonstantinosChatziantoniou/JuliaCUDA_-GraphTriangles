CC = nvcc
CFLAGS = -O2 --default-stream per-thread  -Xcompiler -lcuda -lcudart 
#-std=c99 --default-stream per-thread
LIBS = -lm #-Wall -Wextra
OBJDIR = ./bin
HDRDIR = ./headers
SRCDIR = ./src

_OBJ =  main.o preprocess.o helper.o kernel_row_per_th.o kernel_elem_per_th.o kernel_elem_per_th_limit.o
OBJ = $(patsubst %, $(OBJDIR)/%, $(_OBJ))

_DEPS = preprocess.h helper.h kernel_row_per_th.h kernel_elem_per_th.h kernel_elem_per_th_limit.h
DEPS = $(patsubst %, $(HDRDIR)/%, $(_DEPS))


mainProgram: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)


$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(DEPS)
	$(CC) -c -o $@ $<  $(CFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	gcc -c -o $@ $<  -O3


#bin/cudaWrapper.o: src/cudaWrapper.cu headers/cudaWrapper.h
#	nvcc  -c -o bin/cudaWrapper.o src/cudaWrapper.cu 
clean:
	rm -rf ./*.csv; rm -rf ./bin/*.o; rm -rf mainProgram

clear: 
	rm -rf ./*.csv; rm -rf ./PythonCheck/*.csv