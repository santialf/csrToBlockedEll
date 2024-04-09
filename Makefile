all: spmm_csr_example

spmm_csr_example: spmm_csr_example.c
	gcc $(INC) spmm_csr_example.c mmio.c smsh.c -o spmm_csr_example $(LIBS)
	gcc mmio.h 

clean:
	rm -f spmm_csr_example

test:
	@echo "\n==== SpMM CSR Test ====\n"
	./spmm_csr_example

.PHONY: clean all test
