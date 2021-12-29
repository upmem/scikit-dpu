
compile: compile_dpu compile_host

compile_dpu:
	dpu-clang -O2 -DSIZE_BATCH=32 -DNR_TASKLETS=16 -o trees_dpu_kernel_v2 trees_dpu_kernel_v2.c

compile_host:
	gcc -o host host.c -Wall -Wextra -g -O3 `dpu-pkg-config --cflags --libs dpu`

test: test/test*.h test/test_host_split.c test/test_host_commit.c

test/test%.h: FORCE compile_host
	dpu-clang -O2 -DNR_TASKLETS=16 -DSIZE_BATCH=32 -DDEBUG -DTEST -DTEST$* -o trees_dpu_kernel_v2 trees_dpu_kernel_v2.c
	./host > test/out_test$*

test/test_host_split.c : FORCE compile_dpu
	gcc -o test_host_split test/test_host_split.c ../dpu_management_v2.c -Wall -Wextra -g -O3 -I/usr/include/dpu -ldpu
	./test_host_split > test/out_test_host_split

test/test_host_commit.c : FORCE compile_dpu
	gcc -o test_host_commit test/test_host_commit.c ../dpu_management_v2.c -Wall -Wextra -g -O3 -I/usr/include/dpu -ldpu
	./test_host_commit > test/out_test_host_commit

FORCE:
