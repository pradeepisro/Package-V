CC=clang++-4.0
CFLAGS= -Wall -g -std=c++14

%: %.cpp
	$(CC) -o $@ $@.cpp $(CFLAGS)
clean:
	find . -maxdepth 1 -type f -executable -exec rm {} +
	rm core*
