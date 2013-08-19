CFLAGS="-g -O3" 

all: saftmain.so saftsparse.so saftstats.so

saftmain.so: saftmain.pyx
	CFLAGS=$(CFLAGS) python setup_saftmain.py build_ext --inplace
saftsparse.so: saftsparse.pyx
	CFLAGS=$(CFLAGS) python setup_saftsparse.py build_ext --inplace
saftstats.so: saftstats.pyx
	CFLAGS=$(CFLAGS) python setup_saftstats.py build_ext --inplace

clean:
	rm saftmain.c saftsparse.c saftstats.c saftmain.so saftsparse.so saftstats.so
