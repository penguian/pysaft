CFLAGS="-g -O3"

so_files=MPIsaftmain.so saftmain.so saftmpi.so saftsparse.so saftstage.so saftstats.so
all: $(so_files)

%.so: %.pyx
	CFLAGS=$(CFLAGS) ext_name=$* source_pyx=$*.pyx python setup.py build_ext --inplace
saftstats.so: saftstats.pyx
	CFLAGS=$(CFLAGS) python setup_saftstats.py build_ext --inplace

clean:
	rm -f $(so_files:.so=.c) $(so_files); rm -rf build
