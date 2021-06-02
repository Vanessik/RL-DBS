 swig -python osc.i
 g++ -c -fpic osc_wrap.c osc.c -I/usr/include/python3.7
 g++ -shared osc.o osc_wrap.o -o _oscillator_cpp.so
