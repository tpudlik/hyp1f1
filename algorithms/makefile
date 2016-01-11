FFLAGS = -O
LDFLAGS = -llapack

npb : npb_hyp1f1.f conhyp.f
	f2py -m npb_hyp1f1 -h npb_hyp1f1.pyf npb_hyp1f1.f
	f2py -c npb_hyp1f1.pyf npb_hyp1f1.f conhyp.f

clean :
	rm -rf *.pyf *.so *~
