      function npb_hyp1f1(a, b, z, chf)
      COMPLEX*16 a, b, z, chf
Cf2py intent(in) a, b, z
Cf2py intent(out) chf
      chf=conhyp(a, b, z, 0, 0)
      end
