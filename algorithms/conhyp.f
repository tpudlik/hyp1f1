C     ****************************************************************  
C     *                                                              *  
C     *      SOLUTION TO THE CONFLUENT HYPERGEOMETRIC FUNCTION       *  
C     *                                                              *  
C     *                           by                                 *  
C     *                                                              *  
C     *                      MARK NARDIN,                            *  
C     *                                                              *  
C     *              W. F. PERGER and ATUL BHALLA                    *  
C     *                                                              *  
C     *                                                              *  
C     *  Michigan Technological University, Copyright 1989           *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : A numerical evaluator for the confluent       *  
C     *    hypergeometric function for complex arguments with large  *  
C     *    magnitudes using a direct summation of the Kummer series. *  
C     *    The method used allows an accuracy of up to thirteen      *  
C     *    decimal places through the use of large real arrays       *  
C     *    and a single final division.  LNCHF is a variable which   *  
C     *    selects how the result should be represented.  A '0' will *  
C     *    return the value in standard exponential form.  A '1'     *  
C     *    will return the natural log of the result.  IP is an      *
C     *    integer variable that specifies how many array positions  *
C     *    are desired (usually 10 is sufficient).  Setting IP=0     *
C     *    causes the program to estimate the number of array        *
C     *    positions.                                                *
C     *                                                              *  
C     *    The confluent hypergeometric function is the solution to  *  
C     *    the differential equation:                                *  
C     *                                                              *  
C     *      z M"(a;b;z) + (b-z) M'(a;b;z) - a M(a;b;z) = 0          *  
C     *                                                              *  
C     *  Subprograms called: BITS, CHGF                              *  
C     *                                                              *
C     ****************************************************************  
                                                                        
      FUNCTION CONHYP (A,B,Z,LNCHF,IP)                          
                                                                        
      INTEGER LNCHF,I,BITS,IP
      COMPLEX*16 CHGF,A,B,Z,CONHYP                                      
      DOUBLE PRECISION NTERM,FX,TERM1,MAX,TERM2,ANG,PI
      COMMON/CONSTS/ZERO,HALF,ONE,TWO,TEN,EPS
      DOUBLE PRECISION ZERO,HALF,ONE,TWO,TEN,EPS
                                                                        
      ZERO=0.0D0
      PI=TWO*TWO*ATAN(ONE) 
      IF (CDABS(Z) .NE. ZERO) THEN
        ANG=DATAN2(DIMAG(Z),DBLE(Z))                                   
      ELSE                                                              
        ANG=ONE                                                       
      ENDIF                                                             
      IF (DABS(ANG) .LT. (PI*HALF)) THEN                          
        ANG=ONE                                                       
      ELSE                                                              
        ANG=DSIN(DABS(ANG)-(PI*HALF))+ONE                  
      ENDIF                                                             
      MAX=ZERO                                                             
      NTERM=ZERO                                                           
      FX=ZERO                                                              
      TERM1=ZERO                                                           
10    NTERM=NTERM+ONE                                                   
      TERM2=CDABS((A+NTERM-1)*Z/((B+NTERM-1)*NTERM))                    
      IF (TERM2 .EQ. ZERO) GOTO 20                                     
      IF (TERM2 .LT. ONE) THEN                                        
        IF ((DBLE(A)+NTERM-1) .GT. ONE) THEN                         
          IF ((DBLE(B)+NTERM-1) .GT. ONE) THEN                       
            IF ((TERM2-TERM1) .LT. ZERO) THEN                          
              GOTO 20                                                   
            ENDIF                                                       
          ENDIF                                                         
        ENDIF                                                           
      ENDIF                                                             
      FX=FX+DLOG(TERM2)                                                 
      IF (FX .GT. MAX) MAX=FX                                           
      TERM1=TERM2                                                       
      GOTO 10                                                           
20    MAX=MAX*2/(BITS()*6.93147181D-1)                                  
      I=INT(MAX*ANG)+7                                                  
      IF (I .LT. 5) I=5                                                 
      IF (IP .GT. I) I=IP
      CONHYP=CHGF(A,B,Z,I,LNCHF)                                        
      RETURN                                                            
      END                                                               
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                   FUNCTION BITS                              *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Determines the number of significant figures  *  
C     *    of machine precision to arrive at the size of the array   *  
C     *    the numbers must must be stored in to get the accuracy    *  
C     *    of the solution.                                          *  
C     *                                                              *  
C     *  Subprograms called: none                                    *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      INTEGER FUNCTION BITS()                                           
                                                                        
      DOUBLE PRECISION BIT,BIT2                                                   
      INTEGER COUNT                                                     
      COMMON/CONSTS/ZERO,HALF,ONE,TWO,TEN,EPS
      DOUBLE PRECISION ZERO,HALF,ONE,TWO,TEN,EPS
                                                                        
      BIT=ONE                                                           
      COUNT=0                                                           
10    COUNT=COUNT+1                                                     
      BIT2=BIT*TWO                                                      
      BIT=BIT2+ONE                                                      
      IF ((BIT-BIT2) .NE. 0.0) GOTO 10                                  
      BITS=COUNT-1
      RETURN                                                            
      END                                                               
                                                                        
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                   FUNCTION CHGF                              *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Function that sums the Kummer series and      *  
C     *    returns the solution of the confluent hypergeometric      *  
C     *    function.                                                 *  
C     *                                                              *  
C     *  Subprograms called: ARMULT, ARYDIV, BITS, CMPADD, CMPMUL    *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      FUNCTION CHGF (A,B,Z,L,LNCHF)                          
                                                                        
      PARAMETER (LENGTH=777)
      INTEGER L,I,BITS,BIT,LNCHF,NMACH,ICOUNT,IXCNT
      COMPLEX*16 A,B,Z,FINAL,CHGF                                       
      DOUBLE PRECISION AR,AI,CR,CI,XR,XI,SUMR,SUMI,CNT,SIGFIG,MX1,MX2
      DOUBLE PRECISION NUMR,NUMI,DENOMR,DENOMI,RMAX
      DOUBLE PRECISION QR1,QR2,QI1,QI2,AR2,AI2,CR2,CI2,XR2,XI2
      DIMENSION SUMR(-1:LENGTH),SUMI(-1:LENGTH),NUMR(-1:LENGTH)
      DIMENSION NUMI(-1:LENGTH),DENOMR(-1:LENGTH),DENOMI(-1:LENGTH)
      DIMENSION QR1(-1:LENGTH),QR2(-1:LENGTH),QI1(-1:LENGTH),
     :          QI2(-1:LENGTH)
      COMMON/CONSTS/ZERO,HALF,ONE,TWO,TEN,EPS
      DOUBLE PRECISION ZERO,HALF,ONE,TWO,TEN,EPS
 
                                                                        
      BIT=BITS()                                                        
      RMAX=TWO**(BIT/2)                                               
      SIGFIG=TWO**(BIT/4)                                             
*
* SET TO ZERO ANY ARGUMENTS WHICH ARE BELOW THE PRECISION OF THE
* ALGORITHM.
*
      AR2=DBLE(A)*SIGFIG                                               
      AR=DINT(AR2)                                                      
      AR2=DNINT((AR2-AR)*RMAX)                                          
      AI2=DIMAG(A)*SIGFIG                                               
      AI=DINT(AI2)                                                      
      AI2=DNINT((AI2-AI)*RMAX)                                          
      CR2=DBLE(B)*SIGFIG                                               
      CR=DINT(CR2)                                                      
      CR2=DNINT((CR2-CR)*RMAX)                                          
      CI2=DIMAG(B)*SIGFIG                                               
      CI=DINT(CI2)                                                      
      CI2=DNINT((CI2-CI)*RMAX)                                          
      XR2=DBLE(Z)*SIGFIG                                               
      XR=DINT(XR2)                                                      
      XR2=DNINT((XR2-XR)*RMAX)                                          
      XI2=DIMAG(Z)*SIGFIG                                               
      XI=DINT(XI2)                                                      
      XI2=DNINT((XI2-XI)*RMAX)                                          
*
* WARN THE USER THAT THE INPUT VALUE WAS SO CLOSE TO ZERO THAT IT
* WAS SET EQUAL TO ZERO.
*
      IF ((DBLE(A).NE.ZERO) .AND. (AR.EQ.ZERO) .AND. (AR2.EQ.ZERO)) 
     :   WRITE (6,*) ' WARNING - REAL PART OF A WAS SET TO ZERO'
      IF ((DIMAG(A).NE.ZERO) .AND. (AI.EQ.ZERO) .AND. (AI2.EQ.ZERO))
     :   WRITE (6,*) ' WARNING - IMAG PART OF A WAS SET TO ZERO'
      IF ((DBLE(B).NE.ZERO) .AND. (CR.EQ.ZERO) .AND. (CR2.EQ.ZERO))
     :   WRITE (6,*) ' WARNING - REAL PART OF B WAS SET TO ZERO'
      IF ((DIMAG(B).NE.ZERO) .AND. (CI.EQ.ZERO) .AND. (CI2.EQ.ZERO))
     :   WRITE (6,*) ' WARNING - IMAG PART OF B WAS SET TO ZERO'
      IF ((DBLE(Z).NE.ZERO) .AND. (XR.EQ.ZERO) .AND. (XR2.EQ.ZERO))
     :   WRITE (6,*) ' WARNING - REAL PART OF Z WAS SET TO ZERO'
      IF ((DIMAG(Z).NE.ZERO) .AND. (XI.EQ.ZERO) .AND. (XI2.EQ.ZERO))
     :   WRITE (6,*) ' WARNING - IMAG PART OF Z WAS SET TO ZERO'
*
* SCREENING OF THE CASE WHEN B IS ZERO OR A NEGATIVE INTEGER.
*
      IF ((CR .EQ. ZERO) .AND. (CR2 .EQ. ZERO) .AND.
     :    (CI .EQ. ZERO) .AND. (CI2 .EQ. ZERO)) THEN
        WRITE (6,*) ' ERROR-- ARGUMENT B WAS EQUAL TO ZERO'
        STOP
      END IF
      NMACH=INT(LOG10(TWO**INT(BITS())))
      IF ((CI .EQ. ZERO) .AND. (CI2 .EQ. ZERO) .AND.
     :    (DBLE(B) .LT. ZERO)) THEN
        IF (ABS(DBLE(B)-DBLE(NINT(DBLE(B)))) .LT. TEN**(-NMACH)) THEN 
          WRITE (6,*) ' ERROR-- ARGUMENT B WAS A NEGATIVE INTEGER'
          STOP
        END IF
      END IF
      SUMR(-1)=ONE                                                    
      SUMI(-1)=ONE                                                    
      NUMR(-1)=ONE                                                    
      NUMI(-1)=ONE                                                    
      DENOMR(-1)=ONE                                                  
      DENOMI(-1)=ONE                                                  
      DO 100 I=0,L+1                                                    
        SUMR(I)=ZERO                                                   
        SUMI(I)=ZERO                                                   
        NUMR(I)=ZERO                                                   
        NUMI(I)=ZERO                                                   
        DENOMR(I)=ZERO                                                 
        DENOMI(I)=ZERO                                                 
100   CONTINUE                                                          
      SUMR(1)=ONE                                                     
      NUMR(1)=ONE                                                     
      DENOMR(1)=ONE                                                   
      CNT=SIGFIG                                                        
      ICOUNT=-1
      IF ((AI .EQ. ZERO) .AND. (AI2 .EQ. ZERO) .AND.
     :    (DBLE(A) .LT. ZERO)) THEN
        IF (ABS(DBLE(A)-DBLE(NINT(DBLE(A)))) .LT. TEN**(-NMACH)) 
     :     ICOUNT=-NINT(DBLE(A))
      END IF
      IXCNT=0
 110  IF (SUMR(1) .LT. HALF) THEN
        MX1=SUMI(L+1)
      ELSE IF (SUMI(1) .LT. HALF) THEN                                   
        MX1=SUMR(L+1)                                                   
      ELSE                                                              
        MX1=DMAX1(SUMR(L+1),SUMI(L+1))                                  
      ENDIF                                                             
      IF (NUMR(1) .LT. HALF) THEN                                        
        MX2=NUMI(L+1)                                                   
      ELSE IF (NUMI(1) .LT. HALF) THEN                                   
        MX2=NUMR(L+1)                                                   
      ELSE                                                              
        MX2=DMAX1(NUMR(L+1),NUMI(L+1))                                  
      ENDIF                                                             
      IF (MX1-MX2 .GT.  2.0) THEN                                       
        IF (CR .GT. ZERO) THEN                                         
          IF (CDABS(CMPLX(AR,AI)*CMPLX(XR,XI)/(CMPLX(CR,CI)*CNT))       
     :        .LE. ONE) GOTO 190                                      
        ENDIF                                                           
      ENDIF                                                             
      IF (IXCNT .EQ. ICOUNT) GO TO 190
      IXCNT=IXCNT+1
      CALL CMPMUL(SUMR,SUMI,CR,CI,QR1,QI1,L,RMAX)                       
      CALL CMPMUL(SUMR,SUMI,CR2,CI2,QR2,QI2,L,RMAX)                     
      QR2(L+1)=QR2(L+1)-1                                               
      QI2(L+1)=QI2(L+1)-1                                               
      CALL CMPADD(QR1,QI1,QR2,QI2,SUMR,SUMI,L,RMAX)                     
                                                                        
      CALL ARMULT(SUMR,CNT,SUMR,L,RMAX)                                 
      CALL ARMULT(SUMI,CNT,SUMI,L,RMAX)                                 
      CALL CMPMUL(DENOMR,DENOMI,CR,CI,QR1,QI1,L,RMAX)                   
      CALL CMPMUL(DENOMR,DENOMI,CR2,CI2,QR2,QI2,L,RMAX)                 
      QR2(L+1)=QR2(L+1)-1                                               
      QI2(L+1)=QI2(L+1)-1                                               
      CALL CMPADD(QR1,QI1,QR2,QI2,DENOMR,DENOMI,L,RMAX)                 
                                                                        
      CALL ARMULT(DENOMR,CNT,DENOMR,L,RMAX)                             
      CALL ARMULT(DENOMI,CNT,DENOMI,L,RMAX)                             
      CALL CMPMUL(NUMR,NUMI,AR,AI,QR1,QI1,L,RMAX)                       
      CALL CMPMUL(NUMR,NUMI,AR2,AI2,QR2,QI2,L,RMAX)                     
      QR2(L+1)=QR2(L+1)-1                                               
      QI2(L+1)=QI2(L+1)-1                                               
      CALL CMPADD(QR1,QI1,QR2,QI2,NUMR,NUMI,L,RMAX)                     
                                                                        
      CALL CMPMUL(NUMR,NUMI,XR,XI,QR1,QI1,L,RMAX)                       
      CALL CMPMUL(NUMR,NUMI,XR2,XI2,QR2,QI2,L,RMAX)                     
      QR2(L+1)=QR2(L+1)-1                                               
      QI2(L+1)=QI2(L+1)-1                                               
      CALL CMPADD(QR1,QI1,QR2,QI2,NUMR,NUMI,L,RMAX)                     
                                                                        
      CALL CMPADD(SUMR,SUMI,NUMR,NUMI,SUMR,SUMI,L,RMAX)                 
      CNT=CNT+SIGFIG                                                    
      AR=AR+SIGFIG                                                      
      CR=CR+SIGFIG                                                      
      GOTO 110                                                          
190   CALL ARYDIV(SUMR,SUMI,DENOMR,DENOMI,FINAL,L,LNCHF,RMAX,BIT)       
      CHGF=FINAL                                                        
      RETURN                                                            
      END                                                               
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                 SUBROUTINE ARADD                             *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Accepts two arrays of numbers and returns     *  
C     *    the sum of the array.  Each array is holding the value    *  
C     *    of one number in the series.  The parameter L is the      *  
C     *    size of the array representing the number and RMAX is     *  
C     *    the actual number of digits needed to give the numbers    *  
C     *    the desired accuracy.                                     *  
C     *                                                              *  
C     *  Subprograms called: none                                    *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      SUBROUTINE ARADD(A,B,C,L,RMAX)                                    
                                                                        
      INTEGER L
      DOUBLE PRECISION A,B,C,Z,RMAX                                               
      INTEGER EDIFF,I,J                                                 
      DIMENSION A(-1:*),B(-1:*),C(-1:*),Z(-1:777)                 
      COMMON/CONSTS/ZERO,HALF,ONE,TWO,TEN,EPS
      DOUBLE PRECISION ZERO,HALF,ONE,TWO,TEN,EPS
                                                                        
      DO 110 I=0,L+1                                                    
        Z(I)=ZERO                                                      
110   CONTINUE                                                          
      EDIFF=DNINT(A(L+1)-B(L+1))                                        
      IF (DABS(A(1)) .LT. HALF .OR. EDIFF .LE. -L) GOTO 111              
      IF (DABS(B(1)) .LT. HALF .OR. EDIFF .GE. L) GOTO 113               
      GOTO 115                                                          
111   DO 112 I=-1,L+1                                                   
        C(I)=B(I)                                                       
112   CONTINUE                                                          
      GOTO 311                                                          
113   DO 114 I=-1,L+1                                                   
        C(I)=A(I)                                                       
114   CONTINUE                                                          
      GOTO 311                                                          
115   Z(-1)=A(-1)                                                       
      IF (DABS(A(-1)-B(-1)) .LT. HALF) GOTO 200                          
      IF (EDIFF .GT. 0) THEN                                            
        Z(L+1)=A(L+1)                                                   
        GOTO 233                                                        
      ENDIF                                                             
      IF (EDIFF .LT. 0) THEN                                            
        Z(L+1)=B(L+1)                                                   
        Z(-1)=B(-1)                                                     
        GOTO 266                                                        
      ENDIF                                                             
      DO 120 I=1,L                                                      
        IF (A(I) .GT. B(I)) THEN                                        
          Z(L+1)=A(L+1)                                                 
          GOTO 233                                                      
        ENDIF                                                           
        IF (A(I) .LT. B(I)) THEN                                        
          Z(L+1)=B(L+1)                                                 
          Z(-1)=B(-1)                                                   
          GOTO 266                                                      
        ENDIF                                                           
120   CONTINUE                                                          
      GOTO 300                                                          
                                                                        
200   IF (EDIFF .GT. 0) GOTO 203                                        
      IF (EDIFF .LT. 0) GOTO 207                                        
      Z(L+1)=A(L+1)                                                     
      DO 201 I=L,1,-1                                                   
        Z(I)=A(I)+B(I)+Z(I)                                             
        IF (Z(I) .GE. RMAX) THEN                                        
          Z(I)=Z(I)-RMAX                                                
          Z(I-1)=ONE                                                  
        ENDIF                                                           
201   CONTINUE                                                          
      IF (Z(0) .GT. HALF) THEN                                           
        DO 202 I=L,1,-1                                                 
          Z(I)=Z(I-1)                                                   
202     CONTINUE                                                        
        Z(L+1)=Z(L+1)+ONE                                             
        Z(0)=ZERO                                                      
      ENDIF                                                             
      GOTO 300                                                          
203   Z(L+1)=A(L+1)                                                     
      DO 204 I=L,1+EDIFF,-1                                             
        Z(I)=A(I)+B(I-EDIFF)+Z(I)                                       
        IF (Z(I) .GE. RMAX) THEN                                        
          Z(I)=Z(I)-RMAX                                                
          Z(I-1)=ONE                                                  
        ENDIF                                                           
204   CONTINUE                                                          
      DO 205 I=EDIFF,1,-1                                               
        Z(I)=A(I)+Z(I)                                                  
        IF (Z(I) .GE. RMAX) THEN                                        
          Z(I)=Z(I)-RMAX                                                
          Z(I-1)=ONE                                                  
        ENDIF                                                           
205   CONTINUE                                                          
      IF (Z(0) .GT. HALF) THEN                                           
        DO 206 I=L,1,-1                                                 
          Z(I)=Z(I-1)                                                   
206     CONTINUE                                                        
        Z(L+1)=Z(L+1)+1                                                 
        Z(0)=ZERO                                                      
      ENDIF                                                             
      GOTO 300                                                          
207   Z(L+1)=B(L+1)                                                     
      DO 208 I=L,1-EDIFF,-1                                             
        Z(I)=A(I+EDIFF)+B(I)+Z(I)                                       
        IF (Z(I) .GE. RMAX) THEN                                        
          Z(I)=Z(I)-RMAX                                                
          Z(I-1)=ONE                                                  
        ENDIF                                                           
208   CONTINUE                                                          
      DO 209 I=0-EDIFF,1,-1                                             
        Z(I)=B(I)+Z(I)                                                  
        IF (Z(I) .GE. RMAX) THEN                                        
          Z(I)=Z(I)-RMAX                                                
          Z(I-1)=ONE                                                  
        ENDIF                                                           
209   CONTINUE                                                          
      IF (Z(0) .GT. HALF) THEN                                           
        DO 210 I=L,1,-1                                                 
          Z(I)=Z(I-1)                                                   
210     CONTINUE                                                        
        Z(L+1)=Z(L+1)+ONE                                             
        Z(0)=ZERO                                                      
      ENDIF                                                             
      GOTO 300                                                          
                                                                        
233   IF (EDIFF .GT. 0) GOTO 243                                        
      DO 234 I=L,1,-1                                                   
        Z(I)=A(I)-B(I)+Z(I)                                             
        IF (Z(I) .LT. ZERO) THEN                                       
          Z(I)=Z(I)+RMAX                                                
          Z(I-1)=-ONE                                                 
        ENDIF                                                           
234   CONTINUE                                                          
      GOTO 290                                                          
243   DO 244 I=L,1+EDIFF,-1                                             
        Z(I)=A(I)-B(I-EDIFF)+Z(I)                                       
        IF (Z(I) .LT. ZERO) THEN                                       
          Z(I)=Z(I)+RMAX                                                
          Z(I-1)=-ONE                                                 
        ENDIF                                                           
244   CONTINUE                                                          
      DO 245 I=EDIFF,1,-1                                               
        Z(I)=A(I)+Z(I)                                                  
        IF (Z(I) .LT. ZERO) THEN                                       
          Z(I)=Z(I)+RMAX                                                
          Z(I-1)=-ONE                                                 
        ENDIF                                                           
245   CONTINUE                                                          
      GOTO 290                                                          
                                                                        
266   IF (EDIFF .LT. 0) GOTO 276                                        
      DO 267 I=L,1,-1                                                   
        Z(I)=B(I)-A(I)+Z(I)                                             
        IF (Z(I) .LT. ZERO) THEN                                       
          Z(I)=Z(I)+RMAX                                                
          Z(I-1)=-ONE                                                 
        ENDIF                                                           
267   CONTINUE                                                          
      GOTO 290                                                          
276   DO 277 I=L,1-EDIFF,-1                                             
        Z(I)=B(I)-A(I+EDIFF)+Z(I)                                       
        IF (Z(I) .LT. ZERO) THEN                                       
          Z(I)=Z(I)+RMAX                                                
          Z(I-1)=-ONE                                                 
        ENDIF                                                           
277   CONTINUE                                                          
      DO 278 I=0-EDIFF,1,-1                                             
        Z(I)=B(I)+Z(I)                                                  
        IF (Z(I) .LT. ZERO) THEN                                       
          Z(I)=Z(I)+RMAX                                                
          Z(I-1)=-ONE                                                 
        ENDIF                                                           
278   CONTINUE                                                          
                                                                        
290   IF (Z(1) .GT. HALF) GOTO 300                                       
      I=1                                                               
291     I=I+1                                                           
        IF (Z(I) .LT. HALF .AND. I .LT. L+1) GOTO 291                    
      IF (I .EQ. L+1) THEN                                              
        Z(-1)=ONE                                                     
        Z(L+1)=ZERO                                                    
        GOTO 300                                                        
      ENDIF                                                             
292   DO 293 J=1,L+1-I                                                  
        Z(J)=Z(J+I-1)                                                   
293   CONTINUE                                                          
      DO 294 J=L+2-I,L                                                  
        Z(J)=ZERO                                                      
294   CONTINUE                                                          
      Z(L+1)=Z(L+1)-I+1                                                 
300   DO 310 I=-1,L+1                                                   
        C(I)=Z(I)                                                       
310   CONTINUE                                                          
311   IF (C(1) .LT. HALF) THEN                                           
        C(-1)=ONE                                                     
        C(L+1)=ZERO                                                    
      ENDIF                                                             
      RETURN                                                            
      END                                                               
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                 SUBROUTINE ARSUB                             *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Accepts two arrays and subtracts each element *  
C     *    in the second array from the element in the first array   *  
C     *    and returns the solution.  The parameters L and RMAX are  *  
C     *    the size of the array and the number of digits needed for *  
C     *    the accuracy, respectively.                               *  
C     *                                                              *  
C     *  Subprograms called: ARADD                                   *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      SUBROUTINE ARSUB(A,B,C,L,RMAX)                                    
                                                                        
      INTEGER L,I
      DOUBLE PRECISION A,B,C,B2,RMAX                                              
      DIMENSION A(-1:*),B(-1:*),C(-1:*),B2(-1:777)                
      COMMON/CONSTS/ZERO,HALF,ONE,TWO,TEN,EPS
      DOUBLE PRECISION ZERO,HALF,ONE,TWO,TEN,EPS
                                                                        
      DO 100 I=-1,L+1                                                   
        B2(I)=B(I)                                                      
100   CONTINUE                                                          
      B2(-1)=(-ONE)*B2(-1)                                            
      CALL ARADD(A,B2,C,L,RMAX)                                         
      RETURN                                                            
      END                                                               
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                 SUBROUTINE ARMULT                            *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Accepts two arrays and returns the product.   *  
C     *    L and RMAX are the size of the arrays and the number of   *  
C     *    digits needed to represent the numbers with the required  *  
C     *    accuracy.                                                 *  
C     *                                                              *  
C     *  Subprograms called: none                                    *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      SUBROUTINE ARMULT(A,B,C,L,RMAX)                                   
                                                                        
      INTEGER L
      DOUBLE PRECISION A,B,C,Z,B2,CARRY,RMAX,RMAX2                                
      DIMENSION A(-1:*),C(-1:*),Z(-1:777)                           
      INTEGER I                                                         
      COMMON/CONSTS/ZERO,HALF,ONE,TWO,TEN,EPS
      DOUBLE PRECISION ZERO,HALF,ONE,TWO,TEN,EPS
                                                                        
      RMAX2=ONE/RMAX                                                  
      Z(-1)=DSIGN(ONE,B)*A(-1)                                        
      B2=DABS(B)                                                        
      Z(L+1)=A(L+1)                                                     
      DO 100 I=0,L                                                      
        Z(I)=ZERO                                                      
100   CONTINUE                                                          
      IF (B2 .LE. EPS .OR. A(1) .LE. EPS) THEN                  
        Z(-1)=ONE                                                     
        Z(L+1)=ZERO                                                    
        GOTO 198                                                        
      ENDIF                                                             
      DO 110 I=L,1,-1                                                   
        Z(I)=A(I)*B2+Z(I)                                               
        IF (Z(I) .GE. RMAX) THEN                                        
          CARRY=DINT(Z(I)/RMAX)                                         
          Z(I)=Z(I)-CARRY*RMAX                                          
          Z(I-1)=CARRY                                                  
        ENDIF                                                           
110   CONTINUE                                                          
      IF (Z(0) .LT. HALF) GOTO 150                                       
      DO 120 I=L,1,-1                                                   
        Z(I)=Z(I-1)                                                     
120   CONTINUE                                                          
      Z(L+1)=Z(L+1)+ONE                                               
      Z(0)=ZERO                                                        
150   CONTINUE                                                          
                                                                        
                                                                        
198   DO 199 I=-1,L+1                                                   
        C(I)=Z(I)                                                       
199   CONTINUE                                                          
      IF (C(1) .LT. HALF) THEN                                           
        C(-1)=ONE                                                     
        C(L+1)=ZERO                                                    
      ENDIF                                                             
      RETURN                                                            
      END                                                               
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                 SUBROUTINE CMPADD                            *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Takes two arrays representing one real and    *  
C     *    one imaginary part, and adds two arrays representing      *  
C     *    another complex number and returns two array holding the  *  
C     *    complex sum.                                              *  
C     *              (CR,CI) = (AR+BR, AI+BI)                        *  
C     *                                                              *  
C     *  Subprograms called: ARADD                                   *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      SUBROUTINE CMPADD(AR,AI,BR,BI,CR,CI,L,RMAX)                       
                                                                        
      INTEGER L
      DOUBLE PRECISION AR,AI,BR,BI,CR,CI,RMAX                                     
      DIMENSION AR(-1:*),AI(-1:*),BR(-1:*),BI(-1:*)             
      DIMENSION CR(-1:*),CI(-1:*)                                   
                                                                        
      CALL ARADD(AR,BR,CR,L,RMAX)                                       
      CALL ARADD(AI,BI,CI,L,RMAX)                                       
      RETURN                                                            
      END                                                               
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                 SUBROUTINE CMPSUB                            *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Takes two arrays representing one real and    *  
C     *    one imaginary part, and subtracts two arrays representing *  
C     *    another complex number and returns two array holding the  *  
C     *    complex sum.                                              *  
C     *              (CR,CI) = (AR+BR, AI+BI)                        *  
C     *                                                              *  
C     *  Subprograms called: ARADD                                   *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      SUBROUTINE CMPSUB(AR,AI,BR,BI,CR,CI,L,RMAX)                       
                                                                        
      INTEGER L
      DOUBLE PRECISION AR,AI,BR,BI,CR,CI,RMAX                                     
      DIMENSION AR(-1:*),AI(-1:*),BR(-1:*),BI(-1:*)             
      DIMENSION CR(-1:*),CI(-1:*)                                   
                                                                        
      CALL ARSUB(AR,BR,CR,L,RMAX)                                       
      CALL ARSUB(AI,BI,CI,L,RMAX)                                       
      RETURN                                                            
      END                                                               
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                 SUBROUTINE CMPMUL                            *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Takes two arrays representing one real and    *  
C     *    one imaginary part, and multiplies it with two arrays     *  
C     *    representing another complex number and returns the       *  
C     *    complex product.                                          *  
C     *                                                              *  
C     *  Subprograms called: ARMULT, ARSUB, ARADD                    *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      SUBROUTINE CMPMUL(AR,AI,BR,BI,CR,CI,L,RMAX)                       
                                                                        
      INTEGER L
      DOUBLE PRECISION AR,AI,BR,BI,CR,CI,D1,D2,RMAX                           
      DIMENSION AR(-1:*),AI(-1:*),CR(-1:*),CI(-1:*)             
      DIMENSION D1(-1:777),D2(-1:777)                       
                                                                        
      CALL ARMULT(AR,BR,D1,L,RMAX)                                      
      CALL ARMULT(AI,BI,D2,L,RMAX)                                      
      CALL ARSUB(D1,D2,CR,L,RMAX)                                      
      CALL ARMULT(AR,BI,D1,L,RMAX)                                      
      CALL ARMULT(AI,BR,D2,L,RMAX)                                      
      CALL ARADD(D1,D2,CI,L,RMAX)                                       
      RETURN                                                            
      END                                                               
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                 SUBROUTINE ARYDIV                            *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Returns the double precision complex number   *  
C     *    resulting from the division of four arrays, representing  *  
C     *    two complex numbers.  The number returned will be in one  *  
C     *    two different forms:  either standard scientific or as    *  
C     *    the natural log of the number.                            *  
C     *                                                              *  
C     *  Subprograms called: CONV21, CONV12, EADD, ECPDIV, EMULT     *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      SUBROUTINE ARYDIV(AR,AI,BR,BI,C,L,LNCHF,RMAX,BIT)                 
                                                                        
      INTEGER L,BIT,REXP,IR10,II10,LNCHF
      COMPLEX*16 C                                                      
      DOUBLE PRECISION AR,AI,BR,BI,PHI,N1,N2,N3,E1,E2,E3,RR10,RI10,X              
      DOUBLE PRECISION AE,BE,X1,X2,DUM1,DUM2,CE,RMAX                              
      DIMENSION AR(-1:*),AI(-1:*),BR(-1:*),BI(-1:*)             
      DIMENSION AE(2,2),BE(2,2),CE(2,2)                                 
      COMMON/CONSTS/ZERO,HALF,ONE,TWO,TEN,EPS
      DOUBLE PRECISION ZERO,HALF,ONE,TWO,TEN,EPS
                                                                        
      REXP=BIT/2                                                        
      X=REXP*(AR(L+1)-2)                                                
      RR10=X*DLOG10(TWO)/DLOG10(TEN)                               
      IR10=INT(RR10)                                                    
      RR10=RR10-IR10                                                    
      X=REXP*(AI(L+1)-2)                                                
      RI10=X*DLOG10(TWO)/DLOG10(TEN)                               
      II10=INT(RI10)                                                    
      RI10=RI10-II10                                                    
      DUM1=DSIGN(AR(1)*RMAX*RMAX+AR(2)*RMAX+AR(3),AR(-1))               
      DUM2=DSIGN(AI(1)*RMAX*RMAX+AI(2)*RMAX+AI(3),AI(-1))               
      DUM1=DUM1*10**RR10                                                
      DUM2=DUM2*10**RI10                                                
      CALL CONV12(DCMPLX(DUM1,DUM2),AE)                                 
      AE(1,2)=AE(1,2)+IR10                                              
      AE(2,2)=AE(2,2)+II10                                              
      X=REXP*(BR(L+1)-2)                                                
      RR10=X*DLOG10(TWO)/DLOG10(TEN)                               
      IR10=INT(RR10)                                                    
      RR10=RR10-IR10                                                    
      X=REXP*(BI(L+1)-2)                                                
      RI10=X*DLOG10(TWO)/DLOG10(TEN)                               
      II10=INT(RI10)                                                    
      RI10=RI10-II10                                                    
      DUM1=DSIGN(BR(1)*RMAX*RMAX+BR(2)*RMAX+BR(3),BR(-1))               
      DUM2=DSIGN(BI(1)*RMAX*RMAX+BI(2)*RMAX+BI(3),BI(-1))               
      DUM1=DUM1*10**RR10                                                
      DUM2=DUM2*10**RI10                                                
      CALL CONV12(DCMPLX(DUM1,DUM2),BE)                                 
      BE(1,2)=BE(1,2)+IR10                                              
      BE(2,2)=BE(2,2)+II10                                              
      CALL ECPDIV(AE,BE,CE)                                             
      IF (LNCHF .EQ. 0) THEN                                            
        CALL CONV21(CE,C)                                               
      ELSE                                                              
        CALL EMULT(CE(1,1),CE(1,2),CE(1,1),CE(1,2),N1,E1)               
        CALL EMULT(CE(2,1),CE(2,2),CE(2,1),CE(2,2),N2,E2)               
        CALL EADD(N1,E1,N2,E2,N3,E3)                                    
        N1=CE(1,1)                                                      
        E1=CE(1,2)-CE(2,2)                                              
        X2=CE(2,1)                                                      
        IF (E1 .GT. 74.0D0) THEN                                        
          X1=1.0D75                                                     
        ELSEIF (E1 .LT. -74.0D0) THEN                                   
          X1=0                                                          
        ELSE                                                            
          X1=N1*(10**E1)                                                
        ENDIF                                                           
        PHI=DATAN2(X2,X1)                                               
        C=DCMPLX(HALF*(DLOG(N3)+E3*DLOG(TEN)),PHI)                 
      ENDIF                                                             
      RETURN                                                            
      END                                                               
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                 SUBROUTINE EMULT                             *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Takes one base and exponent and multiplies it *  
C     *    by another numbers base and exponent to give the product  *  
C     *    in the form of base and exponent.                         *  
C     *                                                              *  
C     *  Subprograms called: none                                    *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      SUBROUTINE EMULT(N1,E1,N2,E2,NF,EF)                               
                                                                        
      DOUBLE PRECISION N1,E1,N2,E2,NF,EF                                          
      COMMON/CONSTS/ZERO,HALF,ONE,TWO,TEN,EPS
      DOUBLE PRECISION ZERO,HALF,ONE,TWO,TEN,EPS
                                                                        
      NF=N1*N2                                                          
      EF=E1+E2                                                          
      IF (DABS(NF) .GE. TEN) THEN                                    
        NF=NF/TEN                                                    
        EF=EF+ONE                                                     
      ENDIF                                                             
      RETURN                                                            
      END                                                               
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                 SUBROUTINE EDIV                              *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : returns the solution in the form of base and  *  
C     *    exponent of the division of two exponential numbers.      *  
C     *                                                              *  
C     *  Subprograms called: none                                    *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      SUBROUTINE EDIV(N1,E1,N2,E2,NF,EF)       
                                                                        
      DOUBLE PRECISION N1,E1,N2,E2,NF,EF  
      COMMON/CONSTS/ZERO,HALF,ONE,TWO,TEN,EPS
      DOUBLE PRECISION ZERO,HALF,ONE,TWO,TEN,EPS
                                                                        
      NF=N1/N2                                                          
      EF=E1-E2                                                          
      IF ((DABS(NF) .LT. ONE) .AND. (NF .NE. ZERO)) THEN             
        NF=NF*TEN                                                    
        EF=EF-ONE                                                     
      ENDIF                                                             
      RETURN                                                            
      END                                                               
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                 SUBROUTINE EADD                              *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Returns the sum of two numbers in the form    *  
C     *    of a base and an exponent.                                *  
C     *                                                              *  
C     *  Subprograms called: none                                    *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      SUBROUTINE EADD(N1,E1,N2,E2,NF,EF)                                
                                                                        
      DOUBLE PRECISION N1,E1,N2,E2,NF,EF,EDIFF,THIRSIX
      COMMON/CONSTS/ZERO,HALF,ONE,TWO,TEN,EPS
      DOUBLE PRECISION ZERO,HALF,ONE,TWO,TEN,EPS
                                                                        
      EDIFF=E1-E2                                                       
      THIRSIX=(ONE+TWO)*(TWO+TEN)
      IF (EDIFF .GT. THIRSIX) THEN                                       
        NF=N1                                                           
        EF=E1                                                           
      ELSE IF (EDIFF .LT. -THIRSIX) THEN                                 
        NF=N2                                                           
        EF=E2                                                           
      ELSE                                                              
        NF=N1*(TEN**EDIFF)+N2                                        
        EF=E2                                                           
400     IF (DABS(NF) .LT. TEN) GOTO 410                              
          NF=NF/TEN                                                  
          EF=EF+ONE                                                   
          GOTO 400                                                      
410     IF ((DABS(NF) .GE. ONE) .OR. (NF .EQ. ZERO)) GOTO 420        
          NF=NF*TEN                                                  
          EF=EF-ONE                                                   
          GOTO 410                                                      
      ENDIF                                                             
420   RETURN                                                            
      END                                                               
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                 SUBROUTINE ESUB                              *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Returns the solution to the subtraction of    *  
C     *    two numbers in the form of base and exponent.             *  
C     *                                                              *  
C     *  Subprograms called: EADD                                    *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      SUBROUTINE ESUB(N1,E1,N2,E2,NF,EF)                                
                                                                        
      DOUBLE PRECISION N1,E1,N2,E2,NF,EF         
      COMMON/CONSTS/ZERO,HALF,ONE,TWO,TEN,EPS
      DOUBLE PRECISION ZERO,HALF,ONE,TWO,TEN,EPS
                                                                        
      CALL EADD(N1,E1,N2*(-ONE),E2,NF,EF)                             
      RETURN                                                            
      END                                                               
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                 SUBROUTINE CONV12                            *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Converts a number from complex notation to a  *  
C     *    form of a 2x2 real array.                                 *  
C     *                                                              *  
C     *  Subprograms called: none                                    *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      SUBROUTINE CONV12(CN,CAE)                                         
                                                                        
      COMPLEX*16 CN                                                     
      DOUBLE PRECISION CAE                                                        
      DIMENSION CAE(2,2)                                                
      COMMON/CONSTS/ZERO,HALF,ONE,TWO,TEN,EPS
      DOUBLE PRECISION ZERO,HALF,ONE,TWO,TEN,EPS
                                                                        
      CAE(1,1)=DBLE(CN)                                                
      CAE(1,2)=ZERO                                                    
300   IF (DABS(CAE(1,1)) .LT. TEN) GOTO 310                          
        CAE(1,1)=CAE(1,1)/TEN                                        
        CAE(1,2)=CAE(1,2)+ONE                                         
        GOTO 300                                                        
310   IF ((DABS(CAE(1,1)) .GE. ONE) .OR. (CAE(1,1) .EQ. ZERO))       
     : GOTO 320                                                         
        CAE(1,1)=CAE(1,1)*TEN                                        
        CAE(1,2)=CAE(1,2)-ONE                                         
        GOTO 310                                                        
320   CAE(2,1)=DIMAG(CN)                                                
      CAE(2,2)=ZERO                                                    
330   IF (DABS(CAE(2,1)) .LT. TEN) GOTO 340                          
        CAE(2,1)=CAE(2,1)/TEN                                        
        CAE(2,2)=CAE(2,2)+ONE                                         
        GOTO 330                                                        
340   IF ((DABS(CAE(2,1)) .GE. ONE) .OR. (CAE(2,1) .EQ. ZERO))       
     : GOTO 350                                                         
        CAE(2,1)=CAE(2,1)*TEN                                        
        CAE(2,2)=CAE(2,2)-ONE                                         
        GOTO 340                                                        
350   RETURN                                                            
      END                                                               
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                 SUBROUTINE CONV21                            *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Converts a number represented in a 2x2 real   *  
C     *    array to the form of a complex number.                    *  
C     *                                                              *  
C     *  Subprograms called: none                                    *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      SUBROUTINE CONV21(CAE,CN)                                         
                                                                        
      DOUBLE PRECISION CAE                                                        
      COMPLEX*16 CN                                                     
      DIMENSION CAE(2,2)                                                
      COMMON/CONSTS/ZERO,HALF,ONE,TWO,TEN,EPS
      DOUBLE PRECISION ZERO,HALF,ONE,TWO,TEN,EPS
      IF (CAE(1,2) .GT. 75 .OR. CAE(2,2) .GT. 75) THEN                  
        CN=CMPLX(1.0D75,1.0D75)                                         
      ELSE IF (CAE(2,2) .LT. -75) THEN                                  
        CN=DCMPLX(CAE(1,1)*(10**CAE(1,2)),ZERO)                          
      ELSE                                                              
        CN=DCMPLX(CAE(1,1)*(10**CAE(1,2)),CAE(2,1)*(10**CAE(2,2)))      
      ENDIF                                                             
      RETURN                                                            
      END                                                               
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                 SUBROUTINE ECPMUL                            *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Multiplies two numbers which are each         *  
C     *    represented in the form of a two by two array and returns *  
C     *    the solution in the same form.                            *  
C     *                                                              *  
C     *  Subprograms called: EMULT, ESUB, EADD                       *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      SUBROUTINE ECPMUL(A,B,C)                                          
                                                                        
      DOUBLE PRECISION A,B,C,N1,E1,N2,E2,C2                                       
      DIMENSION A(2,2),B(2,2),C(2,2),C2(2,2)                            
                                                                        
      CALL EMULT(A(1,1),A(1,2),B(1,1),B(1,2),N1,E1)                     
      CALL EMULT(A(2,1),A(2,2),B(2,1),B(2,2),N2,E2)                     
      CALL ESUB(N1,E1,N2,E2,C2(1,1),C2(1,2))                            
      CALL EMULT(A(1,1),A(1,2),B(2,1),B(2,2),N1,E1)                     
      CALL EMULT(A(2,1),A(2,2),B(1,1),B(1,2),N2,E2)                     
      CALL EADD(N1,E1,N2,E2,C(2,1),C(2,2))                              
      C(1,1)=C2(1,1)                                                    
      C(1,2)=C2(1,2)                                                    
      RETURN                                                            
      END                                                               
                                                                        
                                                                        
C     ****************************************************************  
C     *                                                              *  
C     *                 SUBROUTINE ECPDIV                            *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Divides two numbers and returns the solution. *  
C     *    All numbers are represented by a 2x2 array.               *  
C     *                                                              *  
C     *  Subprograms called: EADD, ECPMUL, EDIV, EMULT               *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      SUBROUTINE ECPDIV(A,B,C)                                          
                                                                        
      DOUBLE PRECISION A,B,C,N1,E1,N2,E2,B2,N3,E3,C2                              
      DIMENSION A(2,2),B(2,2),C(2,2),B2(2,2),C2(2,2)                    
      COMMON/CONSTS/ZERO,HALF,ONE,TWO,TEN,EPS
      DOUBLE PRECISION ZERO,HALF,ONE,TWO,TEN,EPS
                                                                        
      B2(1,1)=B(1,1)                                                    
      B2(1,2)=B(1,2)                                                    
      B2(2,1)=-ONE*B(2,1)                                             
      B2(2,2)=B(2,2)                                                    
      CALL ECPMUL(A,B2,C2)                                              
      CALL EMULT(B(1,1),B(1,2),B(1,1),B(1,2),N1,E1)                     
      CALL EMULT(B(2,1),B(2,2),B(2,1),B(2,2),N2,E2)                     
      CALL EADD(N1,E1,N2,E2,N3,E3)                                      
      CALL EDIV(C2(1,1),C2(1,2),N3,E3,C(1,1),C(1,2))                    
      CALL EDIV(C2(2,1),C2(2,2),N3,E3,C(2,1),C(2,2))                    
      RETURN                                                            
      END                                                               
C     ****************************************************************  
C     *                                                              *  
C     *                 BLOCK DATA BLDAT1                            *  
C     *                                                              *  
C     *                                                              *  
C     *  Description : Sets of frequently used numbers in a common   *  
C     *    block.  This makes it easier to convert the code to a     *  
C     *    single precision version.                                 *  
C     *                                                              *  
C     ****************************************************************  
                                                                        
      BLOCK DATA BLDAT1
C
      COMMON/CONSTS/ZERO,HALF,ONE,TWO,TEN,EPS
      DOUBLE PRECISION ZERO,HALF,ONE,TWO,TEN,EPS
      DATA ZERO,HALF,ONE,TWO,TEN,EPS/0.0D0,0.5D0,1.0D0,2.0D0,
     :     10.0D0,1.0D-10/
      END
