/*******  amplitude response curve for the SL oscillators   ******/
/*******            charge-balanced pulses                  ******/

#include <iostream> 
#include <fstream> 
#include <math.h>
#include "my_defs.h"
#include "nrlib.h"
using namespace std;

/*************  important parameters  **************/
const double omega=5, alpha=0.1, Om=omega-alpha;   // systems's parameters
const double Amp=0.1;      //  kick amplitude
const int width_p=10;       //  width of the 1st pulse in steps
const int gap=10;        //   gap in steps 
const double Kfactor=3;    //   ratio of the 2nd width over 1st width
const int width_n=width_p*Kfactor; // width of the 2nd pulse in steps
double P;                  // current rectangular pulse' amplitude
const double tstep=0.001;                          // intergration step

/*******************************************************/
//***********  supplementary routines from nrutil.c  ***********
//**************************************************************

#define NR_END 1
#define FREE_ARG char*

void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
        fprintf(stderr,"Numerical Recipes run-time error...\n");
        fprintf(stderr,"%s\n",error_text);
        fprintf(stderr,"...now exiting to system...\n");
        exit(1);
}


double *dvector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
        double *v;
        v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
        char msg[]="allocation failure in dvector()";
        if (!v) nrerror(msg);
        return v-nl+NR_END;
}

void free_dvector(double *v, long nl, long nh)
/* free a double vector allocated with dvector() */
{
        free((FREE_ARG) (v+nl-NR_END));
}

/*************************************************************/
//==== Simple Runge-Kutta Integrator ===============
void rk(double y[],int n,double &x,double h,
     void (*derivs)(double, double[], double[]), int nt)
{
  int it,i;
  double hh,h6,xh,*yt,*dy0,*dyt,*dym;
  yt=dvector(1,n);    dy0=dvector(1,n);
  dyt=dvector(1,n);   dym=dvector(1,n);

  hh=h*0.5;    h6=h/6.;
  for (it=1;it<=nt;it++)
    {
      xh=x+hh;
      (*derivs)(x,y,dy0);
      for(i=1;i<=n;i++) yt[i]=y[i]+hh*dy0[i];
      (*derivs)(xh,yt,dyt);
      for(i=1;i<=n;i++) yt[i]=y[i]+hh*dyt[i];
      (*derivs)(xh,yt,dym);
      for(i=1;i<=n;i++)
        {
          yt[i]=y[i]+h*dym[i];
          dym[i]=dyt[i]+dym[i];
        }
      x+=h;
      (*derivs)(x,yt,dyt);
      for(i=1;i<=n;i++) y[i]+=h6*(dy0[i]+2.*dym[i]+dyt[i]);
    }
  free_dvector(yt,1,n);     free_dvector(dyt,1,n);
  free_dvector(dy0,1,n);    free_dvector(dym,1,n);
}

/*******************************************************/
void SL(double t, double y[], double ydot[])
{  
  double R2;  // equations in polar coordinates

  R2=SQR(y[1]);
  ydot[1]= y[1]*(1-R2) + P*cos(y[2]); 
  ydot[2]=omega-alpha*R2 -P*sin(y[2])/y[1];
}
/*******************************************************/
void AutonomEvolution(double y[], double tau)
{
  double R0, R02, theta0, ex;
  R0=y[1];  R02=SQR(R0);  theta0=y[2]; ex=exp(-2*tau);

  y[1]=1/sqrt(1+ex*(1-R02)/R02);                  // evolution of R
  y[2]=theta0+Om*tau-alpha*log(R02+(1-R02)*ex)/2; // evolution of theta
}
/*******************************************************/
double ARC(double theta_0,double R0)
{
  const int neqn=2;
  double t=0, *y, phi0, phiend, tau;
  y=dvector(1,neqn); y[1]=R0; y[2]=theta_0; phi0=theta_0-alpha*log(R0);

  P=Amp;    rk(y,neqn,t,tstep,SL,width_p);  // first rectangular pulse
  AutonomEvolution(y,gap*tstep);         // autonomous evolution between the pulses
  P=-Amp/Kfactor; rk(y,neqn,t,tstep,SL,width_n);  // 2nd rectangular pulse
  phiend=y[2]-alpha*log(y[1]); 

  tau=(pi2+phi0-phiend)/Om;  // time to reach the isochrone
  AutonomEvolution(y,tau);    // autonomous evolution to complete the cycle 
  return y[1];               // final amplitude
}
/*******************************************************/
int main () 
{
  const int npt=100;    //  number of points for ARC
  const double R0=1;    //  cycle amplitude  
  
  ofstream fout("ARC.dat",ios::out); // file for output
  fout.setf(ios::scientific);fout.precision(6);

  if((width_p+gap+width_n)*tstep>=pi2/Om)
    cout<<"Pulse too long"<<endl;
  else
    for(int i=1; i<=npt; i++)
        fout << pi2/npt*i << ' ' << ARC(pi2/npt*i,R0) << endl;
}
