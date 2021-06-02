/*******  amplitude response curve for the SL oscillators   ******/
/*******            charge-balanced pulses                  ******/

#include <iostream> 
#include <fstream> 
#include <math.h>
#include "my_defs.h"
#include "nrlib.h"
#include <vector>
using namespace std;



/*************  important parameters  **************/
const double omega=1, alpha=0, Om=omega-alpha;   // systems's parameters
const double Amp=0.9;      //  kick amplitude
int width_p=10;       //  width of the 1st pulse in steps
int gap=10;        //   gap in steps 
int Kfactor=3;    //   ratio of the 2nd width over 1st width
int width_n=width_p*Kfactor; // width of the 2nd pulse in steps
double P;                  // current rectangular pulse' amplitude
int npt=100 ;    //  number of points for ARC 100
double R0=1;
double theta_0=0;
const double tstep=0.001;  
const int neqn=2;
/*******************************************************/

void SL(double t, double y[], double ydot[])
{  
//   ofstream foutt;
//   foutt.open("file.txt", ios::app);
  double R2;  // equations in polar coordinates
  R2=SQR(y[1]);
  ydot[1]= y[1]*(1-R2) + P*cos(y[2]); 
  ydot[2]=omega-alpha*R2 -P*sin(y[2])/y[1];
//   foutt << y[1];
//   foutt << "\n";
//   foutt << y[2];
//   foutt << "\n";  
//   foutt.close();
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
double *  Make_step(double *y, int width_p_, int gap_, int Kfactor_)
{
//   ofstream fout;
//   fout.open("file.txt", ios::app);  
//   fout << "start\n";
//   fout << y[1];
//   fout << "\n"; 
//   fout << y[2];  
//   fout << "\n";
  width_p = width_p_; gap = gap_; Kfactor = Kfactor_; //vector of our actions
  double t=0, phi0, phiend, tau;
  phi0=y[2]-alpha*log(y[1]);
  width_n = width_p*Kfactor; 
//   fout.close();
  P=Amp;    rk(y,neqn,t,tstep,SL,width_p);  // first rectangular pulse
//   fout.open("file.txt", ios::app);  
//   fout << y[1];
//   fout << "\n";
//   fout << y[2];  
//   fout << "\n";  
  AutonomEvolution(y,gap*tstep); // autonomous evolution between the pulses
//   for (int count = 0; count <= gap; count+=1)
//   {
//       fout << y[1];
//       fout << "\n";
//       fout << y[2];
//       fout << "\n"; 
//   }
//   fout << "aftergap\n";  
//   fout << gap;
//   fout << "\n";
//   fout << y[1];
//   fout << "\n";
//   fout << y[2];
//   fout << "\n"; 
//   fout.close();
  P=-Amp/Kfactor; rk(y,neqn,t,tstep,SL,width_n);  // 2nd rectangular pulse
  phiend=y[2]-alpha*log(y[1]); 
//   fout.open("file.txt", ios::app);   
//   fout << y[1];
//   fout << "\n";
//   fout << y[2];
//   fout << "\n";
  tau=(pi2+phi0-phiend)/Om;  // time to reach the isochrone
//   cout << tau << endl;
  AutonomEvolution(y,tau);    // autonomous evolution to complete the cycle
//   for (int count = 0; count <=tau; count+=tstep)
//   {
//       fout << y[1];
//       fout << "\n";
//       fout << y[2];
//       fout << "\n"; 
//   }
//   fout << "aftertau\n";  
//   fout << tau;
//   fout << "\n";  
//   fout << y[1];
//   fout << "\n";
//   fout << y[2];  
//   fout << "\n";
//   fout << "finish\n";  
//   fout.close();
  return y;               // final solution y;   amplitude y[1]
}

//Calculate mean_field x
double Calc_x(double* y)
{  
  return y[1];
}

double Calc_y(double* y)
{  
  return y[2];
}

double * collect_state(double R, double theta, double* y)
{
  y[1] = R; y[2] = theta;
  return y;
}
//Init params
double * init (int npt_, double R0_, double theta_0_, int width_p_, int gap_, int Kfactor_ ) 
{
  npt = npt_;    //  number of points for ARC 100
  R0 = R0_;    //  cycle amplitude  1
  theta_0 = theta_0_;
  width_p = width_p_;       //  width of the 1st pulse in steps
  gap = gap_;        //   gap in steps 
  Kfactor = Kfactor_;
  width_n = width_p * Kfactor;
  double *y;
  y = dvector(1, 2);
  y[1] = R0_; y[2] = theta_0;
  return y;
}
