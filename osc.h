

/*******************************************************/
// this function codes the ordinary differential equations for the coupled system
void SL(double t, double y[], double ydot[]);
void AutonomEvolution(double y[], double tau);
double *  Make_step(double *y, int width_p_, int gap_, int Kfactor_);
double Calc_x(double* y);
double Calc_y(double* y);
double * collect_state(double R, double theta, double* y);
// double get_phase(double *p);
double * init (int npt_, double R0_, double theta_0, int width_p_, int gap_, int Kfactor_ );