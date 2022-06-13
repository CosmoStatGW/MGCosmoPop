
double dC (const double z) const 
{
  double dC;
  dC = GSL_integrate_qag(EE_inv, 0, z); 
  return m_D_H*Dc;
}
