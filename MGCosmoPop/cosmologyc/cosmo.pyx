# %%
import numpy as np 
from scipy.integrate import quad

C_LIGHT = 299792.458 # km/s


class Cosmology():
    def __init__(self):
        pass 


# %%
class fwCDM(Cosmology):
    def __init__(self, H0=70, Om0=0.3, w0=-1, wa=0, Xi0=1, n=0):
        self.H0  = H0
        self.Om0 = Om0
        self.w0  = w0
        self.wa  = wa
        self.Xi0 = Xi0
        self.n   = n

        Og0 = 4.48131e-7 * (2.7255)**4 / (H0/100)**2

        self.Or0  = Og0  +  Og0 * (3.04) * (7/8) * (4/11)**(4/3)
        self.Ode0 = 1 - self.Om0 - self.Or0
        self.dH   = C_LIGHT/H0

        self.par_names = ['H0', 'Om0', 'w0', 'wa', 'Xi0', 'n']
        self.par_vals  = [self.H0, self.Om0, self.w0, self.wa, self.Xi0, self.n]

    def __repr__(self):
        return "".join(["{:s}\t{:.2f}\n".format(n, v) for n,v in zip(self.par_names, self.par_vals)])


    def Xi(self, z):
        return self.Xi0 * (1-self.Xi0) * (1+z)**(-self.n)

    def E_inv(self,z):
        zp1    = z+1.
        exp_de = np.exp(-3.*self.wa*z/zp1) 
        return np.sqrt(self.Om0*zp1**3 + self.Or0*zp1**4 + self.Ode0*zp1**3.*(1.+self.w0+self.wa)*exp_de )**-1

    def dC(self, z):
        return quad(self.E_inv, 0., z)[0]

    def dL(self, z):
        return self.dH*(1+z)*self.dC(z)

    def dL_MG(self, z):
        return self.dL(z) if ((self.Xi0==1) | (self.n==0)) else self.dL(z)*self.Xi(z)

# %%








# %%





# ###############################################################
# ###############################################################  AbsFlatwCDM + ModGWProp
# ###############################################################


# 𝚵(z,𝚵0,n)  = 𝚵0 + (1-𝚵0)/(1+z)^n
# # 𝚵′(z,𝚵0,n) = n*(𝚵0-1)*(1+z)^(-n-1)


# Luminosity distance: standard (EM) -> dL, mod_prop (GW) -> dLGW
# function dL(c::AbsFlatwCDM, z::Vector{Float64}; kws...) 
#     Ωm0, Ωr0, Ωde0, w0, wa, dH = c.Ωm0, c.Ωr0, c.Ωde0, c.w0, c.wa, c.dH
#     E_inv(z) = 1 / sqrt( Ωm0*(1+z)^3 + Ωr0*(1+z)^4 + Ωde0*(1+z)^(3*(1+w0+wa))*exp(-3*wa*z/(1+z)) )    
#     dC(z)    = QuadGK.quadgk(z -> E_inv(z), 0., z; kws...)[1] # Mpc
#     [ dH * (1+zᵢ) * dC(zᵢ) for zᵢ in z ]
# end

# function dLGW(c::AbsFlatwCDM, z::Vector{Float64}; kws...) 
#     𝚵0, n = c.𝚵0, c.n
#     (𝚵0!=1) & (n!=0) ? dL(c,z; kws...) .* 𝚵.(z,𝚵0,n) : dL(c,z; kws...)
# end

# # # Jacobians of d(dL)/dz  &  d(dLGW)/dz
# # function ddL_dz(c::AbsFlatwCDM, z; kws...)
# #     Ωm0, Ωr0, Ωde0, w0, wa, dH = c.Ωm0, c.Ωr0, c.Ωde0, c.w0, c.wa, c.dH
# #     E_inv(z)  = 1 / sqrt( Ωm0*(1+z)^3 + Ωr0*(1+z)^4 + Ωde0*(1+z)^(3*(1+w0+wa))*exp(-3*wa*z/(1+z)) )    
# #     dC(z)     = QuadGK.quadgk(z -> E_inv(z), 0., z; kws...)[1] # Mpc    
# #     dH * (dC(z) + (1+z)*E_inv(z))
# # end




# # # Jacobian of comoving volume
# # function dVdz(c::AbsFlatwCDM, z::Vector{Float64}; kws...) 
# #     Ωm0, Ωr0, Ωde0, w0, wa, dH = c.Ωm0, c.Ωr0, c.Ωde0, c.w0, c.wa, c.dH
# #     E_inv(z) = 1 / sqrt( Ωm0*(1+z)^3 + Ωr0*(1+z)^4 + Ωde0*(1+z)^(3*(1+w0+wa))*exp(-3*wa*z/(1+z)) )    
# #     dC(z)    = QuadGK.quadgk(x -> E_inv(x), 0., z; kws...)[1]    
# #     dVdz(z)  = 4*π*dH^3 * dC(z)^2 * E_inv(z) # Mpc^3   (*10^-9 if #Gpc^3)
# #     [ dVdz(zᵢ) for zᵢ in z ]
# # end

# function E_inv(c::AbsFlatwCDM, z)::Float64
#     1. / sqrt( c.Ωm0*(1+z)^3 + c.Ωr0*(1+z)^4 + c.Ωde0*(1+z)^(3*(1+c.w0+c.wa))*exp(-3*c.wa*z/(1+z)) ) 
# end   

# function dC(c::AbsFlatwCDM, z; kws...)::Float64
#     QuadGK.quadgk(x -> E_inv(c, x), 0., z; kws...)[1] # Mpc
# end

# function dVdz(c::AbsFlatwCDM, z; kws...)::Float64
#     4*π*c.dH^3 * dC(c, z)^2 * E_inv(c, z) # Mpc^3   (*10^-9 if #Gpc^3)
# end

# function ddL_dz(c::AbsFlatwCDM, z; kws...)::Float64
#     c.dH * (dC(c, z) + (1+z)*E_inv(c, z)) 
# end 

# function ddL_dz(c::AbsFlatwCDM, z, dL; kws...)::Float64
#     dL/(1+z) + (1+z)*c.dH*E_inv(c, z)
# end



# #######################
# ## SOLVER FOR DISTANCE-REDSHIFT RELATION
# #######################
    

# function z_from_dLGW(c::AbstractCosmology)
#     # Returns redshift for a given luminosity distance r (in Mpc by default). 
#     z_grid  = vcat( 10 .^ range(-15, stop=log10(9.99e-9), length=10),
#                     10 .^ range(-8, stop=log10(7.99), length=1000),
#                     10 .^ range(log10(8), stop=5, length=100) )
#     dL_grid  = dLGW(c, z_grid)
#     f_interp = QuadraticSpline(z_grid,dL_grid)
#     return f_interp
# end




# end # module



    


 





# ################################################
# ################################################  TRASH
# ################################################









# # fde(c::AbsFlatwCDM, z)             = (1+z)^(3*(1+c.w0+c.wa)) * exp(-3*c.wa*z/(1+z))
# # E(c::AbsFlatCosmology, z)          = sqrt(c.Ωm0*(1+z)^3 + c.Ωr0*(1+z)^4 + c.Ωde0*fde(c,z))
# # # dL(c::AbsFlatCosmology, z; kws...) = c.dH * (1+z) * QuadGK.quadgk(z -> 1. / E(c,z), 0, z; kws...)[1]
# # Z(c::AbsFlatCosmology, z; kws...)  = QuadGK.quadgk(z -> 1. / E(c,z), 0, z; kws...)[1]
# # dL(c::AbsFlatCosmology, z; kws...) = c.dH * (1+z) * Z(c, z; kws...)



# # function log_dVdz(c::AbsFlatwCDM, z::Vector{Float64}; kws...) 
# #     Ωm0, Ωr0, Ωde0, w0, wa, dH = c.Ωm0, c.Ωr0, c.Ωde0, c.w0, c.wa, c.dH
# #     E(z)    = sqrt( Ωm0*(1+z)^3 + Ωr0*(1+z)^4 + Ωde0*(1+z)^(3*(1+w0+wa))*exp(-3*wa*z/(1+z)) )    
# #     dC(z)   = QuadGK.quadgk(z -> 1 / E(z), 0., z; kws...)[1]    
# #     dVdz(z) = log(4*π) + 3*log(dH) + 2*log(dC(z)) - log(E(z)) # Mpc^3   (-9*log(10) if #Gpc^3)
# #     [ dVdz(zᵢ) for zᵢ in z ]
# # end






# # function a2E(c::AbsFlatwCDM, a)
# #     ade = exp((1 - 3 * (c.w0 + c.wa)) * log(a) + 3 * c.wa * (a - 1))
# #     sqrt(c.Ω_r + c.Ω_m * a + c.Ω_Λ * ade)
# # end

# # scale_factor(z) = 1 / (1 + z)
# # E(c::AbstractCosmology, z) = (a = scale_factor(z); a2E(c, a) / a^2)
# # H(c::AbstractCosmology, z) = c.H0 * E(c, z)   # km/s/Mpc

# # Z(c::AbstractCosmology, z::Real; kws...) = 
# #     QuadGK.quadgk(a->1 / a2E(c, a), scale_factor(z), 1; kws...)[1]

# # comoving_radial_dist(c::AbstractCosmology, z; kws...) = 
# #     c_light / c.H0 * Z(c, z; kws...)

# # comoving_volume_element(c::AbstractCosmology, z; kws...) =
# #      c_light / c.H0 * (comoving_radial_dist(c, z; kws...) / (1 + z))^2 / a2E(c, scale_factor(z))


# # 𝚵(c::AbstractCosmology, z)      = (c.𝚵0!=0) & (c.n!=0) ? c.𝚵0+(1-c.𝚵0)/(1+z)^c.n : nothing
# # sPrime(c::AbstractCosmology, z) = 𝚵(c, z) - c.n*(1-c.𝚵0)/(1+z)^n

# # dL(c::AbstractCosmology, z; kws...) = comoving_radial_dist(c, z; kws...) * (1 + z)

# # dV_dz(c::AbstractCosmology, z) = 4*π*comoving_volume_element(c, z; kws...)
# # #         return 4*np.pi*FlatwCDM(H0=H0, Om0=Om, w0=w0).differential_comoving_volume(z).to(self.dist_unit**3/u.sr).value

# # log_dV_dz(c::AbstractCosmology, z; kws...) = 
# #     log(4*π) + 3*log(c_light) - 3*log(c.H0) + 2*log(comoving_radial_dist(c, z; kws...)) - log(E(c, z))


# # function log_ddL_dz(c::AbstractCosmology, z, dL=nothing; kws...)
# #     if (c.𝚵0!=0) & (c.n!=0) 
# #         if dL === nothing
# #             return log(c_light) - log(c.H0) + log(sPrime(c, z)*comoving_radial_dist(c, z; kws...)+
# #                    comoving_radial_dist(c, z; kws...) * (1 + z)/E(c, z))
# #         else 
# #             return log( dL/(1+z) * (1-(c.n*(1-c.𝚵0)) / 𝚵(c, z) * (1+z)^c.n) +
# #                         c_light*(1+z)*𝚵(c, z)/(c.H0*E(c, z)) )
# #         end
# #     else
# #         if dL === nothing
# #             return log(c_light) - log(c.H0) + log(comoving_radial_dist(c, z; kws...) + (1+z)/E(c, z))
# #         else 
# #             return log( dL/(1+z) + c_light * (1+z) / (c.H0*E(c, z)) )
# #         end
# #     end

# # end


# # function dL_GW(c::AbstractCosmology, z; kws...)
# #     # Modified GW luminosity distance in units set by self.dist_unit (default Mpc)   
# #     (c.𝚵0!=0) & (c.n!=0) ? dL(c, z; kws...)*𝚵(c, z) : dL(c, z; kws...)
# # end


# %%
