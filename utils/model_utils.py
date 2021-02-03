import numpy as np

j =1j
pi = np.pi

def model_simplified(f, cw, cm, ri, rp):
    return 2/(j*2*pi*f*cw) + 1/(1/rp+1/(ri+2/(j*2*pi*f*cm)))

def model_simplified_RI(f, cw, cm, ri, rp): # both real and imag part
    f_real = f
    f_imag = f
    Z_fit_real = np.real(model_simplified(f_real, cw, cm, ri, rp))  #[N,]
    Z_fit_imag = np.imag(model_simplified(f_imag, cw, cm, ri, rp))  #[N,]
    return np.concatenate((Z_fit_real, Z_fit_imag),axis=0)