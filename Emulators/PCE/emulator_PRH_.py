import numpy as np
import scipy 
import numpoly
from scipy.signal import savgol_filter
import os

current_dir = os.getcwd()
     
class emu_cons2(object):
    r"""
    PCE for (log)-Bcase_smeared
    Attributes:
        parameters (list):
            model parameters, sorted in the desired order
        modes (numpy.ndarray):
            multipoles or k-values in the (log)-spectra
        n_pcas (int):
            number of PCA components
        parameters_filenames (list [str]):
            list of .npz filenames for parameters
        features_filenames (list [str]):
            list of .npz filenames for (log)-spectra
        verbose (bool):
            whether to print messages at intermediate steps or not
    """
        
 
    
    
    def __init__(self):
        r"""
        Constructor
        """ 
        # attributes
        # attributes
        self.ks_cola = np.loadtxt(f'{current_dir}/projects/lsst_y1/Emulators/PCE/ks_cola_d2.txt')[0:255] 
        
        self.redshift_default_ =[ 0.   ,  0.02 ,  0.041,  0.062,  0.085,  0.109,  0.133,  0.159,
        0.186,  0.214,  0.244,  0.275,  0.308,  0.342,  0.378,  0.417,
        0.457,  0.5  ,  0.543,  0.588,  0.636,  0.688,  0.742,  0.8  ,
        0.862,  0.929,  1.   ,  1.087,  1.182,  1.286,  1.4  ,  1.526,
        1.667,  1.824,  2.   ,  2.158,  2.333,  2.529,  2.75 ,  3.  ] #([0    , 0.020, 0.041, 0.062, 0.085, 0.109, 0.133, 0.159, 0.186, 0.214, 0.244, 0.275, 0.308, 0.342, 0.378, 0.417, 0.457,
                                  #0.500, 0.543, 0.588, 0.636, 0.688, 0.742, 0.800, 0.862, 0.929,   1.0, 1.087, 1.182, 1.286, 1.400, 1.526, 1.667, 1.824, 
                                   # 2.0, 2.158,2.333,2.529,2.750,3.0])
        self.lhs_ = np.loadtxt(f"{current_dir}/projects/lsst_y1/Emulators/PCE/lhs.txt")    
        self.polys={}
        for i in self.redshift_default_:
            self.polys[i] = numpoly.loadtxt(f"{current_dir}/projects/lsst_y1/Emulators/PCE/emu_z" + str(i) +"_colapc15_nsmear_all_1.0%.txt")[0:255]
            #print(i)#"_colateste.txt")#"_hpr.txt")
    
    
        

    def scaler_trans(self, params_):
        X_std = (params_ - self.lhs_.min(axis=0)) / (self.lhs_.max(axis=0) - self.lhs_.min(axis=0))
        max_=1
        min_= -1 

        return X_std * (max_ - min_) + min_
 
    def scaler_invtrans(self, scaled_params_):
        max_=1
        min_= -1 
        scale_ =  (max_ - min_) / (self.lhs_.max(axis=0) - self.lhs_.min(axis=0))
        min_par = min_ - self.lhs_.min(axis=0) *  scale_
        return   (scaled_params_ - min_par)/scale_

    def dict_to_ordered_arr_np(self, input_dict):
        r"""
        Sort input parameters
        Parameters:
            input_dict (dict [numpy.ndarray]):
                input dict of (arrays of) parameters to be sorted
        Returns:
            numpy.ndarray:
                input parameters sorted according to `parameters`
        """
        return np.stack([input_dict[k] for k in input_dict]).T     
    
    
    def smoothpl_boost(self,xx, yy, window_length, degree   ):

        ynew = np.zeros(len(yy))


        # Take care that the window does not spill out of our array
        margin = int(window_length/2)
        xarr = np.arange(-margin, margin+1)/margin

        _X_poly = np.array([np.power(xarr, p) for p in range(degree + 1)]).T # creating the fitting poly
        _X_mat = np.mat(_X_poly)
        weights = np.mat(np.eye((len(_X_mat)))) 

        i = margin
    #fitting the central windows
        for i in range(margin, yy.size-margin):


            _y_mat = np.mat(yy[i-margin:i+margin+1])

            peso =  np.cosh(np.linspace(-9, 9, len(yy)))[i-margin:i+margin+1]  # usei pesos para cada pequena janela, poderia ter usado uma porcentagem do proprio pk
            for j in range(len(xarr)):

                weights[j, j] =peso[j]


            W = np.linalg.pinv(_X_mat.T @ (weights @ _X_mat)) @ (_X_mat.T @ weights @ _y_mat.T)
            ynew[i] = np.asarray(_X_mat @ W).reshape(-1)[margin]


     #fitting the left edge
        _y_mat = np.mat(yy[:window_length])

        peso =  np.cosh(np.linspace(-9, 9, len(yy)))[:window_length]

        for j in range(len(xarr)):

            weights[j, j] =peso[j]        

        W = np.linalg.pinv(_X_mat.T @ (weights @ _X_mat)) @ (_X_mat.T @ weights @ _y_mat.T)    


        for i in range(margin):

            ynew[i] =  np.asarray(_X_mat @ W).reshape(-1)[i]




     #fitting the right edge 
        _y_mat = np.mat(yy[-window_length:])

        peso =  np.cosh(np.linspace(-9, 9, len(yy)))[-window_length:]

        for j in range(len(xarr)):

            weights[j, j] =peso[j]        

        W = np.linalg.pinv(_X_mat.T @ (weights @ _X_mat)) @ (_X_mat.T @ weights @ _y_mat.T)             


        for i in range(margin):
            ynew[yy.size-margin+i] =  np.asarray(_X_mat @ W).reshape(-1)[i+margin+1]



        return ynew


    def get_boost_custom_(self, boost_, kvals, custom_kvec=None, ext_=None   ):
#Extrapolate for custom k-range
        
        if ext_ is None:
            ext_ = 'logk-logB'
            
            
        k_shape = kvals.shape
        do_extrapolate_above = False
        do_extrapolate_below = False   
        if not(custom_kvec is None):
            upper_mask = custom_kvec < max(kvals)
            lower_mask = custom_kvec > min(kvals)
            mask = [u and l for (u,l) in zip(lower_mask, upper_mask)]
            custom_k_within_range = custom_kvec[mask]
            custom_k_below = custom_kvec[[not(l) for l in lower_mask]]
            custom_k_above = custom_kvec[[not(u) for u in upper_mask]]

            if any(custom_kvec > max(kvals)):
                wrn_message = ("Higher k modes constantly extrapolated.")

               # print(wrn_message)
                do_extrapolate_above = True

            if any(custom_kvec < min(kvals)):
                wrn_message = ("Lower k modes constantly extrapolated.")

              #  print(wrn_message)
                do_extrapolate_below = True  

        len_kvals = len(kvals)
        len_redshifts = 1

        logboost =   np.log10(boost_)

        bvals = {}
        i = 0 
        tmp = logboost
        if not(custom_kvec is None):
            bvals[i] = 10.0**scipy.interpolate.CubicSpline(np.log10(kvals),
                                          tmp.reshape(k_shape)
                                          )(np.log10(custom_k_within_range))

            #Extrapolate if necessary
            if do_extrapolate_below:
                # below the k_min of EuclidEmulator2, we are in the linear regime where
                # the boost factor is unity by construction
               # print('do_extrapolate_below')
                b_extrap = np.ones_like(custom_k_below)
                bvals[i]= np.concatenate((b_extrap, bvals[i]))
                

            if do_extrapolate_above:
                # We extrapolate by setting all b(k > k_max) to b(k_max)
                 


#np.concatenate((logboost[:255 -25 ], savgol_filter(logboost, 23, 1  )[255-25: ])), 
 # np.concatenate((logboost[:255 -25 ], self.smoothpl_boost(np.log10(kvals),logboost, 23, 1   )[255-25: ])),


                if ext_ == 'logk-logB':#(logboost[:255 -25 ], self.smoothpl_boost(np.log10(kvals),logboost, 19, 1   )[255-25: ])),
                    inter_boost_2__= scipy.interpolate.interp1d(np.log10(kvals), #255 -25 5 1 ou 255 -25(pode ser -15) 7 1 ou 255 -25(pode ser -15) 9 1 ou ou 255 -25 13 1 ou 255 -25 19 1
                                                                np.concatenate((logboost[:255 -25 ], savgol_filter(logboost, 23, 1  )[255-25: ])), 
                                                                kind='linear',
                                                                fill_value='extrapolate',
                                                                assume_sorted=True)
                   # expansion = chaospy.monomial(start=0, stop=2, dimensions=1)

                   # fitted_polynomial = chaospy.fit_regression(
                       # expansion, np.log10(kvals)[210:],  (logboost)[210:], model = lm.Lasso(alpha=0.000022, fit_intercept=False))




                    bvals[i] = np.concatenate((bvals[i], 10**inter_boost_2__(np.log10(custom_k_above))     ))# np.concatenate((bvals[i], 10**inter_boost_2__(np.log10(custom_k_above))     ))
                elif ext_ == 'logk-B':    
                    inter_boost_2__= scipy.interpolate.interp1d(np.log10(kvals),
                                                                np.concatenate((10**logboost[:250 -5 ], self.smoothpl_boost(np.log10(kvals),10**logboost, 5, 1   )[250-5: ])),
                                                                kind='linear',
                                                                fill_value='extrapolate',
                                                                assume_sorted=True)
                   # expansion = chaospy.monomial(start=0, stop=2, dimensions=1)

                   # fitted_polynomial = chaospy.fit_regression(
                       # expansion, np.log10(kvals)[210:],  (logboost)[210:], model = lm.Lasso(alpha=0.000022, fit_intercept=False))



                    
                    bvals[i] = np.concatenate((bvals[i], inter_boost_2__(np.log10(custom_k_above))     ))# np.concatenate((bvals[i], 10**inter_boost_2__(np.log10(custom_k_above))     ))                   
                    
                elif ext_ == 'euclid':    
                    b_extrap = bvals[i][-1] * np.ones_like(custom_k_above)
                    bvals[i] = np.concatenate((bvals[i], b_extrap))


                    
                    #bvals[i] = np.concatenate((bvals[i], inter_boost_2__(np.log10(custom_k_above))     ))# np.concatenate((bvals[i], 10**inter_boost_2__(np.log10(custom_k_above))     ))                        
                    
                    
                    
                    

        else:
            bvals[i] = 10.**tmp.reshape(k_shape)
                
                
                

        if not(custom_kvec is None):       # This could probably be done cleaner!
            kvals = custom_kvec
         
        return kvals,bvals















    

    def get_boost(self, cosmo_dict, redshifts, custom_kvec=None, ext_ = None, intp_ = None  ):
               
        if ext_ is None:
            ext_ = 'logk-logB'
            
        if isinstance(redshifts, (int, float)):
            redshifts = np.asarray([redshifts])
        else:
            redshifts = np.asarray(redshifts)

        for z in redshifts:
            assert z <= 3.0 and z>=0.0, "BernardoEmulator allows only redshifts in the interval [0.0, 3.0]"                 
        
                
         
        boost = {}        
        if self.dict_to_ordered_arr_np(cosmo_dict).ndim == 1:

            ##print(self.dict_to_ordered_arr_np(cosmo_dict).ndim, 'um só')
            this_cosmo_par_in = self.dict_to_ordered_arr_np(cosmo_dict) 
            cosmo_par_in = self.dict_to_ordered_arr_np(cosmo_dict).reshape(1,-1)

 
            for i in range(redshifts.shape[0]):
                for j in range(len( self.redshift_default_)): 
                    if redshifts[i] == self.redshift_default_[j]: 
                        #print(redshifts[i], self.redshift_default_[j], '==' )
                        if not(custom_kvec is None):
                         
                            boost[redshifts[i]] = self.get_boost_custom_(np.exp(self.polys[redshifts[i]](*self.scaler_trans( cosmo_par_in).T)).T[0],
                                                                         self.ks_cola,
                                                                         custom_kvec=custom_kvec, ext_=ext_)[1][0]   # return kvals,bvals, [1] stands for bvals 
                        else:
                            boost[redshifts[i]] =  np.exp(self.polys[redshifts[i]](*self.scaler_trans( cosmo_par_in).T)).T[0]     
                        
                        
                        
                    elif redshifts[i] > self.redshift_default_[j] and redshifts[i] < self.redshift_default_[j+1]:
                        if not(custom_kvec is None):
                           # print(redshifts[i], self.redshift_default_[j], '><')
                            x_par = np.array([self.redshift_default_[j],self.redshift_default_[j+1]])
                            this_left_ =np.exp(self.polys[x_par[0]](*self.scaler_trans( cosmo_par_in).T)).T   
                            this_right_=  np.exp(self.polys[x_par[1]](*self.scaler_trans( cosmo_par_in).T)).T  
                            this_ynew_par = []

                            for l in range(this_cosmo_par_in.ndim):

                                left_ = this_left_[l]  
                                right_= this_right_[l] 

                                ynew_par=[]
                                
                                if intp_  is None: 
                                  
                                    for k in range(self.ks_cola.shape[0]):
    #
                                        f_par = scipy.interpolate.interp1d(x_par,
                                                                     1/np.array([left_[k],
                                                                                 right_[k]]))
        
        
        
        
                                        ynew_par.append( 1.0/f_par(redshifts[i]))

        
        
        
                                else: 
                                 
                                   # f=  scipy.interpolate.interp2d(np.log10(self.ks_cola),
                                    #                               x_par,
                                    #                               np.log10(np.array([left_,right_])) )   
                        
                        
                                    f = scipy.interpolate.RectBivariateSpline(x_par,
                                                                              np.log10(self.ks_cola),
                                                                              np.log10(np.array([left_,right_])),
                                                                              kx=1,
                                                                              ky=1 )


                                  #  ynew_par = ( 10**(f(np.log10(self.ks_cola) , redshifts[i]) ))
                                    ynew_par = ( 10**(f.ev(redshifts[i], np.log10(self.ks_cola))))
                                    
                                    
                                                            
                                ynew_par= self.get_boost_custom_(np.array(ynew_par),
                                                                         self.ks_cola,
                                                                         custom_kvec=custom_kvec, ext_=ext_)[1][0] 
                                this_ynew_par.append(ynew_par)
                                    
                        
                        
                        
                        
                        else:
                           # print(redshifts[i], self.redshift_default_[j], '><')
                            x_par = np.array([self.redshift_default_[j],self.redshift_default_[j+1]])
                            this_left_ =np.exp(self.polys[x_par[0]](*self.scaler_trans( cosmo_par_in).T)).T   
                            this_right_=  np.exp(self.polys[x_par[1]](*self.scaler_trans( cosmo_par_in).T)).T  
                            this_ynew_par = []

                            for l in range(this_cosmo_par_in.ndim):

                                left_ = this_left_[l]  
                                right_= this_right_[l] 

                                ynew_par=[]
                                
                                if intp_ is None: 
                                   
                                    for k in range(self.ks_cola.shape[0]):

                                        f_par = scipy.interpolate.interp1d(x_par,
                                                                     1/np.array([left_[k],
                                                                                 right_[k]]))
                                        ynew_par.append( 1.0/f_par(redshifts[i]))

                                
                                else:
                                                               
                                   # f=  scipy.interpolate.interp2d(np.log10(self.ks_cola),
                                   # x_par,
                                    #np.log10(np.array([left_,right_])) )   

                                   # ynew_par= ( 10**(f(np.log10(self.ks_cola), redshifts[i]) ) )    
                        
                                    f = scipy.interpolate.RectBivariateSpline(x_par,
                                                                              np.log10(self.ks_cola),
                                                                              np.log10(np.array([left_,right_])),
                                                                              kx=1,
                                                                              ky=1 )


                                    #  ynew_par = ( 10**(f(np.log10(self.ks_cola) , redshifts[i]) ))
                                    ynew_par = ( 10**(f.ev(redshifts[i], np.log10(self.ks_cola))))

                                    
                                    
                                    
                                    
                                    
   
                                this_ynew_par.append(ynew_par)    

                        boost[redshifts[i]] = np.array(this_ynew_par)[0]
             
        else:
        
            cosmo_par_in = self.dict_to_ordered_arr_np(cosmo_dict)   
            for i in range(redshifts.shape[0]):
                for j in range(len( self.redshift_default_)): 
                    if redshifts[i] == self.redshift_default_[j]: 
                      #  print(redshifts[i], self.redshift_default_[j], '==')
                        this_boosts= [ ] 
                        if not(custom_kvec is None):
                           
                            boosts_=np.exp(self.polys[redshifts[i]](*self.scaler_trans( cosmo_par_in).T)).T
                            for k in range(cosmo_par_in.shape[0]):
                        
                                this_boosts.append( self.get_boost_custom_(boosts_[k] ,
                                                                         self.ks_cola,
                                                                         custom_kvec=custom_kvec, ext_=ext_)[1][0])
                                           
                            boost[redshifts[i]] =  np.array(this_boosts)               
                                           
                        else:
                            boost[redshifts[i]] =  np.exp(self.polys[redshifts[i]](*self.scaler_trans( cosmo_par_in).T)).T     
                        
                        
                        
                    
                    
                    
                    
                    elif redshifts[i] > self.redshift_default_[j] and redshifts[i] < self.redshift_default_[j+1]:
                       # print(redshifts[i], self.redshift_default_[j], '><')
                        x_par = np.array([self.redshift_default_[j],self.redshift_default_[j+1]])
                        this_left_ =np.exp(self.polys[x_par[0]](*self.scaler_trans( cosmo_par_in).T)).T   
                        this_right_=  np.exp(self.polys[x_par[1]](*self.scaler_trans( cosmo_par_in).T)).T  
                        this_ynew_par = []

                       
                        if not(custom_kvec is None):
                            for l in range(cosmo_par_in.shape[0]):

                                left_ = this_left_[l]  
                                right_= this_right_[l] 

                                ynew_par=[]
                                if intp_  is None: 
                                   
                                    for k in range(self.ks_cola.shape[0]):

                                        f_par = scipy.interpolate.interp1d(x_par,
                                                                     1/np.array([left_[k],
                                                                                 right_[k]]))
                                        ynew_par.append( 1.0/f_par(redshifts[i]))
                                 
                                
                                else:
                                   
                                    # f=  scipy.interpolate.interp2d(np.log10(self.ks_cola),
                                    #                               x_par,
                                    #                               np.log10(np.array([left_,right_])) )   


                                    f = scipy.interpolate.RectBivariateSpline(x_par,
                                                                              np.log10(self.ks_cola),
                                                                              np.log10(np.array([left_,right_])),
                                                                              kx=1,
                                                                              ky=1 )


                                    #  ynew_par = ( 10**(f(np.log10(self.ks_cola) , redshifts[i]) ))
                                    ynew_par = ( 10**(f.ev(redshifts[i], np.log10(self.ks_cola))))

                                         
                                    
                                
                                ynew_par= self.get_boost_custom_(np.array(ynew_par),
                                         self.ks_cola,
                                         custom_kvec=custom_kvec, ext_=ext_)[1][0] 
                                    
                                this_ynew_par.append(ynew_par)                             
                    
                    
                    
                        else:
                            for l in range(cosmo_par_in.shape[0]):

                                left_ = this_left_[l]  
                                right_= this_right_[l] 

                                ynew_par=[]
                                
                                
                                                                
                                if intp_ is None: 
                              
                                    for k in range(self.ks_cola.shape[0]):

                                        f_par = scipy.interpolate.interp1d(x_par,
                                                                     1/np.array([left_[k],
                                                                                 right_[k]]))
                                        ynew_par.append( 1.0/f_par(redshifts[i]))

                                else:     
                                
                                    # f=  scipy.interpolate.interp2d(np.log10(self.ks_cola),
                                    #                               x_par,
                                    #                               np.log10(np.array([left_,right_])) )   


                                    f = scipy.interpolate.RectBivariateSpline(x_par,
                                                                              np.log10(self.ks_cola),
                                                                              np.log10(np.array([left_,right_])),
                                                                              kx=1,
                                                                              ky=1 )


                                    #  ynew_par = ( 10**(f(np.log10(self.ks_cola) , redshifts[i]) ))
                                    ynew_par = ( 10**(f.ev(redshifts[i], np.log10(self.ks_cola))))
                                    
                                    
                                    
                                    
                                this_ynew_par.append(ynew_par)    

                        boost[redshifts[i]] = np.array(this_ynew_par)

                    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
        if not(custom_kvec is None):       # This could probably be done cleaner!
            kvals = custom_kvec
        else:
            kvals = self.ks_cola
       
                
                    
            
        return  kvals, boost
    
    

    def redshift_default(self):

        return self.redshift_default_
        
        
        
        
    def smoothpl_boost_(self, xx, yy, window_length, degree   ):

        ynew = np.zeros(len(yy))


        # Take care that the window does not spill out of our array
        margin = int(window_length/2)
        xarr = np.arange(-margin, margin+1)/margin

        _X_poly = np.array([np.power(xarr, p) for p in range(degree + 1)]).T # creating the fitting poly
        _X_mat = np.mat(_X_poly)
        weights = np.mat(np.eye((len(_X_mat)))) 

        i = margin
    #fitting the central windows
        for i in range(margin, yy.size-margin):


            _y_mat = np.mat(yy[i-margin:i+margin+1])

            peso =  np.cosh(np.linspace(-4.5, 4.5, len(yy)))[i-margin:i+margin+1]  # usei pesos para cada pequena janela, poderia ter usado uma porcentagem do proprio pk
            for j in range(len(xarr)):

                weights[j, j] =peso[j]


            W = np.linalg.pinv(_X_mat.T @ (weights @ _X_mat)) @ (_X_mat.T @ weights @ _y_mat.T)
            ynew[i] = np.asarray(_X_mat @ W).reshape(-1)[margin]


     #fitting the left edge
        _y_mat = np.mat(yy[:window_length])

        peso =  np.cosh(np.linspace(-4.5, 4.5, len(yy)))[:window_length]

        for j in range(len(xarr)):

            weights[j, j] =peso[j]        

        W = np.linalg.pinv(_X_mat.T @ (weights @ _X_mat)) @ (_X_mat.T @ weights @ _y_mat.T)    


        for i in range(margin):

            ynew[i] =  np.asarray(_X_mat @ W).reshape(-1)[i]




     #fitting the right edge 
        _y_mat = np.mat(yy[-window_length:])

        peso =  np.cosh(np.linspace(-4.5, 4.5, len(yy)))[-window_length:]

        for j in range(len(xarr)):

            weights[j, j] =peso[j]        

        W = np.linalg.pinv(_X_mat.T @ (weights @ _X_mat)) @ (_X_mat.T @ weights @ _y_mat.T)             


        for i in range(margin):
            ynew[yy.size-margin+i] =  np.asarray(_X_mat @ W).reshape(-1)[i+margin+1]



        return ynew        
        
        
        
        
        
    def smoothpk(self, k, y,param,  p_n_w=None ):    


    #applying the filter successively   

        sg_err= self.smoothpl_boost_(np.log(k), np.log(y), 25, 5 )
        sg_err= self.smoothpl_boost_(np.log(k), sg_err, 25, 5 )  
        sg_err= self.smoothpl_boost_(np.log(k), sg_err, 25, 5 )  

        sg_err=np.exp(sg_err) 

    # peguei o primeiro maximo é o primeiro minimo antes desse maximo, e usando esse ponto como pivo 
    #fiz uma spline para poder controlar o pk_nw evolui até o primeiro maximo, estava buscando o mesmo comportamento que o metodo do baumann


        pnw2 = sg_err

    #seguindo o procedimento do bacco    

        integral = scipy.integrate.simpson(y, x=k)
        k_star_inv = (1.0/(3.0 * np.pi**2)) * integral
        Gk = np.array([np.exp(-param*k_star_inv * (k_**2)) for k_ in k])
        pk_smeared = y*Gk + pnw2*(1.0 - Gk)

        if p_n_w is None: 
            return pk_smeared 
        else:
            return pk_smeared ,pnw2
        
     
    
    
