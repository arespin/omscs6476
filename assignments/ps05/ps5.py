"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        '''
        Complete the prediction state in predict(self). Here you will replace the class variables for the
        state and covariance arrays with the prediction process.
        '''
        # State vector 
        self.state = np.array([init_x, init_y, 0., 0.]).T  # state
        
        self.I     = np.diag([1.0]*4)
        
        self.dt    = 0.1
        self.Dt        = np.diag([1.0]*4)
        self.Dt[:2,2:] = np.diag([self.dt]*2)
        
        self.Mt    = np.matrix(np.diag([1.0]*4)[:2])

        # Noise matrix dt, mt
        self.Q, self.R = Q, R
        
        self.P     =  np.diag([1000.0]*4)
        
        # raise NotImplementedError

    def predict(self):
        '''
        Complete the prediction state in predict(self). Here you will replace the class variables for the
        state and covariance arrays with the prediction process.
        '''        
        self.state = self.Dt * self.state
        self.P     = self.Dt * self.P * self.Dt.transpose() + self.Q
        
        # raise NotImplementedError

    def correct(self, meas_x, meas_y):
        '''
        Finally, we need to correct the state and the covariances using the Kalman gain and the
        measurements obtained from our sensor.
        '''
        z          = np.matrix([meas_x, meas_y]).transpose()
        S          = self.Mt * self.P * self.Mt.transpose() + self.R
        K          = self.P * self.Mt.transpose() * np.linalg.inv(S)
        y          = z - (self.Mt * self.state)
        self.state = self.state + (K * y)
        self.P     = (self.I - (K * self.Mt)) * self.P
        #raise NotImplementedError

    def process(self, measurement_x, measurement_y):
        
        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0,0], self.state[1,0]
    
    

class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        
        
        # Convert to BW
        self.img_weight= [0.3, 0.58, 0.12]
        #self.template = self.normalize(np.dot(template.copy(),self.img_weight))* 255.0
        self.template = cv2.cvtColor(template.copy(), cv2.COLOR_BGR2GRAY)
        # Round to even dimensions
        h, w = self.template.shape[:2]
        self.template = self.template[:int(np.floor(h/2)*2), :int(np.floor(w/2)*2)]
        self.template_hist = [self.template]
        self.mse_hist = []
        self.hist_len = 3
        
        
        self.frame = frame.copy()
        self.particles = None  # Initialize during processing
        self.weights = np.ones(self.num_particles)/(1.*self.num_particles)
        # Initialize any other components you may need when designing your filter.
        # Make a state using center position
        self.state = [self.template_rect[i] + np.floor(self.template_rect[j] / 2.) for (i,j) in [('x', 'w'), ('y', 'h')]]
        #raise NotImplementedError

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        # Use mse of template vs frame
        return np.sum((template.astype(float) - frame_cutout.astype(float))**2)/(np.prod(template.shape[:2]))
    
    def normalize(self, val):
        min_val= np.amin(val)
        val= np.subtract(val,min_val)
        max_val= np.amax(val)
        val= np.divide(val,max_val)
        return val

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        return np.random.choice(a=self.num_particles, size=self.num_particles, p=self.weights, replace = True)


    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        h, w = self.state # Initial state for frame
        gray_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        #gray_frame = self.normalize(np.dot(frame.copy(),self.img_weight))* 255.0

        # Random normal distribution of initial state
        rand_h = np.expand_dims(np.random.normal(h, self.sigma_dyn, self.num_particles), 1)
        rand_w = np.expand_dims(np.random.normal(w, self.sigma_dyn, self.num_particles), 1)
        self.particles = np.concatenate((rand_h, rand_w), axis=1)
        
        errorls = []
        for t, p in enumerate(self.particles):
            error = self.get_error_metric(self.template, self.get_cutout(gray_frame, p))
            self.weights[t] = np.exp(-error / (2. * self.sigma_exp ** 2.))
            errorls.append(error)
        self.mse_hist.append(int(np.mean(errorls)))
        self.mse_hist = self.mse_hist[-self.hist_len:]
        #print self.mse_hist
        
        
        self.weights[self.particles[:, 0] != np.clip(self.particles[:, 0], 0, frame.shape[1])]=0
        self.weights[self.particles[:, 1] != np.clip(self.particles[:, 1], 0, frame.shape[0])]=0

        self.weights   /= sum(self.weights)
        self.particles = self.particles[self.resample_particles()]
        self.particles_hist = [self.particles]
        #print self.particles.shape
        self.weights_hist   = [self.weights]
        self.state     = np.average(self.particles, 0, self.weights).astype(np.int)
        #print self.particles

        #self.state[0] = np.clip(self.state[0], 0, frame.shape[1])
        #self.state[1] = np.clip(self.state[1], 0, frame.shape[0])
        
    def get_cutout(self, frame, center):
        '''
        Cutout template frame
        '''
        # Get Shapes
        f_h, f_w        = frame.shape[:2]
        c_w, c_h        = center.astype(int)
        t_h, t_w        = self.template.shape[:2]
        
        # Get frame positions
        fx_from, fx_to  = c_w - (t_w/2), c_w + (t_w/2)
        fy_from, fy_to  = c_h - (t_h/2), c_h + (t_h/2)
        
        # If they fall out of range, bring it back in the picture
        if fx_from<1:
            fx_from, fx_to = 1, 1 + fx_to-fx_from
            
        if fy_from<1:
            fy_from, fy_to = 1, 1 + fy_to-fy_from
            
        if fx_to > f_w-1:
            fx_from, fx_to = fx_from-(fx_to-f_w)-1, f_w-1
            
        if fy_to > f_h-1:
            fy_from, fy_to = fy_from-(fy_to-f_h)-1, f_h-1
        
        return frame[fy_from:fy_to, fx_from:fx_to]


    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """
        '''
        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.
        raise NotImplementedError
        '''
        
        w_s, h_s = self.state
        h, w = self.template.shape[:2]

        # draw rectangle
        cv2.rectangle(frame_in, (w_s - w/2, h_s - h/2), (w_s + w/2, h_s + h/2), (255, 255, 255), 2)
        
        # draw particles
        for px, py in self.particles.astype(np.int64):
            cv2.circle(frame_in, (px, py), 1, (0, 0, 255), -1)
            
        # draw circle
        dist   = np.sqrt(((self.particles - self.state) ** 2).sum(axis=1))
        radius = np.average(dist, axis=0, weights=self.weights).astype(np.int)
        cv2.circle(frame_in, tuple(self.state), radius, (255, 255, 255), 1)




class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.template_orig = self.template

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        
        super(AppearanceModelPF, self).process(frame)
        
        #print self.state, self.frame.shape, self.template.shape
        
        gray_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        #gray_frame = self.normalize(np.dot(frame.copy(),self.img_weight))* 255.0

        #print dir(self)# print self.weights
        
        best   = self.get_cutout(gray_frame, self.state)
        alpha  = self.alpha
        
        #temp_wts = np.array(self.mse_hist, dtype = np.float) / sum(np.array(self.mse_hist) )
        #weighted_template  = sum([wt*temp for (temp, wt) in zip(self.template_hist, temp_wts)])
        #temp_t = alpha*best + (1-alpha)*(weighted_template)
        #print weighted_template.mean(), self.template.mean(), temp_wts
        '''
        print self.particles
        print 50*'--'
        print self.weights
        print 50*'-*-'
        '''
        
        temp_t = alpha*best + (1-alpha)*self.template

        self.template = temp_t.astype(np.uint8)
        
        particles_all = np.concatenate((self.particles, self.particles_hist[-1]), axis=0)
        weights_alpha = np.concatenate((self.weights*(alpha), self.weights_hist[-1]*(1-alpha)), axis=0)
        
        idx = np.random.choice(range(particles_all.shape[0]), size=self.num_particles, p=weights_alpha, replace = True)  
        
        self.particles = particles_all[idx]
        self.weights   = weights_alpha[idx]
        self.weights   /= sum(self.weights)
        self.state = np.average(self.particles, 0, self.weights).astype(np.int)
        self.state[0] = np.clip(self.state[0], 0, frame.shape[1])
        self.state[1] = np.clip(self.state[1], 0, frame.shape[0])
        # print self.state, np.average(self.particles, 0, self.weights).astype(np.int)

        
        
        
        self.template_hist.append(self.template)
        self.particles_hist.append(self.particles)
        self.weights_hist.append(self.weights)
        self.template_hist = self.template_hist[-self.hist_len:]
        self.particles_hist = self.particles_hist[-self.hist_len:]
        self.weights_hist = self.weights_hist[-self.hist_len:]
        
        #print len(self.template_hist), len(self.mse_hist)        
        '''
        out = np.concatenate((self.template, best, self.template_orig), 1)
        cv2.imshow('Tracking',  out)
        cv2.waitKey(1)
        '''
        





class MDParticleFilter(ParticleFilter):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.
        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor

        #pos change references the position change of the last frame to the current
        self.pos_change = (0, 0)
        self.mse_change = 0
        # self.scale = None
        #x = self.template_rect['x'] + np.floor(self.template_rect['w'] / 2.)
        #y = self.template_rect['y'] + np.floor(self.template_rect['h'] / 2.)
        self.mse   = 50
        self.scale = .99
        #self.state = (int(x), int(y))

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.
        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.
        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].
        Returns:
            None.
        """
        h, w = self.state # Initial state for frame
        gray_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        #gray_frame = self.normalize(np.dot(frame.copy(),self.img_weight))* 255.0

        # Random normal distribution of initial state
        rand_h = np.expand_dims(np.random.normal(h, self.sigma_dyn, self.num_particles), 1)
        rand_w = np.expand_dims(np.random.normal(w, self.sigma_dyn, self.num_particles), 1)
        self.particles = np.concatenate((rand_h, rand_w), axis=1)

        mse_ls = []
        similarity_ls = []
        template_list = []
        scale_list = []
        normalization = 0.
        for t, p in enumerate(self.particles):
            
            scale = np.random.randint(80, 100 + 1) / 100.
            template = self.template.copy()
            template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
            template_list.append(template)
            scale_list.append(scale)
            
            frame_cutout = self.get_cutout(gray_frame, p)
            resized_frame_cut = cv2.resize(src=frame_cutout, dsize=(0, 0), fx=self.scale, fy=self.scale).astype(np.uint8)
            
            if template.shape != resized_frame_cut.shape:
                resized_frame_cut = cv2.resize(src=resized_frame_cut, dsize=template.shape[::-1]).astype(np.uint8)
            
            error = self.get_error_metric(template, resized_frame_cut)
            mse_ls.append(error)
            similarity = np.exp(-error / (2. * self.sigma_exp ** 2.))
            similarity_ls.append(similarity)
            self.weights[t] += similarity
            normalization += self.weights[t]
            
        idx = np.argmax(similarity_ls)
        if scale_list[idx] > .95:
            self.template = template_list[idx]
            
        #self.weights   /= sum(self.weights)
        #self.particles = self.particles[self.resample_particles()]  # updates resampled particles
        
        if normalization > 0:
            self.weights /= normalization
            self.weights /= np.sum(self.weights)
        
    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """
        '''
        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.
        raise NotImplementedError
        '''
        
        w_s, h_s = self.state
        w_s, h_s = int(w_s), int(h_s)
        h, w = self.template.shape[:2]

        # draw rectangle
        cv2.rectangle(frame_in, (w_s - w/2, h_s - h/2), (w_s + w/2, h_s + h/2), (255, 255, 255), 2)
        
        # draw particles
        for px, py in self.particles.astype(np.int64):
            cv2.circle(frame_in, (px, py), 1, (0, 0, 255), -1)
            
        # draw circle
        dist   = np.sqrt(((self.particles - self.state) ** 2).sum(axis=1))
        radius = np.average(dist, axis=0, weights=self.weights).astype(np.int)
        cv2.circle(frame_in, (w_s, h_s), radius, (255, 255, 255), 1)
