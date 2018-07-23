import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import copy

###################################################################################################
######### Auxiliary Functions 
###################################################################################################

def plot(samples,theshape):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in  enumerate(samples[:25,:]):    #enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        #plt.imshow(sample.reshape(theshape), cmap='Greys_r')
        plt.imshow(sample.reshape(theshape))

    return fig

def next_random_batch(num, data, start=None):

    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]    
    data_shuffle = data[idx,:]  
    return data_shuffle


def next_batch(num, data, start):

    '''
    Return a total of `num` samples and labels. 
    '''
    idx = np.arange(start, np.min([start+num,len(data)]))
    return data[idx,:] 


def xavier_init(shape):
    in_dim = shape[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=shape, stddev=xavier_stddev)

# This function set the seeds of the tensorflow function
# to make this notebook's output stable across runs

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    #np.random.seed(seed)

###############################################################################################################################
########################################################### Network Descriptor #######################################################################################################################################################################################



class Network_Descriptor:
    def __init__(self, number_hidden_layers=1, input_dim=1, output_dim=1,  List_dims=[2], List_init_functions=[tf.random_uniform,tf.random_uniform], List_act_functions=[None,None], number_loop_train=1):
        self.number_hidden_layers = number_hidden_layers
        self.input_dim = input_dim 
        self.output_dim = output_dim  
        self.List_dims = List_dims
        self.List_init_functions = List_init_functions
        self.List_act_functions = List_act_functions 
        self.number_loop_train = number_loop_train

    def copy_from_othernetwork(self, other_network):       
        self.number_hidden_layers = other_network.number_hidden_layers
        self.input_dim =  other_network.input_dim
        self.output_dim = other_network.output_dim  
        self.List_dims = copy.deepcopy(other_network.List_dims)
        self.List_init_functions = copy.deepcopy(other_network.List_init_functions)
        self.List_act_functions = copy.deepcopy(other_network.List_act_functions)
        self.number_loop_train = other_network.number_loop_train


    def network_add_layer(self,layer_pos,lay_dims,init_w_function,init_a_function):
        """
        Function: network_add_layer()
        Adds a layer at a specified position, with a given  number of units, init weight
        function, activation function.
        If the layer is inserted in layer_pos \in [0,number_hidden_layers] then all the
        other layers are shifted.
        If the layer is inserted in position number_hidden_layers+1, then it is just appended
        to previous layer and it will output output_dim variables.
        If the position for the layer to be added is not within feasible bounds
        in the current architecture, the function silently returns
        """

        # If not within feasible bounds, return
        if layer_pos<0  or layer_pos>=self.number_hidden_layers:
            return

        # We create the new layer and add it to the network descriptor
        self.List_dims.insert(layer_pos,lay_dims) 
        self.List_init_functions.insert(layer_pos,init_w_function)   
        self.List_act_functions.insert(layer_pos,init_a_function)

        # Finally the number of hidden layers is updated
        self.number_hidden_layers = self.number_hidden_layers + 1


    """
    Function: network_remove_layer()
    Adds a layer at a specified position, with a given  number of units, init weight
    function, activation function.
    If the layer is inserted in layer_pos \in [0,number_hidden_layers] then all the
    other layers are shifted.
    If the layer is inserted in position number_hidden_layers+1, then it is just appended
    to previous layer and it will output output_dim variables.
    If the position for the layer to be added is not within feasible bounds
    in the current architecture, the function silently returns
    """

    def network_remove_layer(self,layer_pos):

        # If not within feasible bounds, return
        if layer_pos<=1  or layer_pos>self.number_hidden_layers:
            return

        # We set the number of input and output dimensions for the layer to be
        # added and for the ones in the architecture that will be connected to it
          
        # We delete the layer in pos layer_pos     
        self.List_dims.pop(layer_pos) 
        self.List_init_functions.pop(layer_pos)   
        self.List_act_functions.pop(layer_pos)
         
        # Finally the number of hidden layers is updated
        self.number_hidden_layers = self.number_hidden_layers - 1

    def network_remove_random_layer(self):
        layer_pos = np.random.randint(self.number_hidden_layers)
        self.network_remove_layer(layer_pos)
         

    def change_activation_fn_in_layer(self,layer_pos,new_act_fn):
        # If not within feasible bounds, return
        if layer_pos<0  or layer_pos>self.number_hidden_layers:
            return
        #print(layer_pos, "Old act fn ",self.List_act_functions[layer_pos], "New act fn", new_act_fn)
        self.List_act_functions[layer_pos] = new_act_fn


    def change_weight_init_fn_in_layer(self,layer_pos,new_weight_fn):
        # If not within feasible bounds, return
        if layer_pos<0  or layer_pos>self.number_hidden_layers:
            return
        self.List_init_functions[layer_pos] = new_weight_fn

    def change_all_weight_init_fns(self,new_weight_fn):
        # If not within feasible bounds, return
        for layer_pos in range(self.number_hidden_layers):
            self.List_init_functions[layer_pos] = new_weight_fn


    def change_dimensions_in_layer(self,layer_pos,new_dim):
        # If not within feasible bounds, return
        if layer_pos<0  or layer_pos>self.number_hidden_layers:
            return
        # If the dimension of the layer is identical to the existing one, return
        self.List_dims[layer_pos] = new_dim

    def change_dimensions_in_random_layer(self,max_layer_size):
        layer_pos = np.random.randint(self.number_hidden_layers)
        new_dim = np.random.randint(max_layer_size)+1
        self.change_dimensions_in_layer(layer_pos,new_dim)
        
  
    def print_components(self,identifier):
        print(identifier,' n_hid:', self.number_hidden_layers)
        print(identifier,' Dims:', self.List_dims)
        print(identifier,' Init:', self.List_init_functions)
        print(identifier,' Act:', self.List_act_functions)
        print(identifier,' Loop:', self.number_loop_train)


    def codify_components(self,max_hidden_layers,ref_List_init_functions,ref_List_act_functions):

        max_total_layers = max_hidden_layers + 1
        # The first two elements of code are the number of layers and number of loops      
        code = [self.number_hidden_layers,self.number_loop_train]
 
        # We add all the layer dimension and fill with zeros all positions until max_layers
        code = code + self.List_dims + [-1]*(max_total_layers-len(self.List_dims))
 
        # We add the indices of init_functions in each layer
        # and fill with zeros all positions until max_layers
        aux_f = []
        for init_f in self.List_init_functions:
            aux_f.append(ref_List_init_functions.index(init_f))
        code = code + aux_f + [-1]*(max_total_layers-len(aux_f))
 
        # We add the indices of act_functions in each layer
        # and fill with zeros all positions until max_layers
        aux_a = []
        for act_f in self.List_act_functions:
            aux_a.append(ref_List_act_functions.index(act_f))
        code = code + aux_a + [-1]*(max_total_layers-len(aux_a))
 
        return code 





###############################################################################################################################
########################################################### Network #######################################################################################################################################################################################


class Network:
    def __init__(self, network_descriptor):
        self.descriptor = network_descriptor
        self.List_layers = []
        self.List_weights = []        
        self.List_bias = []    
        
     
    def reset_network(self):  
        self.List_layers = []
        self.List_weights = []        
        self.List_bias = []   
        self.List_dims = []
        self.List_init_functions = []
        self.List_act_functions = []


    def create_hidden_layer(self, in_size, out_size, init_w_function, layer_name):               
        w = tf.Variable(init_w_function(shape=[in_size, out_size]), name="W"+layer_name)
        b = tf.Variable(tf.zeros(shape=[out_size]), name="b"+layer_name)
        return w, b
    
    def network_initialization(self):
        for lay in range(self.descriptor.number_hidden_layers+1):   
            init_w_function = self.descriptor.List_init_functions[lay]    
            if lay==0:
                in_size = self.descriptor.input_dim
                out_size = self.descriptor.List_dims[lay]
            elif lay==self.descriptor.number_hidden_layers:
                in_size = self.descriptor.List_dims[lay-1]
                out_size = self.descriptor.output_dim 
            else:
                in_size = self.descriptor.List_dims[lay-1]
                out_size = self.descriptor.List_dims[lay]
            #print(lay,in_size,out_size) 
            w, b = self.create_hidden_layer(in_size, out_size, init_w_function,str(lay))                   

            self.List_weights.append(w)        
            self.List_bias.append(b) 


    def network_evaluation(self,layer):
        for lay in range(self.descriptor.number_hidden_layers+1):
            w = self.List_weights[lay]
            b = self.List_bias[lay] 
            act = self.descriptor.List_act_functions[lay]
            if act==None:
                layer =  tf.matmul(layer,w) + b
            else:
                if act!=None:
                    layer =  act( tf.matmul(layer,w) + b )

            self.List_layers.append(layer)    

        return layer


                


###############################################################################################################################
########################################################### GAN Descriptor  #######################################################################################################################################################################################

class GAN_Descriptor:
    def __init__(self, X_dim, z_dim, latent_distribution_function=np.random.uniform,  fmeasure="Standard_Divergence"):
        self.X_dim = X_dim
        self.z_dim = z_dim
        self.latent_distribution_function = latent_distribution_function
        self.fmeasure = fmeasure
        self.Gen_network = []
        self.Disc_network = []


    def copy_from_other(self, other):    
        self.X_dim = other.X_dim
        self.z_dim = other.z_dim
        self.latent_distribution_function = other.latent_distribution_function

        self.fmeasure = other.fmeasure

        self.Gen_network = copy.deepcopy(other.Gen_network)     # These are  Network_Descriptor structures
        self.Disc_network =  copy.deepcopy(other.Disc_network)


    def gan_generator_initialization(self,generator_n_hidden=1, generator_dim_list=[10], 
                                     generator_init_functions=[tf.random_normal, tf.random_normal],  
                                     generator_act_functions=[tf.nn.tanh,tf.nn.tanh], generator_number_loop_train=1):

        self.Gen_network = Network_Descriptor(generator_n_hidden, self.z_dim, 
                                       self.X_dim, generator_dim_list, generator_init_functions, 
                                       generator_act_functions, generator_number_loop_train)
   


    def gan_discriminator_initialization(self,discriminator_n_hidden=10, discriminator_dim_list=[10], 
                                         discriminator_init_functions=[tf.random_normal, tf.random_normal],  
                                         discriminator_act_functions=[tf.nn.tanh,tf.nn.tanh], discriminator_number_loop_train=1):
        output_dim = 1     
        self.Disc_network = Network_Descriptor(discriminator_n_hidden, self.X_dim, 
                                        output_dim, discriminator_dim_list, discriminator_init_functions, 
                                        discriminator_act_functions, discriminator_number_loop_train)


    def print_components(self):
        self.Gen_network.print_components("Gen")
        self.Disc_network.print_components("Disc")

        print('Latent:',  self.latent_distribution_function)
        print('Divergence_Measure:', self.fmeasure)
 

    def codify_components(self,max_layers,ref_List_init_functions,ref_List_act_functions, ref_List_divergence_functions,ref_List_latent_functions):

        latent_index = ref_List_latent_functions.index(self.latent_distribution_function)        
        diverg_index = ref_List_divergence_functions.index(self.fmeasure)

        # The first two elements are the indices of the latent and divergence functions
        code = [latent_index,diverg_index]

        # Ve add the code of the generator
        code = code + self.Gen_network.codify_components(max_layers,ref_List_init_functions,ref_List_act_functions)

        # Ve add the code of the discriminator
        code = code + self.Disc_network.codify_components(max_layers,ref_List_init_functions,ref_List_act_functions)

        return code 



###################################################################################################
############################################# GAN  ################################################
###################################################################################################


class GAN:
    def __init__(self, gan_descriptor):
        self.descriptor = gan_descriptor
        self.Div_Function = self.Divergence_Functions[self.descriptor.fmeasure]
        self.Gen_network = []
        self.Disc_network = []

    def reset_network(self):  
        self.Gen_network.reset_network()
        self.Disc_network.reset_network()


    def sample_Z(self,m, n): 
        return self.descriptor.latent_distribution_function(-1., 1., size=[m, n])
        #return np.random.uniform(-1., 1., size=[m, n])

    def generator(self,z):
        G_log_prob = self.Gen_network.network_evaluation(z)
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob
 
    def discriminator(self,x):
        D_logit = self.Disc_network.network_evaluation(x)
        D_prob = tf.nn.sigmoid(D_logit) 
        return D_prob, D_logit

    def training_definition(self,lr):
        # =============================== TRAINING ====================================

        self.X = tf.placeholder(tf.float32, shape=[None, self.descriptor.X_dim])
        self.Z = tf.placeholder(tf.float32, shape=[None, self.descriptor.z_dim])

        self.Gen_network = Network(self.descriptor.Gen_network)
        self.Disc_network = Network(self.descriptor.Disc_network)

        self.Gen_network.network_initialization()
        self.Disc_network.network_initialization()


        with tf.variable_scope('Gen1') as scope:
                self.G_sample = self.generator(self.Z)
        with tf.variable_scope('Disc1') as scope:
                self.D_real, self.D_logit_real = self.discriminator(self.X)
        with tf.variable_scope('Disc2') as scope:    
                self.D_fake, self.D_logit_fake = self.discriminator(self.G_sample)

        self.Div_Function(self)

        self.G_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.G_loss, var_list = [self.Gen_network.List_weights,self.Gen_network.List_bias])
        self.D_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.D_loss, var_list = [self.Disc_network.List_weights,self.Disc_network.List_bias])



    def running(self, batch_type, data, mb_size, number_iterations, nsamples, print_cycle):
        config = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        if not os.path.exists('gan_out/'):
            os.makedirs('gan_out/')

        if batch_type=="random":
                    batch_function = next_random_batch
        else: 
                    batch_function = next_batch

        i = 0
        for it in range(number_iterations):

            X_mb = batch_function(mb_size, data,i)
            i = i+mb_size if (i+mb_size)<len(data) else 0

            z_mb = self.sample_Z(mb_size, self.z_dim)
            _, D_loss_curr = sess.run([self.D_solver, self.D_loss], feed_dict={self.X: X_mb, self.Z: z_mb })
            _, G_loss_curr = sess.run([self.G_solver, self.G_loss], feed_dict={self.Z: z_mb})

            if it>0  and it % print_cycle == 0:
                print('Iter: {}'.format(it))
                print('D Loss: {:.4}'. format(D_loss_curr))
                print('G_Loss: {:.4}'. format(G_loss_curr))
                print()

        samples = sess.run(self.G_sample, feed_dict={self.Z: self.sample_Z(nsamples, self.descriptor.z_dim) })

        return samples 


    def separated_running(self, batch_type, data, mb_size, number_iterations, nsamples, print_cycle):
        config = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        if not os.path.exists('gan_out/'):
            os.makedirs('gan_out/')
        if batch_type=="random":
                    batch_function = next_random_batch
        else: 
                    batch_function = next_batch

        i = 0
        for it in range(number_iterations):
            # Learning loop for discriminator

            for loop_disc in range(self.Gen_network.descriptor.number_loop_train):
                    z_mb = self.sample_Z(mb_size, self.descriptor.z_dim)
                    X_mb = batch_function(mb_size, data,i)
                    i = i+mb_size if (i+mb_size)<len(data) else 0
                    _, D_loss_curr = sess.run([self.D_solver, self.D_loss], feed_dict={self.X: X_mb, self.Z: z_mb })

            # Learning loop for generator
            for loop_gen in range(self.Disc_network.descriptor.number_loop_train):
                    z_mb = self.sample_Z(mb_size, self.descriptor.z_dim)
                    X_mb = batch_function(mb_size, data,i)
                    i = i+mb_size if (i+mb_size)<len(data) else 0                   
                    _, G_loss_curr = sess.run([self.G_solver, self.G_loss], feed_dict={self.Z: z_mb})

            if it>0  and it % print_cycle == 0:
                print('Iter: {}'.format(it))
                print('D Loss: {:.4}'. format(D_loss_curr))
                print('G_Loss: {:.4}'. format(G_loss_curr))
                print()

        samples = sess.run(self.G_sample, feed_dict={self.Z: self.sample_Z(nsamples, self.descriptor.z_dim) })

        return samples 

    def Standard_Divergence(self):
            self.D_loss =  -tf.reduce_mean(self.D_logit_real) + tf.reduce_mean(self.D_logit_fake)
            self.G_loss =  -tf.reduce_mean(self.D_logit_fake)


    def Total_Variation(self):
            self.D_loss = -(tf.reduce_mean(0.5 * tf.nn.tanh(self.D_logit_real)) - tf.reduce_mean(0.5 * tf.nn.tanh(self.D_logit_fake)))
            self.G_loss = -tf.reduce_mean(0.5 * tf.nn.tanh(self.D_logit_fake))


    def Forward_KL(self):
            self.D_loss = -(tf.reduce_mean(self.D_logit_real) - tf.reduce_mean(tf.exp(self.D_logit_fake-1)))
            self.G_loss = -tf.reduce_mean(tf.exp(self.D_logit_fake-1))

    def Reverse_KL(self):
            self.D_loss = -(tf.reduce_mean(-self.D_real) - tf.reduce_mean(1-self.D_logit_fake))
            self.G_loss = -tf.reduce_mean(-1 -self.D_logit_fake)

            # D_loss = -(tf.reduce_mean(-tf.exp(D_real)) - tf.reduce_mean(-1 - D_fake))
            # G_loss = -tf.reduce_mean(-1 - D_fake)


    def Pearson_Chi_squared(self):
            self.D_loss = -(tf.reduce_mean(self.D_logit_real) - tf.reduce_mean(0.25*self.D_logit_fake**2 + self.D_logit_fake))
            self.G_loss = -tf.reduce_mean(0.25*self.D_logit_fake**2 + self.D_logit_fake)

    def Squared_Hellinger(self):
            #self.D_loss = -(tf.reduce_mean(1 - self.D_real) - tf.reduce_mean( (1 - self.D_fake) / self.D_fake))
            self.D_loss = -(tf.reduce_mean(1 - self.D_real) - tf.reduce_mean( self.D_fake / (1-self.D_fake)))
            self.G_loss = -tf.reduce_mean((1 -self.D_fake) / self.D_fake)

    def Least_squared(self):
            self.D_loss = 0.5 * (tf.reduce_mean((self.D_logit_real - 1)**2) + tf.reduce_mean(self.D_logit_fake**2))
            self.G_loss = 0.5 * tf.reduce_mean((self.D_logit_fake - 1)**2)



    Divergence_Functions = {"Standard_Divergence": Standard_Divergence,\
                                 "Total_Variation": Total_Variation, \
                                 "Forward_KL": Forward_KL, \
                                 "Reverse_KL": Reverse_KL, \
                                 "Pearson_Chi_squared": Pearson_Chi_squared, \
                                 "Squared_Hellinger": Squared_Hellinger, \
                                 "Least_squared": Least_squared}


    def set_divergence_function(self,fmeasure):
        self.fmeasure = fmeasure
        self.Div_Function = self.Divergence_Functions[self.fmeasure]


###############################################################################################################################
########################################################### VAE Descriptor  #######################################################################################################################################################################################

class VAE_Descriptor:
    def __init__(self, X_dim, z_dim, latent_distribution_function=np.random.uniform,  fmeasure="Standard_Divergence"):
        self.X_dim = X_dim
        self.z_dim = z_dim
        # latent_distribution_function is left here for consistency with GAN and future extensions
        # but currently, only Normal distribution is used for Z (think about of flow distributions)
        self.latent_distribution_function = latent_distribution_function
        self.fmeasure = fmeasure
        self.Enc_network = []
        self.Dec_network = []

    def vae_encoder_initialization(self,encoder_n_hidden=1, encoder_dim_list=[10], 
                                     encoder_init_functions=[tf.random_normal, tf.random_normal],  
                                     encoder_act_functions=[tf.nn.tanh,tf.nn.tanh], encoder_number_loop_train=1):
        input_dim = self.X_dim
        output_dim = self.z_dim
        self.Enc_network = Network_Descriptor(encoder_n_hidden, input_dim, 
                                       output_dim, encoder_dim_list, encoder_init_functions, 
                                       encoder_act_functions, encoder_number_loop_train) 

    def vae_decoder_initialization(self,decoder_n_hidden=10, decoder_dim_list=[10], 
                                         decoder_init_functions=[tf.random_normal, tf.random_normal],  
                                         decoder_act_functions=[tf.nn.tanh,tf.nn.tanh], decoder_number_loop_train=1):
        output_dim = self.X_dim
        input_dim = self.z_dim

        self.Dec_network = Network_Descriptor(decoder_n_hidden, input_dim, 
                                        output_dim, decoder_dim_list, decoder_init_functions, 
                                        decoder_act_functions, decoder_number_loop_train)


    def print_components(self):
        self.Enc_network.print_components("Enc")
        self.Dec_network.print_components("Dec")

        print('Latent:',  self.latent_distribution_function)
        print('Divergence_Measure:',  self.fmeasure)

    def codify_components(self,max_layers,ref_List_init_functions,ref_List_act_functions, ref_List_divergence_functions,ref_List_latent_functions):

        latent_index = ref_List_latent_functions.index(self.latent_distribution_function)        
        diverg_index = ref_List_divergence_functions.index(self.fmeasure)

        # The first two elements are the indices of the latent and divergence functions
        code = [latent_index,diverg_index]

        # Ve add the code of the encoder
        code = code + self.Enc_network.codify_components(max_layers,ref_List_init_functions,ref_List_act_functions)

        # Ve add the code of the decoder
        code = code + self.Dec_network.codify_components(max_layers,ref_List_init_functions,ref_List_act_functions)

        return code 



###################################################################################################
############################################# VAE  ################################################
###################################################################################################


class VAE:
    def __init__(self, vae_descriptor):
        self.descriptor = vae_descriptor
        self.Div_Function = self.Divergence_Functions[self.descriptor.fmeasure]
        self.Enc_network = []
        self.Dec_network = []

    def reset_network(self):  
        self.Enc_network.reset_network()
        self.Dec_network.reset_network()


    def sample_Z(self,mu, log_var): 
        # Currently sample_Z is implemented for Normal distribution. Alternative posteriors distributions
        # could be implemented, but loss function should be modified accordingly
        eps = tf.random_normal(shape=tf.shape(mu))       
        return mu + tf.exp(log_var / 2) * eps
       

    def encoder(self,z):
        z_log_prob = self.Enc_network.network_evaluation(z)
        z_prob = tf.nn.sigmoid(z_log_prob)
        return z_prob, z_log_prob
 
    def decoder(self,x):
        x_logit = self.Dec_network.network_evaluation(x)
        x_prob = tf.nn.sigmoid(x_logit) 
        return x_prob, x_logit



    def training_definition(self,lr):
        # =============================== VAE TRAINING ====================================

        self.X = tf.placeholder(tf.float32, shape=[None, self.descriptor.X_dim])
        self.Z = tf.placeholder(tf.float32, shape=[None, self.descriptor.z_dim])

        self.Enc_network = Network(self.descriptor.Enc_network)
        self.Dec_network = Network(self.descriptor.Dec_network)

        self.Enc_network.network_initialization()
        self.Dec_network.network_initialization()


        with tf.variable_scope('Enc1') as scope:
                self.z_mu, self.z_logvar = self.encoder(self.X)
                self.z_sample = self.sample_Z(self.z_mu, self.z_logvar)

        with tf.variable_scope('Dec1') as scope:
                _, self.logits = self.decoder(self.z_sample)
                # Sampling from random z
                self.X_samples, _ = self.decoder(self.Z)

        self.VAE_Loss()

        self.solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.vae_loss)


    def VAE_Loss(self):

        self.Div_Function(self)

        # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
        self.kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.z_logvar) + self.z_mu**2 - 1. - self.z_logvar, 1)

        # VAE loss
        self.vae_loss = tf.reduce_mean(self.recon_loss + self.kl_loss)


    def VAE_Gaussian_Div(self):
       # E[log P(X|z)]
        self.recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.X), 1)
        #self.recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.X), 1) + tf.losses.mean_squared_error(predictions=logits, labels=self.X)

    def VAE_MSE_Div(self):
       # E[log P(X|z)]
        self.recon_loss = tf.losses.mean_squared_error(predictions=self.logits, labels=self.X)
   
    def Standard_Divergence(self):
            self.recon_loss =  -tf.reduce_mean(self.logits)

    def Total_Variation(self):
            self.recon_loss  = -tf.reduce_mean(0.5 * tf.nn.tanh(self.logits)) 

    def Forward_KL(self):         
            self.recon_loss = -tf.reduce_mean(tf.exp(self.logits))

    def Reverse_KL(self):
            self.recon_loss = -tf.reduce_mean(-self.logits)


    def Pearson_Chi_squared(self):
            self.recon_loss = -tf.reduce_mean(self.logits) 

    def Squared_Hellinger(self):         
            self.recon_loss = -tf.reduce_mean(1 - self.logits) 


    def Least_squared(self):
            #self.recon_loss = tf.losses.mean_squared_error(predictions=self.logits, labels=self.X)
            self.recon_loss = tf.reduce_mean((self.logits - 1)**2)
         




    Divergence_Functions = {"Standard_Divergence": Standard_Divergence,\
                                 "Total_Variation": Total_Variation, \
                                 "Forward_KL": Forward_KL, \
                                 "Reverse_KL": Reverse_KL, \
                                 "Pearson_Chi_squared": Pearson_Chi_squared, \
                                 "Squared_Hellinger": Squared_Hellinger, \
                                 "Least_squared": Least_squared, \
                                 "VAE_Gaussian_Div": VAE_Gaussian_Div, \
                                 "VAE_MSE_Div": VAE_MSE_Div}

    def running(self, batch_type, data, mb_size, number_iterations, nsamples, print_cycle):
        config = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        if not os.path.exists('vae_out/'):
            os.makedirs('vae_out/')

        if batch_type=="random":
                    batch_function = next_random_batch
        else: 
                    batch_function = next_batch

        i = 0
        for it in range(number_iterations):

            X_mb = batch_function(mb_size, data,i)
            i = i+mb_size if (i+mb_size)<len(data) else 0

            #z_mb = self.sample_Z(mb_size, self.z_dim)
            _, loss = sess.run([self.solver, self.vae_loss], feed_dict={self.X: X_mb})

            if it>0  and it % print_cycle == 0:
                print('Iter: {}'.format(it))
                print('V Loss: {:.4}'. format(loss))
                print()

        samples = sess.run(self.X_samples, feed_dict={self.Z:  np.random.randn(nsamples, self.descriptor.z_dim) })

        return samples 
