import tensorflow as tf
import numpy as np
import copy
import random
###################################################################################################
# ######## Auxiliary Functions
###################################################################################################


def reset_graph(seed):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def next_random_batch(num, data):
    """
    Return a total of `num` random samples and labels.
    """
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx, :]
    return data_shuffle


def next_batch(size, x, i):
    """
    :param x: Poplation; set of solutions intended to be fed to the net in the input
    :param size: Size of the batch desired
    :param i: Index of the last solution used in the last epoch
    :return: The index of the last solution in the batch (to be provided to this same
             function in the next epoch, the solutions in the actual batch, and their
             respective fitness scores
    """

    if i + size > x.shape[0]:  # In case there are not enough solutions before the end of the array
        index = i + size-x.shape[0]  # Select all the individuals until the end and restart
        return np.concatenate((x[i:, :], x[:index, :]))
    else:  # Easy case
        index = i+size
        return x[i:index, :]


def xavier_init(shape, constant=1):
    """ Xavier initialization of network weights"""
    fan_in = shape[0]
    fan_out = shape[1]
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

########################################################################################################################
# ########################################################## Network Descriptor ########################################


class NetworkDescriptor:
    def __init__(self, number_hidden_layers=1, input_dim=1, output_dim=1,  list_dims=None, list_init_functions=None,
                 list_act_functions=None, number_loop_train=1):
        self.number_hidden_layers = number_hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.List_dims = list_dims
        self.List_init_functions = list_init_functions
        self.List_act_functions = list_act_functions
        self.number_loop_train = number_loop_train

    def copy_from_other_network(self, other_network):
        self.number_hidden_layers = other_network.number_hidden_layers
        self.input_dim = other_network.input_dim
        self.output_dim = other_network.output_dim
        self.List_dims = copy.deepcopy(other_network.List_dims)
        self.List_init_functions = copy.deepcopy(other_network.List_init_functions)
        self.List_act_functions = copy.deepcopy(other_network.List_act_functions)
        self.number_loop_train = other_network.number_loop_train

    def network_add_layer(self, layer_pos, lay_dims, init_w_function, init_a_function):
        """
        Function: network_add_layer()
        Adds a layer at a specified position, with a given  number of units, init weight
        function, activation function.
        If the layer is inserted in layer_pos in [0,number_hidden_layers] then all the
        other layers are shifted.
        If the layer is inserted in position number_hidden_layers+1, then it is just appended
        to previous layer and it will output output_dim variables.
        If the position for the layer to be added is not within feasible bounds
        in the current architecture, the function silently returns
        """

        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos >= self.number_hidden_layers:
            return

        # We create the new layer and add it to the network descriptor
        self.List_dims.insert(layer_pos, lay_dims)
        self.List_init_functions.insert(layer_pos, init_w_function)
        self.List_act_functions.insert(layer_pos, init_a_function)

        # Finally the number of hidden layers is updated
        self.number_hidden_layers = self.number_hidden_layers + 1

    def network_remove_layer(self, layer_pos):
        """
        Function: network_remove_layer()
        Adds a layer at a specified position, with a given  number of units, init weight
        function, activation function.
        If the layer is inserted in layer_pos in [0,number_hidden_layers] then all the
        other layers are shifted.
        If the layer is inserted in position number_hidden_layers+1, then it is just appended
        to previous layer and it will output output_dim variables.
        If the position for the layer to be added is not within feasible bounds
        in the current architecture, the function silently returns
        """

        # If not within feasible bounds, return
        if layer_pos <= 1 or layer_pos > self.number_hidden_layers:
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

    def change_activation_fn_in_layer(self, layer_pos, new_act_fn):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        self.List_act_functions[layer_pos] = new_act_fn

    def change_weight_init_fn_in_layer(self, layer_pos, new_weight_fn):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        self.List_init_functions[layer_pos] = new_weight_fn

    def change_all_weight_init_fns(self, new_weight_fn):
        # If not within feasible bounds, return
        for layer_pos in range(self.number_hidden_layers):
            self.List_init_functions[layer_pos] = new_weight_fn

    def change_dimensions_in_layer(self, layer_pos, new_dim):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        # If the dimension of the layer is identical to the existing one, return
        self.List_dims[layer_pos] = new_dim

    def change_dimensions_in_random_layer(self, max_layer_size):
        layer_pos = np.random.randint(self.number_hidden_layers)
        new_dim = np.random.randint(max_layer_size)+1
        self.change_dimensions_in_layer(layer_pos, new_dim)

    def print_components(self, identifier):
        print(identifier, ' n_hid:', self.number_hidden_layers)
        print(identifier, ' Dims:', self.List_dims)
        print(identifier, ' Init:', self.List_init_functions)
        print(identifier, ' Act:', self.List_act_functions)
        print(identifier, ' Loop:', self.number_loop_train)

    def codify_components(self, max_hidden_layers, ref_list_init_functions, ref_list_act_functions):

        max_total_layers = max_hidden_layers + 1
        # The first two elements of code are the number of layers and number of loops
        code = [self.number_hidden_layers, self.number_loop_train]

        # We add all the layer dimension and fill with zeros all positions until max_layers
        # print(code, self.List_dims, [-1]*(max_total_layers-len(self.List_dims)))
        code = code + self.List_dims + [-1]*(max_total_layers-len(self.List_dims))

        # We add the indices of init_functions in each layer
        # and fill with zeros all positions until max_layers
        aux_f = []
        for init_f in self.List_init_functions:
            aux_f.append(ref_list_init_functions.index(init_f))
        code = code + aux_f + [-1]*(max_total_layers-len(aux_f))

        # We add the indices of act_functions in each layer
        # and fill with zeros all positions until max_layers
        aux_a = []
        for act_f in self.List_act_functions:
            aux_a.append(ref_list_act_functions.index(act_f))
        code = code + aux_a + [-1]*(max_total_layers-len(aux_a))

        return code

#######################################################################################################################
# ########################################################## Network ##################################################


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

    @staticmethod
    def create_hidden_layer(in_size, out_size, init_w_function, layer_name):
        w = tf.Variable(init_w_function(shape=[in_size, out_size]), name="W"+layer_name)
        b = tf.Variable(tf.zeros(shape=[out_size]), name="b"+layer_name)
        return w, b

    def network_initialization(self):
        for lay in range(self.descriptor.number_hidden_layers+1):
            init_w_function = self.descriptor.List_init_functions[lay]
            if lay == 0:
                in_size = self.descriptor.input_dim
                out_size = self.descriptor.List_dims[lay]
            elif lay == self.descriptor.number_hidden_layers:
                in_size = self.descriptor.List_dims[lay-1]
                out_size = self.descriptor.output_dim
            else:
                in_size = self.descriptor.List_dims[lay-1]
                out_size = self.descriptor.List_dims[lay]

            w, b = self.create_hidden_layer(in_size, out_size, init_w_function, str(lay))

            self.List_weights.append(w)
            self.List_bias.append(b)

    def network_evaluation(self, layer):
        for lay in range(self.descriptor.number_hidden_layers+1):
            w = self.List_weights[lay]
            b = self.List_bias[lay]

            act = self.descriptor.List_act_functions[lay]
            if act is None:
                layer = tf.matmul(layer, w) + b
            else:
                if act is not None:
                    layer = act(tf.matmul(layer, w) + b)

            self.List_layers.append(layer)

        return layer


########################################################################################################################
# ########################################################## GAN Descriptor  ###########################################

class GANDescriptor:
    def __init__(self, x_dim, z_dim, latent_distribution_function=np.random.uniform,  f_measure="Standard_Divergence",
                 l_rate=0.001):
        self.X_dim = x_dim
        self.z_dim = z_dim
        self.latent_distribution_function = latent_distribution_function
        self.f_measure = f_measure
        self.Gen_network = None
        self.Disc_network = None
        self.l_rate = l_rate

    def copy_from_other(self, other):
        self.X_dim = other.X_dim
        self.z_dim = other.z_dim
        self.latent_distribution_function = other.latent_distribution_function

        self.f_measure = other.fmeasure

        self.Gen_network = copy.deepcopy(other.Gen_network)     # These are  Network_Descriptor structures
        self.Disc_network = copy.deepcopy(other.Disc_network)

    def gan_generator_initialization(self, n_hidden=1, dim_list=None, init_functions=None, act_functions=None,
                                     number_loop_train=1):

        self.Gen_network = NetworkDescriptor(n_hidden, self.z_dim, self.X_dim, dim_list, init_functions, act_functions,
                                             number_loop_train)

    def gan_discriminator_initialization(self, n_hidden=10, dim_list=None, init_functions=None, act_functions=None,
                                         number_loop_train=1):
        output_dim = 1
        self.Disc_network = NetworkDescriptor(n_hidden, self.X_dim, output_dim, dim_list, init_functions, act_functions,
                                              number_loop_train)

    def print_components(self):
        self.Gen_network.print_components("Gen")
        self.Disc_network.print_components("Disc")

        print('Latent:',  self.latent_distribution_function)
        print('Divergence_Measure:', self.f_measure)

    def codify_components(self, max_layers, init_functions, act_functions, divergence_functions, latent_functions):

        latent_index = latent_functions.index(self.latent_distribution_function)
        diverg_index = divergence_functions.index(self.f_measure)

        # The first two elements are the indices of the latent and divergence functions
        code = [latent_index, diverg_index]

        # Ve add the code of the generator
        code = code + self.Gen_network.codify_components(max_layers, init_functions, act_functions)

        # Ve add the code of the discriminator
        code = code + self.Disc_network.codify_components(max_layers, init_functions, act_functions)

        return code


###################################################################################################
# ############################################ GAN  ###############################################
###################################################################################################


class GAN:
    def __init__(self, gan_descriptor):
        self.descriptor = gan_descriptor
        self.Div_Function = Divergence_Fs[self.descriptor.f_measure]
        self.Gen_network = None
        self.Disc_network = None
        self.X = None
        self.Z = None
        self.g_sample = None
        self.d_real = None
        self.d_logit_real = None
        self.d_fake = None
        self.d_logit_fake = None
        self.g_solver = None
        self.d_solver = None
        self.D_loss = None
        self.G_loss = None
        self.f_measure = None

    def reset_network(self):
        self.Gen_network.reset_network()
        self.Disc_network.reset_network()

    def sample_z(self, m, n):
        return self.descriptor.latent_distribution_function(-1., 1., size=[m, n])

    def generator(self, z):
        g_log_prob = self.Gen_network.network_evaluation(z)
        g_prob = tf.nn.sigmoid(g_log_prob)
        return g_prob

    def discriminator(self, x):
        res = self.Disc_network.network_evaluation(x)

        d_logit = -tf.nn.relu(res)
        return res, d_logit

    def training_definition(self, seed=None):
        # =============================== TRAINING ====================================
        if seed is not None:

            reset_graph(seed)

        self.X = tf.placeholder(tf.float32, shape=[None, self.descriptor.X_dim])
        self.Z = tf.placeholder(tf.float32, shape=[None, self.descriptor.z_dim])

        self.Gen_network = Network(self.descriptor.Gen_network)
        self.Disc_network = Network(self.descriptor.Disc_network)

        self.Gen_network.network_initialization()
        self.Disc_network.network_initialization()
        with tf.variable_scope('Gen1') as scope:
            self.g_sample = self.generator(self.Z)
        with tf.variable_scope('Disc1') as scope:
            self.d_real, self.d_logit_real = self.discriminator(self.X)
        with tf.variable_scope('Disc2') as scope:
            self.d_fake, self.d_logit_fake = self.discriminator(self.g_sample)

        self.Div_Function(self)
        self.g_solver = tf.train.AdamOptimizer(learning_rate=self.descriptor.l_rate).minimize(
            self.G_loss, var_list=[self.Gen_network.List_weights, self.Gen_network.List_bias])

        self.d_solver = tf.train.AdamOptimizer(learning_rate=self.descriptor.l_rate).minimize(
            self.D_loss, var_list=[self.Disc_network.List_weights, self.Disc_network.List_bias])

    def running(self, batch_type, data, mb_size, number_iterations, nsamples, print_cycle):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        if batch_type == "random":
            batch_function = next_random_batch
        else:
            batch_function = next_batch

        i = 0
        for it in range(number_iterations):

            x_mb = batch_function(mb_size, data, i)
            i = i+mb_size if (i+mb_size) < len(data) else 0

            z_mb = self.sample_z(mb_size, self.descriptor.z_dim)
            _, d_loss_curr = sess.run([self.d_solver, self.D_loss], feed_dict={self.X: x_mb, self.Z: z_mb})
            _, g_loss_curr = sess.run([self.g_solver, self.G_loss], feed_dict={self.Z: z_mb})

            if it > 0 and it % print_cycle == 0:
                print('Iter: {}'.format(it))
                print('D Loss: {:.4}'. format(d_loss_curr))
                print('G_Loss: {:.4}'. format(g_loss_curr))
                print()

        samples = sess.run(self.g_sample, feed_dict={self.Z: self.sample_z(nsamples, self.descriptor.z_dim)})

        return samples

    def separated_running(self, batch_type, data, mb_size, number_iterations, nsamples, print_cycle, convergence=None):
        if convergence is not None:
            gen_losses = [1] + [2147483647]*(convergence-1)
            disc_losses = [1] + [2147483647]*(convergence-1)
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}))
        sess.run(tf.global_variables_initializer())

        if batch_type == "random":
            batch_function = next_random_batch
        else:
            batch_function = next_batch

        i = 0
        d_loss_curr = 0
        g_loss_curr = 0
        best_gen_loss = 2147483647
        best_disc_loss = 2147483647

        for it in range(number_iterations):
            # Learning loop for discriminator

            for loop_disc in range(self.Disc_network.descriptor.number_loop_train):
                z_mb = self.sample_z(mb_size, self.descriptor.z_dim)
                x_mb = batch_function(mb_size, data, i)

                i = i+mb_size if (i+mb_size) < len(data) else 0

                _, d_loss_curr = sess.run([self.d_solver, self.D_loss], feed_dict={self.X: x_mb, self.Z: z_mb})

                if it > 0 and it % print_cycle == 0:
                    print('Iter: {}'.format(it))
                    print('D Loss: {:.4}'. format(d_loss_curr))
                    print()

            # Learning loop for generator
            for loop_gen in range(self.Gen_network.descriptor.number_loop_train):
                z_mb = self.sample_z(mb_size, self.descriptor.z_dim)
                i = i+mb_size if (i+mb_size) < len(data) else 0
                _, g_loss_curr = sess.run([self.g_solver, self.G_loss], feed_dict={self.Z: z_mb})

                if it > 0 and it % print_cycle == 0:
                    print('Iter: {}'.format(it))
                    print('G_Loss: {:.4}'. format(g_loss_curr))
                    print()

            if convergence is not None:
                disc_losses = [d_loss_curr] + disc_losses[:-1]
                best_disc_loss = np.min([best_disc_loss, d_loss_curr])
                gen_losses = [g_loss_curr] + gen_losses[:-1]

                # if best_disc_loss >= np.min(disc_losses[1:])*1.1:
                #     print(best_disc_loss, np.min(disc_losses[1:]))
                #     print("Discriminator early termination")
                #     break

                if np.max(gen_losses) == np.min(gen_losses):
                    print(best_gen_loss, np.min(gen_losses), it)
                    print("Generator early termination")
                    break

            if np.isnan(g_loss_curr) or np.isnan(d_loss_curr):
                print("NaN loss value")
                break

        samples = sess.run(self.g_sample, feed_dict={self.Z: self.sample_z(nsamples, self.descriptor.z_dim)})

        return samples

    def standard_divergence(self):
        self.D_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real, labels=tf.ones_like(self.d_real)) +
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.zeros_like(self.d_fake)))
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.ones_like(self.d_fake)))

    def total_variation(self):
        self.D_loss = -tf.reduce_mean(0.5 * tf.nn.tanh(self.d_real)) + tf.reduce_mean(self.d_fake)
        self.G_loss = -tf.reduce_mean(self.d_fake)

    def forward_kl(self):
        self.D_loss = -tf.reduce_mean(self.d_real) + tf.reduce_mean(tf.exp(self.d_fake-1))
        self.G_loss = -tf.reduce_mean(tf.exp(self.d_fake-1))

    def reverse_kl(self):
        self.D_loss = -tf.reduce_mean(-tf.exp(self.d_real)) + tf.reduce_mean(-1 - self.d_fake)
        self.G_loss = -tf.reduce_mean(-1 - self.d_fake)

    def pearson_chi_squared(self):
        self.D_loss = -tf.reduce_mean(self.d_real) + tf.reduce_mean(0.25*self.d_logit_fake**2 + self.d_logit_fake)
        self.G_loss = -tf.reduce_mean(0.25*self.d_logit_fake**2 + self.d_logit_fake)

    def squared_hellinger(self):
        self.D_loss = -tf.reduce_mean(1 - tf.exp(-self.d_real)) + tf.reduce_mean(self.d_fake / (1-self.d_fake))
        self.G_loss = -tf.reduce_mean(self.d_fake / (1-self.d_fake))

    def least_squared(self):
        self.D_loss = 0.5 * (tf.reduce_mean((self.d_real - 1) ** 2) + tf.reduce_mean(self.d_fake ** 2))
        self.G_loss = 0.5 * tf.reduce_mean((self.d_fake - 1) ** 2)

    def wassertain(self):
        self.G_loss = tf.contrib.gan.losses.wargs.wasserstein_generator_loss(self.d_fake)
        self.D_loss = tf.contrib.gan.losses.wargs.wasserstein_discriminator_loss(self.d_real, self.d_fake)

    def modified(self):
        self.G_loss = tf.contrib.gan.losses.wargs.modified_generator_loss(self.d_fake)
        self.D_loss = tf.contrib.gan.losses.wargs.modified_discriminator_loss(self.d_real, self.d_fake)


Divergence_Fs = {"Standard_Divergence": GAN.standard_divergence, "Forward_KL": GAN.forward_kl,
                 "Reverse_KL": GAN.reverse_kl, "Pearson_Chi_squared": GAN.pearson_chi_squared,
                 "Squared_Hellinger": GAN.squared_hellinger, "Least_squared": GAN.least_squared,
                 "Modified": GAN.modified, "Wassertain": GAN.wassertain}
