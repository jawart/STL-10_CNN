# No 1
# data_description function
# author or code source: Artur Jaworowski

def data_description(train_x, train_y, valid_x, valid_y, test_x, test_y, convert):

    # shows input and output data of data_conversion function

    import pandas as pd
    
    d_set = [train_x, train_y, valid_x, valid_y, test_x, test_y]
    
    idx = ['TRAIN_X', 'TRAIN_Y', 'VALID_X', 'VALID_Y', 'TEST_X', 'TEST_Y']
    
    data_list = list()
    
    for i in range(0,6):
    
        data_descr = dict()
        
        data_descr['set name'] = idx[i]
       
        data_descr['shape'] = d_set[i].shape   
    
        data_descr['dtype'] = d_set[i].dtype
    
        if i%2 == 0:
            data_descr['data sight'] = d_set[i][25,0,0]
        else:
            data_descr['data sight'] = d_set[i][25]
            
        data_list.append(data_descr) 

    if convert == True:
      
    	print(pd.DataFrame(data_list))

    return data_list

# ###########################################################################

# No 2
# data_conversion function
# author or code source: Artur Jaworowski

def data_conversion(train_x_name, train_y_name):
    
   import numpy as np
   import warnings

   warnings.filterwarnings('ignore')
    
   # prepared data loading

   train_x_file = './Used_data/' + train_x_name
   train_y_file = './Used_data/' + train_y_name

   TRAIN_IMAGES = np.load(train_x_file)
   TRAIN_LABELS = np.load(train_y_file)
   VALID_IMAGES = np.load('./Used_data/VALID_IMAGES.npy')
   VALID_LABELS = np.load('./Used_data/VALID_LABELS.npy')
   TEST_IMAGES = np.load('./Used_data/TEST_IMAGES.npy')
   TEST_LABELS = np.load('./Used_data/TEST_LABELS.npy')

   print() 
   print('LOADED DATA (before conversion):')

   data_list = data_description(TRAIN_IMAGES, TRAIN_LABELS, VALID_IMAGES, VALID_LABELS, TEST_IMAGES, TEST_LABELS, convert=True)  

   # convert from integers to floats
    
   TRAIN_X = TRAIN_IMAGES.astype('float32')
   VALID_X = VALID_IMAGES.astype('float32')
   TEST_X = TEST_IMAGES.astype('float32')
    
   # normalize to range 0-1
    
   TRAIN_X = TRAIN_X / 255.0
   VALID_X = VALID_X / 255.0
   TEST_X = TEST_X / 255.0
    
   # to categorical
   
 
   from keras.utils import to_categorical

   TRAIN_Y = to_categorical(TRAIN_LABELS)
   VALID_Y = to_categorical(VALID_LABELS)
   TEST_Y = to_categorical(TEST_LABELS)



   # removing of the first column

   TRAIN_Y = np.delete(TRAIN_Y, 0, axis=1)
   VALID_Y = np.delete(VALID_Y, 0, axis=1)
   TEST_Y = np.delete(TEST_Y, 0, axis=1)

   print() 
   print('DATA FOR TRAINING (after conversion):')

   data_list = data_description(TRAIN_X, TRAIN_Y, VALID_X, VALID_Y, TEST_X, TEST_Y, convert=True)

    
   return TRAIN_X, VALID_X, TEST_X, TRAIN_Y, VALID_Y, TEST_Y

# ###########################################################################

# No 3
# model_save function
# author or code source: Artur Jaworowski

def model_save(model, my_model_name):
     
    import os
    
    # shows already saved models
    
    existing_models = list(os.listdir('./Models'))
    
    if '.ipynb_checkpoints' in existing_models:
        existing_models.remove('.ipynb_checkpoints')  
          
    # saves new model
        
    my_model_name = my_model_name + '.h5'    
    
    
    if my_model_name in existing_models:
        print()
        print('Such model already exists')
        
    else:        
        location = './Models/' + my_model_name
        model.save(location)

# ###########################################################################

# No 4
# training_stl10 function
# author or code source: Artur Jaworowski

def training_stl10(batch_size, epochs, model, optimizer,  
                   TRAIN_X, TRAIN_Y, VALID_X, VALID_Y, trd_model_name):

    # model compilation and training
    
    # global history_dict
    
    import datetime
    time_start = datetime.datetime.now()

    
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=optimizer)

    history = model.fit(TRAIN_X, TRAIN_Y, batch_size=batch_size, epochs=epochs, 
                        validation_data = (VALID_X, VALID_Y), verbose=1)

    score = model.evaluate(VALID_X, VALID_Y, verbose=1)
    
    history_dict = history.history
    
    print(history_dict.keys())
    
    print('Validation accuracy', score[1]) 

    time_stop = datetime.datetime.now()

    print('')
    print('Execution time:  ',time_stop-time_start)
    
    return history_dict

# ###########################################################################

# No 5
# results_chart function
# author or code source: Artur Jaworowski based on Francois' Chollet code
    
def results_chart(history_dict):
    
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    
    import matplotlib.pyplot as plt

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

# ###########################################################################

# No 6
# results_update function
# author or code source: Artur Jaworowski, based on web available code

'''
Checks whether results file exists, if not creates it and invokes other 
function for data writing.
'''

def results_update(model_name, model_descr, notebook, dataset_size, 
                   batch_size, epochs, optimizer_name, acc, remark):

    try:
        with open('results_summary.txt') as file:
        
            add_result(model_name, model_descr, notebook, dataset_size, 
                       batch_size, epochs, optimizer_name, acc, remark)
        
    except IOError:
    
        import simplejson
        
        results_summary_list = []
        
        f = open('results_summary.txt', 'w')
        simplejson.dump(results_summary_list, f)
        f.close()
    
        add_result(model_name, model_descr, notebook, dataset_size, 
                   batch_size, epochs, optimizer_name, acc, remark)

# ###########################################################################

# No 7
# add_result function
# author or code source: Artur Jaworowski

def add_result(model_name, model_descr, notebook, dataset_size, 
                            batch_size, epochs, optimizer_name, acc, remark):
    
    # Adds to the project results file the result of the model's training
    
    single_result = dict()
       
    single_result['model_name'] = model_name   
    
    single_result['model_description'] = model_descr

    single_result['notebook'] = notebook

    single_result['dataset_size'] = dataset_size

    single_result['batch_size'] = batch_size
    
    single_result['epochs'] = epochs 
    
    single_result['optimizer'] = optimizer_name
    
    single_result['acc'] = acc
    
    single_result['remarks'] = remark
    
    import simplejson
 
    results_summary_list = results_preview()
     
    results_summary_list.append(single_result)
    
    f = open('results_summary.txt', 'w')
    simplejson.dump(results_summary_list, f)
    f.close()

# ###########################################################################

# No 8
# results_preview function
# author or code source: Artur Jaworowski

def results_preview():
    
    import pandas as pd
        
    try:
        with open('results_summary.txt', 'r') as f:
            
            import simplejson
    
            f = open('results_summary.txt', 'r')
            file = simplejson.load(f)
        
        return file
        
    except IOError:
    
        print('Not any results yet')
        print('You are starting first training, start function ????????')

# ###########################################################################

# No 9
# colors_scale function 
# author or code source: Artur Jaworowski

def colors_scale():

    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    x = np.zeros((1, 11))
    x[:] = np.arange(0,11)

    t = ['min values or nan','',' ','',' ','',' ','',' ','','max values']
    ti = list(range(len(t)))

    fig = plt.imshow(x)
    fig.set_cmap('viridis')
    fig.axes.get_xaxis().set_visible(True)
    fig.axes.get_yaxis().set_visible(False)
    plt.xticks(ti, t)
    del(x)

# ###########################################################################

# No 10
# activations function 
# author or code source: Artur Jaworowski

def activations(model, n_layers, img_tensor): 

    from keras import models

    # The output of the top n layers:

    # convolutional and max_pooling layers in this case

    layer_outputs = [layer.output for layer in model.layers[:n_layers]]

    # Creation a model that returns the output generated by passing the specified inputs.

    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    # Returning a list of Numpy arrays: one array for each activation layer.

    activations = activation_model.predict(img_tensor)

    return (activations)

# ###########################################################################

# No 11
# activation_channels function
# author or code source: Francois Chollet

def activation_channels(model, layers_range, img_tensor):

    import numpy as np
    import matplotlib.pyplot as plt

    activ = activations(model, layers_range, img_tensor)

    colors_scale()

    # Adds layer names.
    layer_names = []
    for layer in model.layers[:layers_range]:
        layer_names.append(layer.name)

    images_per_row = 16

    # Feature maps display.
    for layer_name, layer_activation in zip(layer_names, activ):
        # The number of features on the map.
        n_features = layer_activation.shape[-1]

        # The feature map has the shape (1, size, size, n_features).
        size = layer_activation.shape[1]

        # Creates activation channel tiles in this matrix.
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # Creates a horizontal grid of filters.
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
                # Processing a feature to generate a readable visualization.
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image

        # Displays the grid.
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
    plt.show()

# ###########################################################################

# No 12
# activations_preview function
# author or code source: Artur Jaworowski

def activations_preview(layer, model_0, model, layers_range, img_tensor, chanels_range):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm 
    import numpy as np

    activs_0 = activations(model_0, layers_range, img_tensor)
    activs = activations(model, layers_range, img_tensor)
    activation_0 = activs_0[layer]
    activation = activs[layer]

    for i in range(chanels_range[0], chanels_range[1]):
        
        print('_'*105)
        print()
        print('Layer', layer, 'activations, ', 'channel ', i)
        print()
        print('Min. value of activation after training:   ', np.min(activation[0, :, :, i]))
        print('It\'s coordinates on the picture as (y, x): ',
              np.unravel_index(activation[0, :, :, i].argmin(), activation[0, :, :, i].shape))
        print()
        print('Max. value of activation after training:   ',np.max(activation[0, :, :, i]))
        print('It\'s coordinates on the picture as (y, x): ',
              np.unravel_index(activation[0, :, :, i].argmax(), activation[0, :, :, i].shape))
    
        lista = [img_tensor[0], activation_0[0, :, :, i], activation[0, :, :, i]]

        f, axarr = plt.subplots(1,3, figsize=(15, 0.3))
        plt.rc('font', size=14)
        axarr[0].axis('off')
        axarr[0].text(0.4,0, 'Original image')
        axarr[1].axis('off')
        axarr[1].text(0.4,0, 'Before training')
        axarr[2].axis('off')
        axarr[2].text(0.4,0, 'After training')
    
        f, axarr = plt.subplots(1,3, figsize=(15, 15))
        axarr[0].imshow(img_tensor[0])
        axarr[1].matshow(lista[1], cmap='viridis')
        axarr[2].matshow(lista[2], cmap='viridis')
    
        plt.show()

    print('_'*105)

# ###########################################################################

# No 13
# # prime_factorization function
# author or code source: generally available

# decomposition dims_product into prime factors

def prime_factorization(dims_product):
    factors = []
    k = 2
    while dims_product != 1:
        while dims_product % k == 0:
            dims_product //= k
            factors.append(k)
        k += 1
    return factors

# ###########################################################################

# No 14
# picture_format function
# author or code source: Artur Jaworowski

# optimization of reckangles dimensions

def picture_format(dims_product):
    
    import numpy as np
    
    a = prime_factorization(dims_product)
    
    if len(a) % 2 > 0:
        i = len(a)//2+1
    else:
        i = len(a)//2
        
    dim_a = np.prod(a[:i])
    dim_b = np.prod(a[i:])
    
    return(dim_a, dim_b)

# ###########################################################################

# No 15
# weights_visualization function
# author or code source: Artur Jaworowski

# it requires keras model

def weights_visualization(selected_layers, selected_models, size_X, size_Y):
    
    colors_scale() 
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    print('All layers in this model:')
    print('')
    for i in range (len(selected_models[0].layers)):
        print('Layer', i,':' , selected_models[0].layers[i].name)
    

    
    for model in selected_models:        
        for i_layer in selected_layers:
            weights = model.layers[i_layer].get_weights() 
            w = weights[0].T
    
            dims_product = w.shape[0]
    
            neurons_grid = picture_format(dims_product)
     
            dims_product = w.shape[1] * w.shape[2] * w.shape[3]

            neuron_pic_dims = tuple(np.sort(picture_format(dims_product)))
            
            print('_'*80)
            print('')
            print('Layer',i_layer,model.layers[i_layer].name)
            print('Neurons:',w.shape[0])
            print('Chanels per neuron:', dims_product)
            print('Neurons\' picture dimensions:', neuron_pic_dims)
        
            if i_layer == 0:
                fsz0 = size_X
                fsz1 = size_Y
            else:
                fsz0 = 17
                fsz1 = neuron_pic_dims[0] + 3
    
            fig = plt.figure(figsize=(fsz0, fsz1))
  
            w = weights[0].T          
            for neuron in range(w.shape[0]):         
                ax = fig.add_subplot(neurons_grid[0], neurons_grid[1], neuron+1)
                ax.axis("off")
                ax.set_title('N. '+ str(neuron))
                ax.imshow(np.reshape(w[neuron], neuron_pic_dims))
            '''
            if model == model_0:
                pic_name = 'Weights_before_training--layer_' + str(i_layer) + '.png'
            else:
                pic_name = 'Weights_after_training--layer_' + str(i_layer) + '.png'
            
            plt.savefig(pic_name, dpi=300)

	    '''
            
            plt.show()


# ###########################################################################

# No 16
# weights_training_results function
# author or code source: Artur Jaworowski

# it requires keras model

def weights_training_results(selected_layer, selected_models_list, selected_neurons_list):

    import matplotlib.pyplot as plt
    import numpy as np

    for neuron in selected_neurons_list:
    
        w = [0,0]
        i=0
        for model in selected_models_list:
        
            weights = model.layers[selected_layer].get_weights() 
            w[i] = weights[0].T
                 
            dims_product = w[i].shape[1] * w[i].shape[2] * w[i].shape[3]
        
            i += 1
            
        print('Layer:', selected_layer, ' - ', model.layers[selected_layer].name)
        print('Neuron:', neuron)

        neuron_pic_dims = tuple(np.sort(picture_format(dims_product)))
          
        fig = plt.figure(figsize = (16,1))
        #plt.rc('font', size=15)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.axis('off')
        ax1.text(0.1,2, 'Before training')
        ax1.text(0.1,1.5,('Min. weigth\'s value:   ' + str(np.min(w[0][neuron]))))
        ax1.text(0.1,1, ('It\'s coordinates on the picture as (y, x):   ' 
                         + str(np.unravel_index(w[0][neuron].argmin(), 
                               (np.reshape(w[0][neuron], neuron_pic_dims)).shape))))
        ax1.text(0.1,0.5,('Max. weigth\'s value:   ' + str(np.max(w[0][neuron]))))
        ax1.text(0.1,0, ('It\'s coordinates on the picture as (y, x):   ' 
                         + str(np.unravel_index(w[0][neuron].argmax(), 
                               (np.reshape(w[0][neuron], neuron_pic_dims)).shape))))        
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axis('off')
        ax2.text(0.1,2, 'After training') 
        ax2.text(0.1,1.5,('Min. weigth\'s value:   ' + str(np.min(w[1][neuron]))))
        ax2.text(0.1,1, ('It\'s coordinates on the picture as (y, x):   ' 
                         + str(np.unravel_index(w[1][neuron].argmin(), 
                               (np.reshape(w[1][neuron], neuron_pic_dims)).shape))))
        ax2.text(0.1,0.5,('Max. weigth\'s value:   ' + str(np.max(w[1][neuron]))))
        ax2.text(0.1,0, ('It\'s coordinates on the picture as (y, x):   ' 
                         + str(np.unravel_index(w[1][neuron].argmax(), 
                               (np.reshape(w[1][neuron], neuron_pic_dims)).shape)))) 
        
        fig = plt.figure(figsize = (16,16))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(np.reshape(w[0][neuron], neuron_pic_dims))
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(np.reshape(w[1][neuron], neuron_pic_dims)) 
 
        plt.show()
    
        print('_'*105)
        print()

# ###########################################################################

# No 17
# deprocess_image function
# author or code source: Francois Chollet

def deprocess_image(x):

    import numpy as np

    # Tensor normalization: centering on point 0, providing 0.1 standard deviation.
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # Restricted to the range [0, 1].
    x += 0.5
    x = np.clip(x, 0, 1)

    # Convert to RGB array form.
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# ###########################################################################

# No 18
# generate_pattern function
# author or code source: Francois Chollet

def generate_pattern(model, layer_name, filter_index, size):

    from keras import models
    from keras import backend as K
    import numpy as np

    # Builds a loss function that maximizes the activation of the nth filter of the analyzed layer.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Calculates the gradient of the input image considering the loss value.
    grads = K.gradients(loss, model.input)[0]

    # The trick to normalize the gradient.
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # Funkcja zwracająca stratę i gradient dla danego obrazu wejściowego.
    iterate = K.function([model.input], [loss, grads])
    
    # Start with a gray image with noise.
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Performs 40 steps of the gradient increasing algorithm.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    
    return deprocess_image(img)
 
# ###########################################################################  

# No 19
# filters_visualization function
# author or code source: Artur Jaworowski

def filters_visualization(selected_layers, selected_models, columns, size, namespace):

    import matplotlib.pyplot as plt
    import numpy as np
    import datetime    
    time_start = datetime.datetime.now()

    for model in selected_models:

        print('Model: ', namestr(model,namespace)[0])       

        for i_layer in selected_layers:
    
            layer_name = model.layers[i_layer].name
            
            print('Layer: ', i_layer, '-', layer_name)
            print('Filters for this layer: ', model.layers[i_layer].output.shape[3].value)

            filter_index_range = model.layers[i_layer].output.shape[3].value
    
            rows = filter_index_range / columns

            fig = plt.figure(figsize = (17, rows * 2.1))
     
            for filter_index in range(0,filter_index_range):         
                ax = fig.add_subplot(rows, columns, filter_index+1)
                ax.axis("off")
                ax.set_title('Filter ' + str(filter_index))
                ax.imshow(generate_pattern(model, layer_name, filter_index, size))

            # plt.savefig("filter_index_conv2d_1.png", dpi=300)    
            plt.show()

            print('_'*105)
            print()


    time_stop = datetime.datetime.now()
    print('')
    print('Execution time:  ',time_stop-time_start) 
 
# ###########################################################################

# No 20
# filters_zoom function
# author or code source: Artur Jaworowski

# it requires keras model

def filters_zoom(selected_layer, selected_filters_list, selected_models_list, size, namespace):

    import matplotlib.pyplot as plt
    import numpy as np
    import datetime 
    
    time_start = datetime.datetime.now()
    
    for model in selected_models_list:

        print('Model: ', namestr(model, namespace)[0])

        layer_name = model.layers[selected_layer].name                 
     
        for filter_index in selected_filters_list:
                    
            print('layer:',layer_name, ' /  filter_index:', filter_index, ' /  size:', size )
                
            fig = plt.figure(figsize = (8,8))
            plt.imshow(generate_pattern(model, layer_name, filter_index, size))

            # plt.savefig("filter_index_conv2d_1.png", dpi=300)    
            plt.show() 

        print('_'*114)
        print()


    time_stop = datetime.datetime.now()
    print('')
    print('Execution time:  ',time_stop-time_start)

# ###########################################################################

# No 21
# filters_training_results function
# author or code source: Artur Jaworowski

# it requires keras model

def filters_training_results(selected_layer, selected_filters_list, selected_models_list, size, namespace):

    import matplotlib.pyplot as plt
    import numpy as np
    import datetime 
    
    time_start = datetime.datetime.now()
    
    image = [0,0]
    model_name = [0,0]  
  
    for filter_index in selected_filters_list:
        
        print('')
        print('Layer: ', selected_layer)
        print('Filter: ', filter_index)
        
        i_model = 0
        
        for model in selected_models_list:

            model_name[i_model] = namestr(model, namespace)[0] 
                
            layer_name = model.layers[selected_layer].name
            image[i_model] = generate_pattern(model, layer_name, filter_index, size)
                                              
            i_model += 1 
                                              
           
        fig = plt.figure(figsize = (17,0.2))
        plt.rc('font', size = 14)               
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.axis('off')
        ax1.text(0.1,3.5, model_name[0]) 
        ax1.text(0.1,1.5, 'Min. value: ' + str(np.min(image[0]))) 
        ax1.text(0.1,0, 'Max. value: ' + str(np.max(image[0])))  
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axis('off')
        ax2.text(0.1,3.5, model_name[1])
        ax2.text(0.1,1.5, 'Min. value: ' + str(np.min(image[1]))) 
        ax2.text(0.1,0, 'Max. value: ' + str(np.max(image[1])))
                
        fig = plt.figure(figsize = (17,10))
        plt.rc('font', size = 10)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(image[0])
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(image[1])

        # plt.savefig("filter_index_conv2d_1.png", dpi=300)    
        plt.show()         
    
    print('_'*114)
    print()

    time_stop = datetime.datetime.now()
    print('')
    print('Execution time:  ',time_stop-time_start)

# ###########################################################################

# No 22
# namestr function
# author or code source: Stack Overflow

# gets the name of the variable

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

# ########################################################################### 

# No 23
# train_and_chart
# author or code source: Artur Jaworowski

# training code reduction

def train_and_chart(batch_size, epochs, model, optimizer, 
                    TRAIN_X, TRAIN_Y, VALID_X, VALID_Y, trd_model_name):

    history_dict = training_stl10(batch_size, epochs, model, optimizer,  
                                  TRAIN_X, TRAIN_Y, VALID_X, VALID_Y, trd_model_name)

    results_chart(history_dict)

    model_save(model, trd_model_name) # saving model after training !

    acc = history_dict['val_acc'][-1]

    return acc

# ###########################################################################

# No 24
# summary_barplot
# author or code source: Artur Jaworowski

def summary_barplot():

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    rows = len(results_preview())
    dim_y = 0.38 * rows
    plt.figure(figsize=(12, dim_y))
    frm = pd.DataFrame(results_preview())
    #ind = np.array([i+1 for i in range(0,len(frm.index))])
    #ind = np.array([1,2,3],)
    frm = frm[:].values
    #print(frm)
    #print(ind.T)
    sns.set(style="darkgrid")
    palette = sns.color_palette("hls", rows)
    sns.barplot(y=frm[:,0],x=frm[:,7], palette=(palette))
    #sns.barplot(y=ind[:],x=frm[:,7], palette=(palette))
    plt.title('Validation accuracy')
    plt.xlim(0.85,1.0)
    plt.show()

# ###########################################################################

# Code block 1
# as a function only for readability of the 0_Project_summary.ipynb file 

def code_block_1(TEST_X, TEST_Y):

   import keras
   import warnings
   import tensorflow as tf
   warnings.filterwarnings('ignore')
   tf.logging.set_verbosity(tf.logging.ERROR)

   print()
   model_no_33 = keras.models.load_model('./Models/model_3_8-100-30-RMSp-trd.h5')
   test_loss, test_acc = model_no_33.evaluate(TEST_X, TEST_Y)
   print()
   print('Model no. 33','(Module_6)',',', 'validation accuracy = 0.932',',','test accuracy = ',test_acc)
   del(model_no_33, test_loss, test_acc)
   print()
   model_no_37 = keras.models.load_model('./Models/model_4_8-100-40-RMSp-trd.h5')
   test_loss, test_acc = model_no_37.evaluate(TEST_X, TEST_Y)
   print()
   print('Model no. 37','(Module_7)',',', 'validation accuracy = 0.946',',','test accuracy = ',test_acc)
   del(model_no_37, test_loss, test_acc)
   print()
   model_no_46 = keras.models.load_model('./Models/model_6_8-100-30-RMSp-trd.h5')
   test_loss, test_acc = model_no_46.evaluate(TEST_X, TEST_Y)
   print()
   print('Model no. 46', '(Module_15)',',', 'validation accuracy = 0.952',',', 'test accuracy = ',test_acc)
   del(model_no_46, test_loss, test_acc)
   print()
   model_no_56 = keras.models.load_model('./Models/model_7_8-100-5-3vgg16s-trd_a.h5')
   test_loss, test_acc = model_no_56.evaluate(TEST_X, TEST_Y)
   print()
   print('Model no. 56', '(Module_19a)',',', 'validation accuracy = 0.986',',', 'test accuracy = ',test_acc)
   del(model_no_56, test_loss, test_acc)

# ###########################################################################

# Code block 0
# as a function only for readability of the 0_Project_summary.ipynb file

def code_block_0():
    import numpy as np
    import pandas as pd

    TRAIN_X = np.load('./Used_data/TRAIN_X_EXTD.npy')
    TRAIN_Y = np.load('./Used_data/TRAIN_Y_EXTD.npy')
    VALID_X = np.load('./Used_data/VALID_IMAGES.npy')
    VALID_Y = np.load('./Used_data/VALID_LABELS.npy')
    TEST_X = np.load('./Used_data/TEST_IMAGES.npy')
    TEST_Y = np.load('./Used_data/TEST_LABELS.npy')
    return pd.DataFrame(data_description(TRAIN_X, TRAIN_Y, VALID_X, VALID_Y, TEST_X , TEST_Y, convert=False))
    # convert=False to show the data info once only 

    

