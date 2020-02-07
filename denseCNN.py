from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Conv2DTranspose, Reshape, Activation 
from keras.models import Model
from keras import backend as K
import numpy as np
import json

class denseCNN:
    def __init__(self,name='',weights_f=''):
        self.name=name
        self.pams ={
            'CNN_layer_nodes'  : [8],  #n_filters
            'CNN_kernel_size'  : [3],
            'CNN_pool'         : [False],
            'Dense_layer_nodes': [], #does not include encoded layer
            'encoded_dim'      : 12,
            'shape'            : (1,4,4),
            'channels_first'   : False,
            'arrange'          : [],
            'arrMask'          : [],
        }
        self.weights_f =weights_f
        

    def setpams(self,in_pams):
        for k,v in in_pams.items():
            self.pams[k] = v
            
    def prepInput(self,normData):
      shape = self.pams['shape']
      
      if len(self.pams['arrange'])>0:
          arrange = self.pams['arrange']
          inputdata = normData[:,arrange]
      if len(self.pams['arrMask'])>0:
          arrMask = self.pams['arrMask']
          inputdata[:,arrMask==0]=0  #zeros out repeated entries

      shaped_data = inputdata.reshape(len(inputdata),shape[0],shape[1],shape[2])

      return shaped_data
            
    def init(self,printSummary=True):
        encoded_dim = self.pams['encoded_dim']

        CNN_layer_nodes   = self.pams['CNN_layer_nodes']
        CNN_kernel_size   = self.pams['CNN_kernel_size']
        CNN_pool          = self.pams['CNN_pool']
        Dense_layer_nodes = self.pams['Dense_layer_nodes'] #does not include encoded layer
        channels_first    = self.pams['channels_first']

        inputs = Input(shape=self.pams['shape'])  # adapt this if using `channels_first` image data format
        x = inputs

        for i,n_nodes in enumerate(CNN_layer_nodes):
            if channels_first:
              x = Conv2D(n_nodes, CNN_kernel_size[i], activation='relu', padding='same',data_format='channels_first')(x)
            else:
              x = Conv2D(n_nodes, CNN_kernel_size[i], activation='relu', padding='same')(x)
            if CNN_pool[i]:
              if channels_first:
                x = MaxPooling2D((2, 2), padding='same',data_format='channels_first')(x)
              else:
                x = MaxPooling2D((2, 2), padding='same')(x)

        shape = K.int_shape(x)

        x = Flatten()(x)

        #encoder dense nodes
        for n_nodes in Dense_layer_nodes:
            x = Dense(n_nodes,activation='relu')(x)

        encodedLayer = Dense(encoded_dim, activation='relu',name='encoded_vector')(x)

        # Instantiate Encoder Model
        self.encoder = Model(inputs, encodedLayer, name='encoder')
        if printSummary:
          self.encoder.summary()

        encoded_inputs = Input(shape=(encoded_dim,), name='decoder_input')
        x = encoded_inputs

        #decoder dense nodes
        for n_nodes in Dense_layer_nodes:
             x = Dense(n_nodes, activation='relu')(x)

        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        for i,n_nodes in enumerate(CNN_layer_nodes):
            if CNN_pool[i]:
              if channels_first:
                  x = UpSampling2D((2, 2),data_format='channels_first')(x)
              else:
                  x = UpSampling2D((2, 2))(x)
            if channels_first:
              x = Conv2DTranspose(n_nodes, CNN_kernel_size[i], activation='relu', padding='same',data_format='channels_first')(x)
            else:
              x = Conv2DTranspose(n_nodes, CNN_kernel_size[i], activation='relu', padding='same')(x)

        if channels_first:
          #shape[0] will be # of channel
          x = Conv2DTranspose(filters=self.pams['shape'][0],kernel_size=3,padding='same',data_format='channels_first')(x)
        else:
          x = Conv2DTranspose(filters=1,kernel_size=3,padding='same')(x)

        outputs = Activation('sigmoid', name='decoder_output')(x)


        self.decoder = Model(encoded_inputs, outputs, name='decoder')
        if printSummary:
          self.decoder.summary()

        self.autoencoder = Model(inputs, self.decoder(self.encoder(inputs)), name='autoencoder')
        if printSummary:
          self.autoencoder.summary()

        self.autoencoder.compile(loss='mse', optimizer='adam')
        self.encoder.compile(loss='mse', optimizer='adam')

        CNN_layers=''
        if len(CNN_layer_nodes)>0:
            CNN_layers += '_Conv'
            for i,n in enumerate(CNN_layer_nodes):
                CNN_layers += f'_{n}x{CNN_kernel_size[i]}'
                if CNN_pool[i]:
                    CNN_layers += 'pooled'
        Dense_layers = ''
        if len(Dense_layer_nodes)>0:
            Dense_layers += '_Dense'
            for n in Dense_layer_nodes:
                Dense_layers += f'_{n}'

        self.name = f'Autoencoded{CNN_layers}{Dense_layers}_Encoded_{encoded_dim}'
        
        if not self.weights_f=='':
            self.autoencoder.load_weights(self.weights_f)
    def get_models(self):
       return self.autoencoder,self.encoder
           
    def predict(self,x):
        decoded_Q = self.autoencoder.predict(x)
        encoded_Q = self.encoder.predict(x)
        s = self.pams['shape'] 
        decoded_Q = np.reshape(decoded_Q,(len(decoded_Q),s[0],s[1]))
        encoded_Q = np.reshape(encoded_Q,(len(encoded_Q),self.pams['encoded_dim'],1))
        return decoded_Q, encoded_Q

    def summary(self):
      self.encoder.summary()
      self.decoder.summary()
      self.autoencoder.summary()

    ##get pams for writing json
    def get_pams(self):
      jsonpams={}
      for k,v in self.pams.items():
          if type(v)==type(np.array([])):
              jsonpams[k] = v.tolist()
          else:
              jsonpams[k] = v 
      return jsonpams   
      
