"""
For training EMD_CNN with different hyperparameters
@author: Rohan
"""
import emd_loss_cnn #Script for training the CNN to approximate EMD
import pandas as pd
import os
import numpy as np
import argparse
from utils.logger import _logger

parser = argparse.ArgumentParser()
parser.add_argument('-i',"--inputFile", type=str, default='nElinks_5/', dest="inputFile",
                    help="input TSG files")
parser.add_argument("--epochs", type=int, default = 64, dest="num_epochs",
                    help="number of epochs to train")
parser.add_argument("--bestEMD", type=int, default = 8, dest="best_num",
                    help="number of emd_models to save")
parser.add_argument("--nELinks", type=int, default = 5, dest="nElinks",
                    help="n of active transceiver e-links eTX")

parser.add_argument("--skipPlot", action='store_true', default=False, dest="skipPlot",
                    help="skip the plotting step")
parser.add_argument("--full", action='store_true', default = False,dest="full",
                    help="run all algorithms and metrics")

parser.add_argument("--quickTrain", action='store_true', default = False,dest="quickTrain",
                    help="train w only 5k events for testing purposes")
parser.add_argument("--retrain", action='store_true', default = False,dest="retrain",
                    help="retrain models even if weights are already present for testing purposes")
parser.add_argument("--evalOnly", action='store_true', default = False,dest="evalOnly",
                    help="only evaluate the NN on the input sample, no train")

parser.add_argument("--double", action='store_true', default = False,dest="double",
                    help="test PU400 by combining PU200 events")
parser.add_argument("--overrideInput", action='store_true', default = False,dest="overrideInput",
                    help="disable safety check on inputs")
parser.add_argument("--nCSV", type=int, default = 1, dest="nCSV",
                    help="n of validation events to write to csv")
parser.add_argument("--maxVal", type=int, default = -1, dest="maxVal",
                    help="clip outputs to maxVal")
parser.add_argument("--AEonly", type=int, default=1, dest="AEonly",
                    help="run only AE algo")
parser.add_argument("--rescaleInputToMax", action='store_true', default=False, dest="rescaleInputToMax",
                    help="rescale the input images so the maximum deposit is 1. Else normalize")
parser.add_argument("--rescaleOutputToMax", action='store_true', default=False, dest="rescaleOutputToMax",
                    help="rescale the output images to match the initial sum of charge")
parser.add_argument("--nrowsPerFile", type=int, default=500, dest="nrowsPerFile",
                    help="load nrowsPerFile in a directory")
parser.add_argument("--occReweight", action='store_true', default = False,dest="occReweight",
                    help="train with per-event weight on TC occupancy")

parser.add_argument("--maskPartials", action='store_true', default = False,dest="maskPartials",
                    help="mask partial modules")
parser.add_argument("--maskEnergies", action='store_true', default = False,dest="maskEnergies",
                    help="Mask energy fractions <= 0.05")
parser.add_argument("--saveEnergy", action='store_true', default = False,dest="saveEnergy",
                    help="save SimEnergy from input data")
parser.add_argument("--noHeader", action='store_true', default = False,dest="noHeader",
                    help="input data has no header")

parser.add_argument("--models", type=str, default="8x8_c8_S2_tele", dest="models",
                    help="models to run, if empty string run all")

def main(args):

    data=[]
    
    def load_data(args):
        # charge data headers of 48 Input Trigger Cells (TC) 
        CALQ_COLS = ['CALQ_%i'%c for c in range(0, 48)]

        def mask_data(data,args):
            # mask rows where occupancy is zero
            mask_occupancy = (data[CALQ_COLS].astype('float64').sum(axis=1) != 0)
            data = data[mask_occupancy]

            if args.maskPartials:
                mask_isFullModule = np.isin(data.ModType.values,['FI','FM','FO'])
                _logger.info('Mask partial modules from input dataset')
                data = data[mask_isFull]
            if args.maskEnergies:
                try:
                    mask_energy = data['SimEnergyFraction'].astype('float64') > 0.05
                    data = data[mask_energy]
                except:
                    _logger.warning('No SimEnergyFraction array in input data')
            return data

        if os.path.isdir(args.inputFile):
            df_arr = []
            for infile in os.listdir(args.inputFile):
                if os.path.isdir(args.inputFile+infile): continue
                infile = os.path.join(args.inputFile,infile)
                if args.noHeader:
                    df_arr.append(pd.read_csv(infile, dtype=np.float64, header=0, nrows = args.nrowsPerFile, usecols=[*range(0,48)], names=CALQ_COLS))
                else:
                    df_arr.append(pd.read_csv(infile, nrows=args.nrowsPerFile))
            data = pd.concat(df_arr)
        else:
            data = pd.read_csv(args.inputFile, nrows=args.nrowsPerFile)
        data = mask_data(data,args)

        if args.saveEnergy:
            try:
                simEnergyFraction = data['SimEnergyFraction'].astype('float64') # module simEnergyFraction w. respect to total event's energy
                simEnergy = data['SimEnergyTotal'].astype('float64') # module simEnergy
                simEnergyEvent = data['EventSimEnergyTotal'].astype('float64') # event simEnergy
            except:
                simEnergyFraction = None
                simEnergy = None
                simEnergyEvent = None
                _logger.warning('No SimEnergyFraction or SimEnergyTotal or EventSimEnergyTotal arrays in input data')

        data = data[CALQ_COLS].astype('float64')
        data_values = data.values
        _logger.info('Input data shape')
        print(data.shape)
        data.describe()

        # duplicate data (e.g. for PU400?)
        if args.double:
            def double_data(data):
                doubled=[]
                i=0
                while i<= len(data)-2:
                    doubled.append( data[i] + data[i+1] )
                    i+=2
                return np.array(doubled)
            doubled_data = double_data(data_values.copy())
            _logger.info('Duplicated the data, the new shape is:')
            print(doubled_data.shape)
            data_values = doubled_data

        return data_values
    
    data=load_data(args)

    current_directory=os.getcwd()

    #Data to track the performance of various CNN models

    df=[]
    mean_data=[]
    std_data=[]
    nfilt_data=[]
    ksize_data=[]
    neuron_data=[]
    numlayer_data=[]
    convlayer_data=[]
    epoch_data=[]
    z_score=[]

    #List of lists of Hyperparamters <- currently initialized from previous training
    hyp_list=[[32,5,256,1,3],
              [32,5,32,1,4],
              [64,5,32,1,4],
              [128,5,32,1,4],
              [128,5,64,1,3],
              [32,5,128,1,3],
              [128,3,256,1,4],
              [128,5,256,1,4]
             ]
    
    num_epochs=args.num_epochs
    best_num=args.best_num

    #Best EMD Models Per SD

    best=[[0,0,0,0,0,0,0]]*(best_num)     

    for hyp in hyp_list:
        num_filt=hyp[0]
        kernel_size=hyp[1]
        num_dens_neurons=hyp[2]
        num_dens_layers=hyp[3]
        num_conv_2d=hyp[4]
        
        #Each model per set of hyperparamters is trained thrice to avoid bad initialitazion discarding a good model. (We vary num_epochs by 1 to differentiate between these 3 trainings)
        
        for i in [0,1,2]:
            obj=emd_loss_cnn.EMD_CNN()
            mean, sd = obj.ittrain(data, num_filt,kernel_size, num_dens_neurons, num_dens_layers, num_conv_2d,num_epochs+i)
            mean_data.append(mean)
            std_data.append(sd)
            nfilt_data.append(num_filt)
            ksize_data.append(kernel_size)
            neuron_data.append(num_dens_neurons)
            numlayer_data.append(num_dens_layers)
            convlayer_data.append(num_conv_2d)
            epoch_data.append(num_epochs+i)
            z=abs(mean)/sd
            z_score.append(z)
            
            #The best 8 models are saved in a list, as [sd, num_filt,kernel_size,num_dens_neurons,num_dens_layers,num_conv_2d,num_epochs+i]
            #We rank the models per their standard deviation(sd), ie model[0] 

            max=0;
            for j in range(0,best_num):
                model=best[j]
                maximum=best[max]
                if model[0]>maximum[0]:
                    max=j
            best[max]=[sd,num_filt,kernel_size, num_dens_neurons, num_dens_layers, num_conv_2d,num_epochs+i]


    for_pdata=[mean_data,std_data,nfilt_data,ksize_data,neuron_data,numlayer_data,convlayer_data,z_score,epoch_data]

    #Saving data from the entire optimization training 
    
    opt_data_directory=os.path.join(current_directory,r'EMD_Loss CNN Optimization Data.xlsx')
    df=pd.DataFrame(for_pdata)
    df.to_excel(opt_data_directory)
    
    #Saving another .xlsx for the best models    
    best_data_directory=os.path.join(current_directory,r'Best EMD_CNN Models.xlsx')

    df=pd.DataFrame(best)
    df.to_excel(best_data_directory)

    #Preparing Best EMD Loss Models for ECON Training as {i}.h5, which during CNN optimization were saved as: 
    #(num_filt)+(kernel_size)+(num_dens_neurons)+(num_dens_layers)+(num_conv_2d)+(num_epochs)+best.h5 
    # and Renaming them to {i}.h5, i={1,2,...8}

    model_directory=os.path.join(os.getcwd(),r'emd_loss_models')
    i=1
    for models in best:
        mpath=os.path.join(model_directory,str(models[1])+str(models[2])+str(models[3])+str(models[4])+str(models[5])+str(models[6])+'best.h5')
        best_path=os.path.join(os.getcwd(),'best_emd')
        os.mkdir(best_path)
        os.rename(mpath,os.getcwd()+'/best_emd/'+str(i)+'.h5')
        i+=1

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

