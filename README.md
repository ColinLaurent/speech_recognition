# speech_recognition

Audio classification on the SPEECHCOMMANDS dataset. 
The Deep Neural Network model contains 3 Convolutionnal layers, including Normalization, MaxPooling, Dropout with a classification part containing 2 Fully-connected layers. The model is applied on the MelSpectrograms of the wavefiles.
The performance are optimized using regularization, data augmentation and early stopping.