# PatchTST for Transformer-Based Preventative Maintenance #

Artificial Neural Networks of various kinds are used for preventative maintenance. Generally, it is the job of these neural networks to convert the current performance of a machine into a latent space which can then be used as an indicator of machine performance. In this project, in place of a simple embedding layer or the more commonly used GRU, we have elected to use a transformer encoder to produce fix-length embeddings, due to their ability to gather more semantic information over longer timespans than a recurrent network. 

The transformer model was trained using random masking over the first 100,000 samples of a dataset which measured the degradation of a cutting blade used in a Vega Shrink-Wrapping machine. This model then performed inference over these "like-new" samples, and the final hidden layer outputs of the inferences were stored and turned into a KNN containing 199 neurons. Then, the other 900,000 samples of the data were evaluated using this hidden layer analysis, and the distance of these test sequences were measured from the known "good" KNN, and plotted.

Training vs Validation Loss Curve : 

Outputs for the remaining 900k samples : 

This project is inspired by the work from the following paper : 

von Birgelen, Alexander & Buratti, Davide & Mager, Jens & Niggemann, Oliver. (2018). Self-Organizing Maps for Anomaly Localization and Predictive Maintenance in Cyber-Physical Production Systems. Procedia CIRP. 72. 480-485. 10.1016/j.procir.2018.03.150. 

Link : 

https://www.researchgate.net/publication/326038342_Self-Organizing_Maps_for_Anomaly_Localization_and_Predictive_Maintenance_in_Cyber-Physical_Production_Systems
