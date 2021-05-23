# Sequence-to-Sequence-Model
The code will starts from main_htr file from mian function.
Data loading is done in loadData2_vgg file. 
In a main function a seqto seq method is called from the seq2seq_htr file. where the input will goes to encoder first and then decoder. 
In decoder we are calculating attention for every time step. 
The output from the decoder will be match with the xpected output to calculate the loss in main function using Cross entopy loss. 
In Residual lstm code is taken from  https://github.com/kdgutier/residual_lstm/blob/master/residual_lstm.py
The revese lstm code is taken from https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
