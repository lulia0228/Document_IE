# Document_IE

This repo uses GCN to extract entities from semi-structured document.     

The main reference : https://github.com/tkipf/gcn.     

# Details
For quickly verify GCN's powerful ability of adjoining neighbours, I use lstm to express a span sentence currently and gcn to predict span label for box level prediction on SROIE infomation extract task3.       

Indeed, we can change the structure to predict entity directly like the traditional NER task.        

Before training, we need to use grapher.py to trans a receipt into graph structure and obtain the adjacent matrix and span vector for gcn. The following picture shows graph connections on a receipt.   

<div align="center">
    <img src="https://github.com/lulia0228/Document_IE/blob/master/graph/grapher_outputs/graph_X51006401836.png" width="450px">
</div>   


Unfortunately, my net structure only supports 1 batch training now and I will add real batch training when I have spare time. It needs to add virtual nodes(spans) on adjacent matrix and pad span into fixed lenth, then the input can be (batch, num_span, sentence_len) format.     

# Result 
The train data contains 626 scanned receipts with some labeling errors and the test data contains 347 scanned receipts. 

It takes about 20 seconds for training a epoch on one GTX1080 GPU and occupies only 900MB memory.     

My box level prediction on SROIE infomation extract task3 is as follows:    

                           |  precision |  recall  | f1-score   
               ————————————|————————————|——————————|——————————
               COMPANY     |  0.91      |   0.90   |   0.90  
               ————————————|————————————|——————————|——————————
               ADDRESS     |  0.91      |   0.96   |   0.93    
               ————————————|————————————|——————————|——————————
               DATE        |  0.89      |   0.95   |   0.92       
               ————————————|————————————|——————————|——————————
               TOTAL       |  0.78      |   0.76   |   0.77    
      
# Better relation work 
I have also tried "Pick model"  as described in paper: (https://arxiv.org/abs/2004.07464). The result performes better in all entities but the "total" entity also performed not so well. It needs at least 10GB GPU memory and costs more time to train.   

Another model is "LayoutLM" (https://arxiv.org/abs/1912.13318) which combines bert pretrained model and trained and released by microsoft. It also occupies huge GPU memory and needs to train for a long time.   

