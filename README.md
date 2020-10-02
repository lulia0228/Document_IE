# Document_IE

This repo use GCN to extract entities from semi-structured document.

The main refrence : https://github.com/tkipf/gcn.   

For quickly verify GCN's powerful ability of adjoining neighbours, I use lstm to express a span sentence currently and gcn to predict span label for box level prediction on sroie infomation extract task3.     

Indeed, we can change the structure to predict entity directly like the traditional NER task.     

# Result 
The train data contains 626 scanned recipts with some labeling errors and the test data contains 347 recipts.   
The model costs about 20 seconds training one epoch on a GTX1080 Gpu.
My box level prediction on sroie infomation extract task3 is as fllows:

                            |  precision |  recall  | f1-score   
                ————————————|————————————|——————————|——————————
                COMPANY     |  0.91      |   0.90   |   0.90  
                ————————————|————————————|——————————|——————————
                ADDRESS     |  0.91      |   0.96   |   0.93    
                ————————————|————————————|——————————|——————————
                DATE        |  0.89      |   0.95   |   0.92       
                ————————————|————————————|——————————|——————————
                TOTAL       |  0.78      |   0.76   |   0.77    
      
