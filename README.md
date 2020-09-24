# Document_IE

This repo use GCN to extract entities from semi-structured document.

The main refrence : https://github.com/tkipf/gcn.   

For quickly verify GCN's powerful ability of adjoining neighbours, I only use word bag to express a single span currently which has the position cordinate and text content by OCR in one document. we can also deal a single span sentence with lstm or transformer and use the ouput to express a span as the aftering gcn input.  

I'm adding lstm/transformer+GCN method for box level predict on sroie infomation extract task3. I will upload it soon.
