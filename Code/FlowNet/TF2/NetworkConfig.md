# Choosing Best ICSTN Warp Architecture: VanillaNet (Model Size <= 25 MB and Model FPS >= 20 FPS on All Cores i7)
- Running on Image Size of 128x128x(3x2)  
- No Data Augmentation on MSCOCO  
- Train on  LR = 1e-4, BatchSize = 32, NumEpochs = 100  

## **BEST ONE!** ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Trans, Trans, Scale, Scale: [100EpochModel](https://drive.google.com/open?id=1WFWTxuDO77i6x8oB3sfzO3mEvcdqBrgX) 
self.InitNeurons = 18  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 43744396604  
NumParams = 2079870  
Expected Model Size = 23.8192596436 MB  
Network Used: Network.VanillaNet3  
Lambda = [1.0, 10.0, 10.0] # [Scale, Translation]  
warpType = ['translation', 'translation', 'scale', 'scale']  

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Scale, Scale, Trans, Trans: [100EpochModel](https://drive.google.com/open?id=1NEQ9gMixBjpzLiUwFsjC6_b7HLP88nm8) 
self.InitNeurons = 18  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 43744396636  
NumParams = 2079870  
Expected Model Size = 23.8192596436 MB  
Network Used: Network.VanillaNet2  
Lambda = [10.0, 1.0, 1.0] # [Scale, Translation]  
warpType = ['scale', 'scale', 'translation', 'translation']  

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Pseudosimilarity x 1: [100EpochModel](https://drive.google.com/open?id=1Pj6Uqr3PeMCJF_vpkd_Nr4ljQ_WfgJiY) 
self.InitNeurons = 36  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 35238910489  
NumParams = 2065935  
Expected Model Size = 23.6512718201 MB  
Network Used: Network.VanillaNet  
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation]  
warpType = ['pseudosimilarity']  

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Pseudosimilarity x 2: [100EpochModel](https://drive.google.com/open?id=1p4UJ1vybf15NSuWK2m--meqbagxpeR9H)
self.InitNeurons = 26  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 40148554316  
NumParams = 2171890  
Expected Model Size = 24.8676147461 MB  
Network Used: Network.VanillaNet  
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation]  
warpType = ['pseudosimilarity', 'pseudosimilarity']    

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Pseudosimilarity x 4: [100EpochModel](https://drive.google.com/open?id=1IhJLQ0rc1mPV6ZrQkFiBFK8Nhkp8hMjl) 
self.InitNeurons = 18  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 43749760630  
NumParams = 2107524  
Expected Model Size = 24.1357345581 MB  
Network Used: Network.VanillaNet  
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation]  
warpType = ['pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity']  

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Scale, Trans [100EpochModel](https://drive.google.com/open?id=1fw3vzSNM0VSy8vz6wXbPqd5l0tbmB_E3) 
self.InitNeurons = 26  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 40144680527  
NumParams = 2151919  
Expected Model Size = 24.6390647888 MB  
Network Used: Network.VanillaNet2Simpler  
Lambda = [10.0, 1.0, 1.0] # [Scale, Translation]  
warpType = ['scale', 'translation']  

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Trans, Scale [100EpochModel](https://drive.google.com/open?id=1PnYX1PXEgZsQ6UUbYeD9uS3OxDzXVX4Y) 
self.InitNeurons = 26  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops =   
NumParams =   
Expected Model Size =  MB  
Network Used: Network.VanillaNet3Simpler  
Lambda = [1.0, 10.0, 10.0] # [Scale, Translation]  
warpType = ['translation', 'scale']  


**Best Combination is 2T2S.**  

# Choosing Best Large Network Architecture (Model Size <= 25 MB and Model FPS >= 20 FPS on All Cores i7)
- Running on Image Size of 128x128x(3x2)  
- No Data Augmentation on MSCOCO  
- Train on  LR = 1e-4, BatchSize = 32, NumEpochs = 100  

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) VanillaNet [50EpochModel](https://drive.google.com/open?id=1IAu6idOuJp0-4TMCpnDYFgIVZ4Xa3Glg) [100EpochModel](https://drive.google.com/open?id=1WFWTxuDO77i6x8oB3sfzO3mEvcdqBrgX) 
self.InitNeurons = 18  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 43744396604  
NumParams = 2079870  
Expected Model Size = 23.8192596436 MB  
Network Used: Network.VanillaNet3  
Lambda = [1.0, 10.0, 10.0] # [Scale, Translation]  
warpType = ['translation', 'translation', 'scale', 'scale'] 

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) ResNet [50EpochModel](https://drive.google.com/open?id=1CVcOdikijaZzUGBUeTwCtMZJ7k2M8Az5) [100EpochModel](https://drive.google.com/open?id=1q2vSRg2_LSkkEkL9X46Lz4TVkKlSrQbY) 
self.InitNeurons = 13  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 55175035536  
NumParams = 2119578  
Expected Model Size = 24.268951416 MB  
Network Used: Network.ResNet3  
Lambda = [1.0, 10.0, 10.0] # [Scale, Translation]  
warpType = ['translation', 'translation', 'scale', 'scale']  


## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) ResNet [50EpochModel](https://drive.google.com/open?id=1jT7kfmUtdMdLisf7851-t1umaKSSDBPv) 
self.InitNeurons = 13  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 55175035536  
NumParams = 2119578  
Expected Model Size = 24.268951416 MB  
Network Used: Network.ResNet3  
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation]  
warpType = ['translation', 'translation', 'scale', 'scale']  

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) SqueezeNet [50EpochModel](https://drive.google.com/open?id=1pWipRE8KVd8vvkTkEXmrvKarg4rQDwvj) [100EpochModel](https://drive.google.com/open?id=126Q_1jvxzSVhUPPkDVTURtEz34wwnLFd)  
self.InitNeurons = 12    
self.ExpansionFactor = 1.2 
self.DropOutRate = 0.7  
self.NumBlocks = 2  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 209774502264  
NumParams = 2120962  
Expected Model Size = 24.2732849121 MB  
Network Used: Network.SqueezeNet3    
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation] at LR 1e-4
warpType = ['translation', 'translation', 'scale', 'scale']  

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) MobileNetv1 [50EpochModel](https://drive.google.com/open?id=1wHxVJJxk1uKW3W4vOnisXPWzNSGtag31) [100EpochModel](https://drive.google.com/open?id=1Yd5lF0pnGivFd81kSdk4c8ZEVg_Io_JH)
self.InitNeurons = 14   
self.ExpansionFactor = 1.95 
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 37762557976  
NumParams = 2041798  
Expected Model Size = 23.389084 MB  
Network Used: Network.MobileNetv13    
Lambda = [1.0, 10.0, 10.0] # [Scale, Translation]  oscillates a lot
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation] trains at LR = 1e-5
warpType = ['translation', 'translation', 'scale', 'scale']  


## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) ShuffleNetv2 [50EpochModel](https://drive.google.com/open?id=1NpDK6JWlej-U4dAvWaDiNhSvoFsu9dNo) [100EpochModel](https://drive.google.com/open?id=15lij9mKISyo19DEWbHSh-OpPlThz5xsn) 
self.InitNeurons = 16  
self.ExpansionFactor = 2.0 
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 111691692124  
NumParams = 2103110  
Expected Model Size = 24.1038360596 MB  
Network Used: Network.ShuffleNetv23    
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation] LR = 1e-4 oscillates and trains super slowly, trying LR = 1e-5
warpType = ['translation', 'translation', 'scale', 'scale']  

# Choosing Best Small Network Architecture (Model Size <= 2.5 MB and Model FPS >= 200 FPS on All Cores i7)
- Running on Image Size of 128x128x(3x2)  
- No Data Augmentation on MSCOCO  
- Train on  LR = 1e-4, BatchSize = 32, NumEpochs = 100  

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) VanillaNet: [50EpochModel](https://drive.google.com/open?id=1DYNKILEF9AkV9ogCW-VWhGt6P1g9mscZ) 
self.InitNeurons = 10  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 2  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 16597160988  
NumParams = 208286  
Expected Model Size = 2.38822937012 MB  
Network Used: Network.VanillaNet3Small  
Lambda = [1.0, 10.0, 10.0] # [Scale, Translation]  
warpType = ['translation', 'translation', 'scale', 'scale']  

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) ResNet [50EpochModel](https://drive.google.com/open?id=1cChldU9Uwa409kHdlksm8xjA1tUVx8-p) 
self.InitNeurons = 8 
self.ExpansionFactor = 1.95 
self.DropOutRate = 0.7  
self.NumBlocks = 2  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 18004062284  
NumParams = 195466  
Expected Model Size = 2.24032592773MB  
Network Used: Network.ResNet3Small    
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation]  
warpType = ['translation', 'translation', 'scale', 'scale']  


## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) SqueezeNet [50EpochModel](https://drive.google.com/open?id=16_CdLylpog_HOlwVj6WguSekSZvyl6WN) 
self.InitNeurons = 10    
self.ExpansionFactor = 1.15 
self.DropOutRate = 0.7  
self.NumBlocks = 1  
self.NumFire = 1
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 17588437784  
NumParams = 196826  
Expected Model Size = 2.25314331055 MB  
Network Used: Network.SqueezeNet3Small    
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation]  
warpType = ['translation', 'translation', 'scale', 'scale']  


## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) MobileNetv1 [50EpochModel](https://drive.google.com/open?id=1ItLc5uGO5CDfVArHOGS1mhkFkWcXbbpZ) 
self.InitNeurons = 8    
self.ExpansionFactor = 1.95 
self.DropOutRate = 0.7  
self.NumBlocks = 2  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 13937688800  
NumParams = 198226  
Expected Model Size = 2.2744140625 MB  
Network Used: Network.MobileNetv13Small      
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation] trains at LR = 1e-5
warpType = ['translation', 'translation', 'scale', 'scale'] 


## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) ShuffleNetv2 [50EpochModel](https://drive.google.com/open?id=1FNmTMTgk4V38iZ7P9GhFbUVuP7Wq_eZQ)
self.InitNeurons = 8  
self.ExpansionFactor = 1.6
self.DropOutRate = 0.7  
self.NumBlocks = 2  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 11372248916  
NumParams = 213654  
Expected Model Size = 2.4493560791 MB  
Network Used: Network.ShuffleNetv23Small    
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation] 
warpType = ['translation', 'translation', 'scale', 'scale']  


# Choosing Best Loss Function for VanillaNet Large Network Architecture (Model Size <= 25 MB and Model FPS >= 20 FPS on All Cores i7)
- Running on Image Size of 128x128x(3x2)  
- No Data Augmentation on MSCOCO  
- Train on  LR = 1e-4, BatchSize = 32, NumEpochs = 100  
- Network Config:
VanillaNet 
self.InitNeurons = 18  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 43744396604  
NumParams = 2079870  
Expected Model Size = 23.8192596436 MB  
Network Used: Network.VanillaNet3  
Lambda = [1.0, 10.0, 10.0] # [Scale, Translation]  
LR = 1e-4 oscillates a lot and doesn't train, trying LR = 1e-5 DOESNT TRAIN!
warpType = ['translation', 'translation', 'scale', 'scale']

Trying!
self.InitNeurons = 26  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 40148554316  
NumParams = 2171890  
Expected Model Size = 24.8676147461 MB  
Network Used: Network.VanillaNet  
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation]  
warpType = ['pseudosimilarity', 'pseudosimilarity']  DOESNT TRAIN!

Trying S for 10 Epochs then FT.


## ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) Photo L1 [50EpochModel]() [100EpochModel]() Currently Training On Nitin's GPU 1 

## ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) Photo Chab [50EpochModel]() [100EpochModel]() 

## ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) Photo L1 + Reg L1 (Cornerness) [50EpochModel]() [100EpochModel]() 

## ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) Photo Chab + Reg Chab (Cornerness) [50EpochModel]() [100EpochModel]() 

## ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) Photo L1 + Reg Robust (Cornerness) [50EpochModel]() [100EpochModel]() 


- ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) `#f03c15` Red Not staged to be Trained
- ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) `#c5f015` Green  Trained
- ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) `#1589F0` Blue  Training
