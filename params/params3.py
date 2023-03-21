import numpy as np
import math
from torchvision import transforms as T

num_workers = 6
batch_size = 32#8

v_batch_size = 32#80
dataset = 'Chest_XRay'

learning_rate = 1e-3 #1e-5
num_epochs = 50

lr_scheduler = "cosine" #"patience_based" #"loss_based (default)" #cosine
cosine_lr_array = list(np.linspace(0.01,1, 5)) + [(math.cos(x) + 1)/2 for x in np.linspace(0,math.pi/0.99, num_epochs-5)]

scheduler_patience = 1
lr_reduce_factor = 2
warmup = True
warmup_array = [0.1, 1]
val_freq = 3
opt_type = 'adam' #'sgd', 'adamW'

val_array =  list(range(0,num_epochs,2)) #[0,5,10] + list(range(12, num_epochs,2)) #could be empty to take defaults
test_array =  list(range(0,num_epochs,2)) 

pretraining = False
input_reso = 224


aug_train = T.Compose([
                # T.Resize(size = (256,256)),
                # T.RandomRotation(degrees = (-20,+20)),
                T.RandomResizedCrop(size = input_reso, scale = (0.9, 1.0), ratio = (0.99, 1.01)),
                T.RandomHorizontalFlip(p=0.5),
                # T.CenterCrop(size=224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                
        ])

aug_test = T.Compose([ T.Resize(size = (input_reso, input_reso)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])