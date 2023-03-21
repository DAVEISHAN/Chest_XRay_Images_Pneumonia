#Importing dependecies
import numpy as np
import pandas as pd
import torch
import os, time
import matplotlib.pyplot as plt
import torch.nn.functional as F 
import torch
from torch import nn,optim
from torchvision import transforms as T, datasets, models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
torch.backends.cudnn.benchmark = True
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from torch.utils.tensorboard import SummaryWriter


#Setting directories
data_dir = 'chest_xray/' #"../input/chest-xray-pneumonia/chest_xray/chest_xray"
TEST = 'test'
TRAIN = 'train'
VAL ='val'
logs_dir = 'logs'
saved_models_dir = 'saved_models'

if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

def train_epoch(run_id, learning_rate2,  epoch, data_loader, model, criterion, optimizer, writer, use_cuda, scaler,device_name):
        print('train at epoch {}'.format(epoch))
        for param_group in optimizer.param_groups:
                param_group['lr']=learning_rate2
                writer.add_scalar('Learning Rate', learning_rate2, epoch)  

                print("Learning rate is: {}".format(param_group['lr']))


        losses, weighted_losses = [], []
        loss_mini_batch = 0
        predictions, gt = [], []

        model.train()
        # print()

        for i, (inputs, label) in enumerate(data_loader):
                optimizer.zero_grad()
                if use_cuda:
                        inputs = inputs.to(device=torch.device(device_name))
                        label = label.to(device=torch.device(device_name), dtype= float)
                with autocast(): 
                        # print(label)
                        output = model(inputs).squeeze()
                        loss = criterion(output,label)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()       
                predictions.extend(torch.sigmoid(output).detach().cpu().numpy()) 
                gt.extend(label.cpu().numpy())

                losses.append(loss.item())
                if i % 500 == 0: 
                        print(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}', flush=True)
        print('Training Epoch: %d, Loss: %.4f' % (epoch,  np.mean(losses)))
        writer.add_scalar('Training Loss', np.mean(losses), epoch)
        
        
        predictions = np.asarray(predictions)
        predictions_acc = predictions>0.5
        predictions_map = predictions


        gt = np.asarray(gt)
        
        # print(predictions)
        # print(gt)

        accuracy = ((predictions_acc == gt).sum())/np.size(predictions)

        print(f'Training Accuracy at Epoch {epoch} is {accuracy*100 :0.3f}')
        writer.add_scalar('Training Accuracy', accuracy, epoch)

        ap = average_precision_score(gt, predictions, average=None)
        print(f'Training Macro AP is {np.nanmean(ap)}')
        writer.add_scalar('Training map', np.nanmean(ap), epoch)

        results = precision_recall_fscore_support(gt, predictions_acc, average='macro')

        prec, rec, f1 = results[0], results[1], results[2]

        print(f'Training Prec is {prec}')
        print(f'Training Recall is {rec}')
        print(f'Training f1 is {f1}')

        writer.add_scalar('Training Precision', prec, epoch)
        writer.add_scalar('Training Recall', rec, epoch)
        writer.add_scalar('Training F1', f1, epoch)

        return model, np.mean(losses), scaler

def val_epoch(run_id, learning_rate2,  epoch, data_loader, model, criterion, optimizer, writer, use_cuda, scaler, device_name):
        # print('train at epoch {}'.format(epoch))
        # for param_group in optimizer.param_groups:
        # param_group['lr']=learning_rate2
        # writer.add_scalar('Learning Rate', learning_rate2, epoch)  

        # print("Learning rate is: {}".format(param_group['lr']))


        losses, weighted_losses = [], []
        loss_mini_batch = 0
        predictions, gt = [], []

        model.eval()

        for i, (inputs, label) in enumerate(data_loader):
                optimizer.zero_grad()
                if use_cuda:
                        inputs = inputs.to(device=torch.device(device_name))
                        label = label.to(device=torch.device(device_name), dtype= float)
                with autocast(): 
                        # print(label)
                        output = model(inputs).squeeze()
                        loss = criterion(output,label)
                        # scaler.scale(loss).backward()
                        # scaler.step(optimizer)
                        # scaler.update()       
                predictions.extend(torch.sigmoid(output).detach().cpu().numpy()) 
                gt.extend(label.cpu().numpy())

                losses.append(loss.item())
                if i % 500 == 0: 
                        print(f'Val Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}', flush=True)
        print('Val Epoch: %d, Loss: %.4f' % (epoch,  np.mean(losses)))
        writer.add_scalar('Val Loss', np.mean(losses), epoch)
        
        predictions = np.asarray(predictions)
        predictions_acc = predictions>0.5
        predictions_map = predictions
        

        gt = np.asarray(gt)
        
        accuracy = ((predictions_acc == gt).sum())/np.size(predictions)

        print(f'Val Accuracy at Epoch {epoch} is {accuracy*100 :0.3f}')
        writer.add_scalar('Val Accuracy', accuracy, epoch)

        ap = average_precision_score(gt, predictions, average=None)
        print(f'Val Macro AP is {np.nanmean(ap)}')
        writer.add_scalar('Val map', np.nanmean(ap), epoch)

        results = precision_recall_fscore_support(gt, predictions_acc, average='macro')

        prec, rec, f1 = results[0], results[1], results[2]

        print(f'Val Prec is {prec}')
        print(f'Val Recall is {rec}')
        print(f'Val f1 is {f1}')

        writer.add_scalar('Val Precision', prec, epoch)
        writer.add_scalar('Val Recall', rec, epoch)
        writer.add_scalar('Val F1', f1, epoch)

        return model, np.mean(losses), scaler, predictions, gt, accuracy

def test_epoch(run_id, learning_rate2,  epoch, data_loader, model, criterion, optimizer, writer, use_cuda, scaler, device_name):
        # print('train at epoch {}'.format(epoch))
        # for param_group in optimizer.param_groups:
        # param_group['lr']=learning_rate2
        # writer.add_scalar('Learning Rate', learning_rate2, epoch)  

        # print("Learning rate is: {}".format(param_group['lr']))


        losses, weighted_losses = [], []
        loss_mini_batch = 0
        predictions, gt = [], []

        model.eval()

        for i, (inputs, label) in enumerate(data_loader):
                optimizer.zero_grad()
                if use_cuda:
                        inputs = inputs.to(device=torch.device(device_name))
                        label = label.to(device=torch.device(device_name), dtype= float)
                with autocast(): 
                        # print(label)
                        output = model(inputs).squeeze()
                        loss = criterion(output,label)
                        # scaler.scale(loss).backward()
                        # scaler.step(optimizer)
                        # scaler.update()       
                predictions.extend(torch.sigmoid(output).detach().cpu().numpy()) 
                gt.extend(label.cpu().numpy())

                losses.append(loss.item())
                if i % 500 == 0: 
                        print(f'Test Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}', flush=True)
        print('Test Epoch: %d, Loss: %.4f' % (epoch,  np.mean(losses)))
        writer.add_scalar('Test Loss', np.mean(losses), epoch)
        
        predictions = np.asarray(predictions)
        predictions_acc = predictions>0.5
        predictions_map = predictions


        gt = np.asarray(gt)
        
        accuracy = ((predictions_acc == gt).sum())/np.size(predictions)

        print(f'Test Accuracy at Epoch {epoch} is {accuracy*100 :0.3f}')
        writer.add_scalar('Test Accuracy', accuracy, epoch)

        ap = average_precision_score(gt, predictions, average=None)
        print(f'Test Macro AP is {np.nanmean(ap)}')
        writer.add_scalar('Test map', np.nanmean(ap), epoch)

        results = precision_recall_fscore_support(gt, predictions_acc, average='macro')

        prec, rec, f1 = results[0], results[1], results[2]

        print(f'Test Prec is {prec}')
        print(f'Test Recall is {rec}')
        print(f'Test f1 is {f1}')

        writer.add_scalar('Test Precision', prec, epoch)
        writer.add_scalar('Test Recall', rec, epoch)
        writer.add_scalar('Test F1', f1, epoch)

        return model, np.mean(losses), scaler, predictions, gt, accuracy


def train_classifier(run_id, restart, saved_model, linear, params, devices, dataset1):         
        for item in dir(params):
                if '__' not in item:
                        print(f'{item} =  {params.__dict__[item]}') 
        save_dir = os.path.join(saved_models_dir, run_id)
        use_cuda = True

        if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        # Defining the augmentations

        aug_train = params.aug_train
        
        aug_test = params.aug_test
        
                

        # Dataloader

        trainset = datasets.ImageFolder(os.path.join(data_dir, TRAIN),transform = aug_train)
        testset = datasets.ImageFolder(os.path.join(data_dir, TEST),transform = aug_test)
        validset = datasets.ImageFolder(os.path.join(data_dir, VAL),transform = aug_test)

        trainloader = DataLoader(trainset, batch_size = params.batch_size, shuffle = True, num_workers=params.num_workers)
        validloader = DataLoader(validset, batch_size = params.v_batch_size, shuffle = False, num_workers=params.num_workers)
        testloader = DataLoader(testset, batch_size = params.v_batch_size, shuffle = False, num_workers=params.num_workers)
        print(f'Train dataset length: {len(trainset)}')
        print(f'Train dataset steps per epoch: {len(trainset)/params.batch_size}')

        print(f'Val dataset length: {len(validset)}')
        print(f'Val dataset steps per epoch: {len(validset)/params.v_batch_size}')

        print(f'Test dataset length: {len(testset)}')
        print(f'Test dataset steps per epoch: {len(testset)/params.v_batch_size}')
    

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = models.resnet18(pretrained=params.pretraining)
        model.fc = nn.Linear(512,1) # 1 class for the logistic regression, sigmoid will be applied later
        # print(model)
        if linear:
                print('Its linear evaluation')
                for name, param in model.named_parameters():
                
                        if not ('.fc' in name):
                                param.requires_grad = False
                                # print(f'frezzing: {name}')

                        else:
                                print(f'unfrozen: {name}')


        criterion = nn.BCEWithLogitsLoss()
        writer = SummaryWriter(os.path.join(logs_dir, str(run_id)))


        if params.opt_type == 'adam':
                optimizer = optim.Adam(model.parameters(),lr=params.learning_rate)
        elif params.opt_type == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9)
        elif params.opt_type == 'adamW':
                optimizer = optim.AdamW(model.parameters(),lr=params.learning_rate, weight_decay=1e-8)
        else:
                raise NotImplementedError(f"not supporting {params.opt_type}")        

        epoch0 = 0
        learning_rate1 = params.learning_rate

        device_name =  'cuda:' + str(devices[0]) # This is used to move the data no matter if it is multigpu
        print(f'Device name is {device_name}')
        if len(devices)>1:
                print(f'Multiple GPUS found!')
                # model=nn.DataParallel(model)
                model = torch.nn.DataParallel(model, device_ids=devices)
                model.cuda()
                # criterion.
                # cuda()
        else:
                print('Only 1 GPU is available')
                
                model.to(device=torch.device(device_name))
                # model.cuda()
                criterion.to(device=torch.device(device_name))
                # criterion.cuda()

        val_array = params.val_array
        test_array = params.test_array
        accuracy = 0
        best_acc = 0 

        learning_rate2 = learning_rate1 
        scheduler_step = 1  
        scheduler_epoch = 0
        train_loss = 1000
        best_score = 10000

        scaler = GradScaler()


        for epoch in range(epoch0, params.num_epochs):
                print('-------------------------------------')
                print(f'Epoch {epoch} started')
                start=time.time()

                # try:
                if params.lr_scheduler == "cosine":
                        learning_rate2 = params.cosine_lr_array[epoch]*learning_rate1
                elif params.warmup and epoch < len(params.warmup_array):
                        learning_rate2 = params.warmup_array[epoch]*learning_rate1
                elif params.lr_scheduler == "loss_based":
                        if train_loss < 1.0 and train_loss>=0.5:
                                learning_rate2 = learning_rate1/2
                        elif train_loss <0.5:
                                learning_rate2 = learning_rate1/10
                        elif train_loss <0.1:
                                learning_rate2 = learning_rate1/20    
                elif params.lr_scheduler == "patience_based":

                        if scheduler_epoch == params.scheduler_patience:
                                print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
                                print(f'Dropping learning rate to {learning_rate2/10} for epoch')
                                print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
                                learning_rate2 = learning_rate1/(params.lr_reduce_factor**scheduler_step)
                                scheduler_epoch = 0
                                scheduler_step += 1
        
                model, train_loss, scaler = train_epoch(run_id, learning_rate2,  epoch, trainloader, model, criterion, optimizer, writer, use_cuda, scaler,device_name)
                print()
                if train_loss < best_score:
                # scheduler_epoch += 1

                        best_score = train_loss
                        scheduler_epoch = 0
                else:
                        scheduler_epoch+=1             

                if epoch in val_array:
                        
                        model, val_loss, scaler, val_pred_array, val_gt, val_acc = val_epoch(run_id, learning_rate2,  epoch, validloader, model, criterion, optimizer, writer, use_cuda, scaler, device_name)
                if epoch in test_array:
                        model, test_loss, scaler, test_pred_array, test_gt, accuracy = test_epoch(run_id, learning_rate2,  epoch, testloader, model, criterion, optimizer, writer, use_cuda, scaler, device_name)
                
                if accuracy > best_acc:
                        print('++++++++++++++++++++++++++++++')
                        print(f'Epoch {epoch} is the best model till now for {run_id}!')
                        print('++++++++++++++++++++++++++++++')
                        if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                        save_file_path = os.path.join(save_dir, 'model_{}_bestAcc_{}.pth'.format(epoch, str(accuracy)[:6]))
                        states = {
                                'epoch': epoch + 1,
                                # 'arch': params.arch,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'amp_scaler': scaler,
                                'test_preds': test_pred_array,
                                'test_gt': test_gt
                        }
                        torch.save(states, save_file_path)
                        best_acc = accuracy
        
                if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                        save_file_path = os.path.join(save_dir, 'model_temp.pth')
                
                states = {
                'epoch': epoch + 1,
                # 'arch': params.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp_scaler': scaler,
                }
                torch.save(states, save_file_path)
                
                # except:
                #     print("Epoch ", epoch, " failed")
                #     print('-'*60)
                #     traceback.print_exc(file=sys.stdout)
                #     print('-'*60)
                #     continue
                taken = time.time()-start
                print(f'Time taken for Epoch-{epoch} is {taken}')
                print()
                trainloader = DataLoader(trainset, batch_size = params.batch_size, shuffle = True, num_workers=params.num_workers)

                if (params.lr_scheduler != 'cosine') and learning_rate2 < 1e-10 and epoch > 10:
                        print(f'Learning rate is very low now, ending the process...s')
                        exit()






if __name__ == '__main__':
    import argparse, importlib

    parser1 = argparse.ArgumentParser(description='Script to do linear evaluation ')

    parser1.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy_linear",
                        help='run_id')
    parser1.add_argument("--restart", action='store_true')
    parser1.add_argument("--saved_model", dest='saved_model', type=str, required=False, default= None,
                        help='run_id')
    parser1.add_argument("--linear", action='store_true')
    parser1.add_argument("--params", dest='params_file_location', type=str, required=True, default= "paras",
                        help='params_file_location')
    parser1.add_argument("--devices", dest='devices', action='append', type =int, required=False, default=None,
                        help='devices should be a list even when it is single')
    parser1.add_argument("--dataset", dest='dataset1', type=str, required=False, default= None,
                        help='Not Used, put anything')
    # print()
    # print('Repeating r3d57, Optimizer grad inside each iteration')
    # print()
    

    args = parser1.parse_args()
    print(f'Restart {args.restart}')
    
    params_filename = args.params_file_location.replace('.py', '')
    if os.path.exists(params_filename + '.py'):
        # import args.params_file_location as params
        params = importlib.import_module(params_filename)
        print(f' {params_filename} is loaded as params')
    else:
        print(f'{params_filename} dne, give it correct path!')
        
#     from dataloaders.dl_linear_frameids import *
    
    
    run_id = args.run_id
    saved_model = args.saved_model
    linear = args.linear
    devices = args.devices
    dataset1 = args.dataset1

    if devices is None: 
        devices = list(range(torch.cuda.device_count()))
    
    print(f'devices are {devices}') 
    # exit()
    if saved_model is not None and len(saved_model):
        saved_model = '/' +saved_model
        saved_model = saved_model.replace('-symlink', '')
    else:
        saved_model = None


    train_classifier(str(run_id), args.restart, saved_model, linear, params, devices, dataset1)
