
import os

import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import time
from tools import *
from server import *
from models import UNet,largeUnet, UNet3D, largeUNet3D
from tools import *

from options import args_parser
from utils import exp_details, get_datasets,get_public_dataset, average_weights
from update import LocalUpdate
from magic_tool import *
from resnet_family import *
from sampling import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm


if __name__ == '__main__':
    start_time = time.time()


    args = args_parser()
    exp_details(args)
    
    device = args.device
    


    train_dataset, test_dataset, user_groups = get_datasets(args)
    for value in user_groups.values():
        print("---")
        print(len(value[0]))
        print(len(value[1]))
    exit()

    if args.public==1:
        
        public_data_list=get_public_dataset(args,dst_name='3DSeg')
 
        


    large_client_idx = ranking_dict(user_groups)[:3]
    print("large client idx is: {}".format(large_client_idx))

    if args.dataset == 'ChestXray_seg':
        large_model = largeUnet(n_class = 1)
        global_model = UNet(n_class = 1)
    if args.dataset == 'Fed_IXI':
        global_model = UNet3D()
        large_model = UNet3D()


    logits_from_llm = None 
    global_weights = global_model.state_dict()
    train_loss, test_accuracy = [], []
    test_commmunication_dice = []
    test_commmunication_dice_user_vice = []
    print_every = 1
    with tqdm(total= args.communication_round, desc="Training Progress") as pbar:
        for epoch in range(args.communication_round):
            local_weights, local_losses = [], []
            print(f'\n | communication round : {epoch+1} |\n')
            global_model.train()
            # 
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            # for idx in idxs_users:
            epoch_user_dice_test_score = []
            for idx in idxs_users:
                if args.test_pass == 0:
                    if idx in large_client_idx:
                        print("start large client {}".format(idx))
                        large_model.train()
                        local_model = LocalUpdate(args=args, dataset=[train_dataset,test_dataset],
                                                idxs= user_groups[idx], small_model = global_model)

                        w, loss = local_model.large_client_update_weights_seg(large_model=large_model,
                            small_model=copy.deepcopy(global_model), global_round=epoch)
                        
                        user_dice_score = local_model.seg_inference(model=global_model, model_dict=w)
                        print("large_user_{}_dice_score is: {}".format(idx, user_dice_score))
               
                        local_weights.append(copy.deepcopy(w))
                        local_losses.append(copy.deepcopy(loss))
  

                    
                else:
                    print("start small client {}".format(idx))
              
                    local_model = LocalUpdate(args=args, dataset=[train_dataset,test_dataset],
                                            idxs= user_groups[idx], small_model = global_model)
                    w, loss = local_model.small_client_update_weights_seg_3d(
                            small_model=copy.deepcopy(global_model), global_round=epoch, public_data=public_data_list)
               
                    user_dice_score = local_model.seg_inference(model=global_model, model_dict=w)
                    print("small_user_{}_dice_score is: {}".format(idx, user_dice_score))

                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))
   
                epoch_user_dice_test_score.append(user_dice_score)
            test_commmunication_dice.append(sum(epoch_user_dice_test_score)/len(epoch_user_dice_test_score))
            test_commmunication_dice_user_vice.append(epoch_user_dice_test_score)

     
            print("start global avg")
            global_weights = average_weights(local_weights)

   
            global_model.load_state_dict(global_weights)
            global_model.to(device)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            if (epoch+1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Avg Local Training Loss : {np.mean(np.array(train_loss))}')
                print('Avg All Local dice: {} \n'.format(test_commmunication_dice[-1]))
            time.sleep(0.1)
            pbar.update(1)

    save_path = './save_our_seg_lung/{}_pub_{}_seg_largemodel_{}'.format(args.dataset, args.public, args.large_model_is)
    isExist = os.path.exists(save_path)
    if not isExist:
        os.makedirs(save_path)
        print("The new directory is created!")

    #         filehandle.write('%s\n' % listitem) 
    torch.save(global_model.state_dict(), save_path + '/global_model_comm_{}.pth'.format(args.communication_round))
    with open(save_path + '/local_avg_test_dice_ours_{}_comm_{}.txt'.format(args.ours, args.communication_round), 'w') as filehandle:
        for listitem in test_commmunication_dice:
            filehandle.write('%s\n' % listitem) 
    with open(save_path + '/local_avg_test_dice_ours_{}_comm_{}_each_client.txt'.format(args.ours, args.communication_round), 'w') as filehandle:
        for listitem in test_commmunication_dice_user_vice:
            filehandle.write('%s\n' % listitem) 

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
        



    plt.figure()
    plt.title('Local Average Test dice vs Communication rounds')
    plt.plot(range(len(test_commmunication_dice)), test_commmunication_dice, color='r')
    plt.ylabel('Test Dice')
    plt.xlabel('Communication Rounds')
    plt.savefig(save_path + '/fed_test_dice_ours_{}_comm_{}.png'.format(args.ours, args.communication_round))

          

    





    
    



