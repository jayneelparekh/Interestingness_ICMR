import keras.backend as K
from keras.models import Model
import model
from keras import callbacks as cbk
import numpy as np
import random
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import Ranker
import sys



# Constants
batch_size = 300 # 300 = 25C2
batch_img = 25
train_indices = np.array([30, 38, 50, 32, 63, 68, 24, 66, 34, 70, 42, 22, 55, 41, 43, 45, 7, 35, 33, 71, 40, 23, 16, 75, 12, 21, 64,  4, 61, 44, 15, 18, 51, 54, 6,  0, 67, 13, 31, 17, 39, 53, 26, 59, 48, 77, 19,  1, 76, 58, 47, 20, 46,  8,  2, 62,  3, 74, 72, 60])
val_indices = np.array([56, 28, 37, 52, 14, 69, 36, 65, 11,  9,  5, 73, 27, 29, 49, 57, 10, 25])
num_epoch = 37




class CB(cbk.Callback):
    def __init__(self, predictor=None):
	self.model = predictor




    def write_out(self, epoch, names_test, num_img_test, inp_feat_test, rank_algo, out_file='results_nn') :

	start_idx = -1
        end_idx = 0

        pred_file = open(out_file, 'r+')
        pred_file.seek(0)

        for i in range(len(num_img_test)):

	    print "Predicting for video: ", i+78
            vid_name = "video_" + str(i+78)
    	    start_idx = end_idx
            end_idx = start_idx + num_img_test[i]
	    interest_vec_sp = 0.0*np.array(range(num_img_test[i]))
            interest_vec_pp = 0.0*np.array(range(num_img_test[i]))
    	    binary_int = 0*np.array(range(num_img_test[i])) 
	    num_int = int(0.12*num_img_test[i]) + 1
	    cur_PPM = np.zeros([num_img_test[i], num_img_test[i]])
	    
	    for j in range(start_idx, end_idx, 1):
                for k in range(j+1, end_idx, 1):
		    pred = self.model.predict(  np.append(inp_feat_test[j], inp_feat_test[k]).reshape(1, -1)  )[0][0]
		    pred_int = int(np.round(pred))
		    cur_PPM[j-start_idx, k-start_idx] = pred_int
                    
                    interest_vec_pp[pred_int*(k-start_idx) + (1-pred_int)*(j-start_idx)] += abs(pred-0.5)
		    interest_vec_sp[pred_int*(k-start_idx) + (1-pred_int)*(j-start_idx)] += 1   #Less interesting image gets 1 added in score

	    if (rank_algo == 'sp'):
                interest_vec = interest_vec_sp
                temp = interest_vec.argsort()
                ranks = np.arange(len(interest_vec))[temp.argsort()]
                ranks = ranks + 1

            elif (rank_algo == 'pp'):
                interest_vec = interest_vec_pp
                temp = interest_vec.argsort()
                ranks = np.arange(len(interest_vec))[temp.argsort()]
                ranks = ranks + 1

            elif (rank_algo == 'mih_to'):
                ranks = Ranker.mih_to(cur_PPM)

            elif (rank_algo == 'mih_ro'):
                ranks = Ranker.mih_ro(cur_PPM)

	    binary_int = (ranks <= num_int).astype(int)

	    print "Predictions made for video: ", i+78
	    for j in range(num_img_test[i]):
                str_line = vid_name + ',' + str(names_test[i][j]) + ',' + str(binary_int[j]) + ',' + str(1.0/ranks[j])
	        pred_file.write(str_line + '\n')
        pred_file.close()    	    
        return

    def on_epoch_end(self, epoch, logs={}):
	#self.model.save_weights('/home/jayneel/Desktop/MediaEval_2017/All_Codes/weights_v2.hdf5')
	print "Epoch number: ", epoch
	if (epoch == 20):
	    self.write_out(21, 'results_nn_21')
	#res = self.validate(epoch)
	





def my_gen(annot_train, num_img_train, inp_feat_train, train_type='complete'):

    inp = np.zeros([batch_size, 8192])
    inp_1 = np.zeros([batch_size, 4096])
    inp_2 = np.zeros([batch_size, 4096])
    out = np.zeros([batch_size, 1])

    while True:
        start_idx = -1
        end_idx = 0

	for i in range(len(num_img_train)):
	    start_idx = end_idx
	    end_idx = end_idx + num_img_train[i]

	
	    if (train_type == 'val' and i in val_indices):
		continue
	    

	    cur_imgs = inp_feat_train[start_idx:end_idx]
	    cur_annot = annot_train[:, 3][start_idx:end_idx]

	    # Randomly permute the images
	    indices = np.array(range(len(cur_annot)))  # For shuffling
	    np.random.shuffle(indices)
	    cur_imgs = cur_imgs[indices]
	    cur_annot = cur_annot[indices]

	    select_imgs = cur_imgs[:batch_img]
	    select_annot = cur_annot[:batch_img]
	    shuffle_obj = np.array([0,1])

	    # Make all possible pairs
	    pair_idx = 0
	    for j in range(batch_img-1):
		for k in range(j+1, batch_img):

		    np.random.shuffle(shuffle_obj)
		    if(shuffle_obj[0] == 0):
			idx1 = j
			idx2 = k
		    else:
			idx1 = k
			idx2 = j
			
		    inp[pair_idx] = np.append(select_imgs[idx1], select_imgs[idx2])
		    inp_1[pair_idx] = select_imgs[idx1]
		    inp_2[pair_idx] = select_imgs[idx2]
		    temp = int(float(select_annot[idx1])-float(select_annot[idx2]) > 0)
		    #out[pair_idx] = np.array([temp, 1-temp])
		    out[pair_idx] = temp
		    pair_idx += 1

	    z = (inp, out)
	    #z = ([inp_1, inp_2], out)
	    yield z





if __name__== '__main__':
    # Image/Video, train/test, rank_algo

    args = sys.argv[1:]

    if (args[0] == 'image'):
        inp_feat_train = np.load('Dataset/fc7_all.npy')
        num_img_train = np.load('Dataset/num_img.npy')
        annot_train = np.load('Dataset/Devset_images.npy') # Annotations for images

        names_test = np.load('Dataset/Testset_img.npy')
        num_img_test = np.load('Dataset/num_img_test.npy')
        inp_feat_test = np.load('Dataset/fc7_all_test.npy')

    elif (args[0] == 'video'):
        inp_feat_train = np.load('Dataset/c3d_all.npy')
        num_img_train = np.load('Dataset/num_img.npy')
        annot_train = np.load('Dataset/Devset_videos.npy') # Annotations for images

        names_test = np.load('Dataset/Testset_vid.npy')
        num_img_test = np.load('Dataset/num_img_test.npy')
        inp_feat_test = np.load('Dataset/c3d_all_test.npy')

    else :
        print "ERROR: Choose between image and video"
        sys.exit()


    if (args[2] == 'sp' or args[2] == 'pp' or args[2] == 'mih_to' or args[2] == 'mih_ro'):
        rank_algo = args[2]
    else :
        print "ERROR: Choose between sp, pp, mih_to, mih_ro"
        sys.exit()
    



    model_vnum = 15
    network = model.create_model(args[0])
    cb = CB(network)

    if (args[1] == 'train'):
        network.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        print "Training to start"
        history = network.fit_generator(my_gen(annot_train, num_img_train, inp_feat_train), steps_per_epoch=2*78, epochs=num_epoch, initial_epoch=0, callbacks=[cb])
        print "Training Completed"
        network.save_weights('weights_v' + str(model_vnum) + '.hdf5')
        print "Weights saved"

    elif (args[1] == 'test'):
        network.load_weights('weights_v' + str(model_vnum) + '.hdf5')
        print "Weights loaded" 
        #cb.validate(0, annot_train, num_img_train, inp_feat_train)   
        cb.write_out(0, names_test, num_img_test, inp_feat_test, rank_algo)

    else:
        print "ERROR: Choose between train or test"
        sys.exit()




# Plotting MAP@10 and MAP
#plt.plot(np.array(range(num_epoch)), MAP_at_10)
#plt.show()




    
