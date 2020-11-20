from data_class_old import fsd_dataset
import torch



train_dataset = fsd_dataset(csv_file = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.meta/train_mini_dataset.csv',
                                path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.audio_train/',
#     # mmap_path = 'D:/DL_research/DL_research/memmap_stuff/train/memmaps/',
#     # json_path = 'D:/DL_research/DL_research/memmap_stuff/train/json/',                                
                                
                                train = True)




test_dataset = fsd_dataset(csv_file = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.meta/test_mini_dataset.csv',
                                path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.audio_test/',
                                
#     # mmap_path = 'D:/DL_research/DL_research/memmap_stuff/test/memmaps/',
#     # json_path = 'D:/DL_research/DL_research/memmap_stuff/test/json/',                                           

                                train = False)





class ram_pusher():

    def __init__(self, args):
        super(self).__init__()
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                            shuffle=True,
                            batch_size = args.batch_size)

        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                    shuffle=False,
                                    batch_size = args.batch_size)


