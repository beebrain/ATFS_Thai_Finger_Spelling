import numpy as np 
class BatchHelp:

    def __init__(self,data,label):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = np.array(data)
        self._label = np.array(label)
        self._num_examples = len(data)
        pass


    @property
    def data(self):
        return self._data,self._label

    def resetIndex(self):
        self._index_in_epoch = 0
        self._epochs_completed = 0

    def next_batch(self,batch_size,shuffle = False):
        start = self._index_in_epoch

        #if first state shuffle dataset
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            if shuffle:
                np.random.shuffle(idx)  # shuffle indexe
            self._data = self._data[idx]  # get list of `num` random samples
            self._label = self._label[idx]


        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            label_rest_part = self._label[start:self._num_examples]


            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            if shuffle:
                np.random.shuffle(idx0)  # shuffle indexe

            ## concat data to full batch size
            self._data = self._data[idx0]  # get list of `num` random samples
            self._label = self._label[idx0] #get lable data

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch
            data_new_part =  self._data[start:end]
            label_new_part = self._label[start:end]

            if rest_num_examples != 0:
                data_new_part =  np.concatenate((data_rest_part, data_new_part), axis=0)
                label_new_part = np.concatenate((label_rest_part, label_new_part), axis=0)

            return data_new_part,label_new_part

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end],self._label[start:end]
