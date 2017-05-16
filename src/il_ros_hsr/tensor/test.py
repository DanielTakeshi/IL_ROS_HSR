from inputdata import AMTData

train_path = '/Users/JonathanLee/Desktop/sandbox/vision/Net/hdf/train.txt'
test_path = '/Users/JonathanLee/Desktop/sandbox/vision/Net/hdf/test.txt'
data = AMTData(train_path, test_path)

print data.next_train_batch(100)[0][0].shape
