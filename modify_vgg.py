#
# images = tf.placeholder(tf.float32, [None, 224, 224, 3])
# #to freeze, use trainable=false
# kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights', trainable='false')
# conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
# sess = tf.Session()
# w = weights[0]
# print(w.shape)
# w = np.moveaxis(w, 0, -1)
# print(w.shape)
# #do for each conv layer kernel and bias in VGG- see if any different then unrefined VGG
# sess.run(kernel.assign(w))
