"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('/home/darragh/omscs6476/ps06')

import ps6

# I/O directories
INPUT_DIR = "input_images"
OUTPUT_DIR = "output"

YALE_FACES_DIR = os.path.join(INPUT_DIR, 'Yalefaces')
FACES94_DIR    = os.path.join(INPUT_DIR, 'faces94')
POS_DIR        = os.path.join(INPUT_DIR, "pos")
NEG_DIR        = os.path.join(INPUT_DIR, "neg")
NEG2_DIR       = os.path.join(INPUT_DIR, "neg2")


def load_images_from_dir(data_dir, size=(24, 24), ext=".png"):
    imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(ext)]
    imgs = [np.array(cv2.imread(os.path.join(data_dir, f), 0)) for f in imagesFiles]
    imgs = [cv2.resize(x, size) for x in imgs]

    return imgs

# Utility function
def plot_eigen_faces(eig_vecs, fig_name="", visualize=False):
    r = np.ceil(np.sqrt(len(eig_vecs)))
    c = int(np.ceil(len(eig_vecs)/r))
    r = int(r)
    fig = plt.figure()

    for i,v in enumerate(eig_vecs):
        sp = fig.add_subplot(r,c,i+1)

        plt.imshow(v.reshape(32,32).real, cmap='gray')
        sp.set_title('eigenface_%i'%i)
        sp.axis('off')

    fig.subplots_adjust(hspace=.5)

    if visualize:
        plt.show()

    if not fig_name == "":
        plt.savefig("output/{}".format(fig_name))


# Functions you need to complete
def visualize_mean_face(x_mean, size, new_dims):
    """Rearrange the data in the mean face to a 2D array

    - Organize the contents in the mean face vector to a 2D array.
    - Normalize this image.
    - Resize it to match the new dimensions parameter

    Args:
        x_mean (numpy.array): Mean face values.
        size (tuple): x_mean 2D dimensions
        new_dims (tuple): Output array dimensions

    Returns:
        numpy.array: Mean face uint8 2D array.
    """
    
    x = np.reshape(x_mean, size)
    
    x = cv2.resize(x, new_dims, interpolation=cv2.INTER_CUBIC)
    
    return x


def part_1a_1b():

    orig_size = (192, 231)
    small_size = (32, 32)
    X, y = ps6.load_images(YALE_FACES_DIR, small_size)

    # Get the mean face
    x_mean = ps6.get_mean_face(X)

    x_mean_image = visualize_mean_face(x_mean, small_size, orig_size)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "ps6-1-a-1.png"), x_mean_image)

    # PCA dimension reduction
    k = 10
    eig_vecs, eig_vals = ps6.pca(X, k)

    plot_eigen_faces(eig_vecs.T, "ps6-1-b-1.png")


def part_1c():
    p = 0.5  # Select a split percentage value
    k = 5  # Select a value for k

    size = [32, 32]
    X, y = ps6.load_images(YALE_FACES_DIR, size)

    print 30*'--'
    for i in range(10):
        rand_accuracy = (y.flatten() == np.random.choice(range(1, 16), len(y))).sum()*100./len(y)
        print 'Part 1C : Random accuracy iter '+str(i)+' : {0:.2f}% '.format(rand_accuracy)
    print 30*'-*'

    # training
    def score_pca(k, p = 0.5, seed = 1):
        
        Xtrain, ytrain, Xtest, ytest = ps6.split_dataset(X, y, p, seed)
        mu = ps6.get_mean_face(Xtrain)
        eig_vecs, eig_vals = ps6.pca(Xtrain, k)
        Xtrain_proj = np.dot(Xtrain - mu, eig_vecs)
    
        # testing
        mu = ps6.get_mean_face(Xtest)
        Xtest_proj = np.dot(Xtest - mu, eig_vecs)
    
        good = 0
        bad = 0
    
        for i, obs in enumerate(Xtest_proj):
    
            dist = [np.linalg.norm(obs - x) for x in Xtrain_proj]
    
            idx = np.argmin(dist)
            y_pred = ytrain[idx]
    
            if y_pred == ytest[i]:
                good += 1
            else:
                bad += 1
        return good, bad
    good, bad = score_pca(k)
    print 30*'--'
    print 'PCA Good predictions = ', good, 'Bad predictions = ', bad
    print '{0:.2f}% accuracy'.format(100 * float(good) / (good + bad))
    print 30*'-*'
    
    multi_iters = [score_pca(k, p=0.5, seed = i) for i in range(10)]
    multi_iters = sum([(100 * float(g) / (g + b)) for (g,b) in multi_iters])/10
    print '{0:.2f}% accuracy over 10 iterations'.format(multi_iters)
    
    

    print 30*'--'
    scores = [(k_, score_pca(k_))  for k_ in [1,2,3,4,6,8,10,15,20, 30, 100]]
    scoremat = [ [t, 100 * float(g) / (g + b)]  for (t, (g,b)) in scores ]
    for row in scoremat:    
        print 'PCA k: '+str(row[0])+', {0:.2f}% accuracy'.format(row[1])
    print 30*'-*'
    
    print 30*'--'
    scores = [((0.1*p_), score_pca(8, 0.1*p_))  for p_ in range(1,10)]
    scoremat = [ [t, 100 * float(g) / (g + b)]  for (t, (g,b)) in scores ]
    for row in scoremat:    
        print 'Split p: '+str(row[0])+', {0:.2f}% accuracy'.format(row[1])
    print 30*'-*'
        

def part_2a(p = 0.8, num_iter = 5, seed = 1, verbose = True):
    
    y0 = 1
    y1 = 2

    X, y = ps6.load_images(FACES94_DIR)

    # Select only the y0 and y1 classes
    idx = y == y0
    idx |= y == y1

    X = X[idx.flatten(),:]
    y = y[idx]

    # Label them 1 and -1
    y0_ids = y == y0
    y1_ids = y == y1
    y[y0_ids] = 1
    y[y1_ids] = -1

    Xtrain, ytrain, Xtest, ytest = ps6.split_dataset(X, y, p, seed = seed)

    # Picking random numbers
    rand_y = np.random.choice([-1, 1], (len(ytrain)))
    # TODO: find which of these labels match ytrain and report its accuracy
    rand_accuracy =  100*sum(ytrain==rand_y)*1./ytrain.shape[0]
    # raise NotImplementedError
    if verbose:
        print '(Random) Training accuracy: {0:.2f}%'.format(rand_accuracy)

    # Using Weak Classifier
    uniform_weights = np.ones((Xtrain.shape[0],)) / Xtrain.shape[0]
    wk_clf = ps6.WeakClassifier(Xtrain, ytrain, uniform_weights)
    wk_clf.train()
    wk_results = [wk_clf.predict(x) for x in Xtrain]
    # TODO: find which of these labels match ytrain and report its accuracy
    wk_accuracy = 100*sum(ytrain==wk_results)*1./ytrain.shape[0]
    #raise NotImplementedError
    if verbose:
        print '(Weak) Training accuracy {0:.2f}%'.format(wk_accuracy)

    boost = ps6.Boosting(Xtrain, ytrain, num_iter)
    boost.train()
    good, bad = boost.evaluate()
    boost_accuracy = 100 * float(good) / (good + bad)
    if verbose:
        print '(Boosting) Training accuracy {0:.2f}%'.format(boost_accuracy)
    results = [p, num_iter, rand_accuracy, wk_accuracy, boost_accuracy ]

    # Picking random numbers
    rand_y = np.random.choice([-1, 1], (len(ytest)))
    # TODO: find which of these labels match ytest and report its accuracy
    rand_accuracy = 100*sum(ytest==rand_y)*1./ytest.shape[0]
    if verbose:
        print '(Random) Testing accuracy: {0:.2f}%'.format(rand_accuracy)

    # Using Weak Classifier
    wk_results = [wk_clf.predict(x) for x in Xtest]
    # TODO: find which of these labels match ytest and report its accuracy
    wk_accuracy = 100*sum(ytest==wk_results)*1./ytest.shape[0]
    if verbose:
        print '(Weak) Testing accuracy {0:.2f}%'.format(wk_accuracy)

    y_pred = boost.predict(Xtest)
    # TODO: find which of these labels match ytest and report its accuracy
    boost_accuracy = 100*sum(ytest==y_pred)*1./ytest.shape[0]
    if verbose:
        print '(Boosting) Testing accuracy {0:.2f}%'.format(boost_accuracy)
    results += [rand_accuracy, wk_accuracy, boost_accuracy ]
    return results

def part_3a():
    """Complete the remaining parts of this section as instructed in the
    instructions document."""

    
    posn = {1 : [(2, 1), (25, 30), (50, 100)],
                 2 : [(1, 2), (10, 25), (50, 150)],
                 3 : [(3, 1), (50, 50), (100, 50)],
                 4 : [(1, 3), (50, 125), (100, 50)],
                 5 : [(2, 2), (50, 25), (100, 150)]}
    
    for k, p in posn.items():
        feature = ps6.HaarFeature(*p)
        feature.preview((200, 200), filename="ps6-3-a-%s"%(k))


def part_4_a_b():

    pos = load_images_from_dir(POS_DIR)
    neg = load_images_from_dir(NEG_DIR)

    train_pos = pos[:35]
    train_neg = neg[:]
    images = train_pos + train_neg
    labels = np.array(len(train_pos) * [1] + len(train_neg) * [-1])

    integral_images = ps6.convert_images_to_integral_images(images)
    VJ = ps6.ViolaJones(train_pos, train_neg, integral_images)
    VJ.createHaarFeatures()

    VJ.train(4)

    VJ.haarFeatures[VJ.classifiers[0].feature].preview(filename="ps6-4-b-1")
    VJ.haarFeatures[VJ.classifiers[1].feature].preview(filename="ps6-4-b-2")

    predictions = VJ.predict(images)    
    vj_accuracy = 100.0 * ((predictions==labels).astype(np.int16).sum()) / len(labels)
    print "Prediction accuracy on training: {0:.2f}%".format(vj_accuracy)

    neg = load_images_from_dir(NEG2_DIR)

    test_pos = pos[35:]
    test_neg = neg[:35]
    test_images = test_pos + test_neg
    real_labels = np.array(len(test_pos) * [1] + len(test_neg) * [-1])
    predictions = VJ.predict(test_images)

    vj_accuracy = 100.0 * ((predictions==real_labels).astype(np.int16).sum()) / len(real_labels)
    print "Prediction accuracy on testing: {0:.2f}%".format(vj_accuracy)


def part_4_c():
    pos = load_images_from_dir(POS_DIR)[:20]
    neg = load_images_from_dir(NEG_DIR)

    images = pos + neg

    integral_images = ps6.convert_images_to_integral_images(images)
    VJ = ps6.ViolaJones(pos, neg, integral_images)
    VJ.createHaarFeatures()

    VJ.train(4)

    image = cv2.imread(os.path.join(INPUT_DIR, "man.jpeg"), -1)
    image = cv2.resize(image, (120, 60))
    VJ.faceDetection(image, filename="ps4-4-c-1")


if __name__ == "__main__":
    part_1a_1b()
    part_1c()
    _ = part_2a()
    print 30*'--'
    print('p,split; Iters       ;Trn Rand; Trn Weak; Trn Boost; Tst Rand; Tst Weak; Tst Boost')
    for iters in [2,5,10,20]:
        iter_res = [part_2a(0.8, iters, s, False) for s in range(5)]
        np.set_printoptions(precision=5)
        print(np.array(iter_res).mean(axis = 0))
    print 30*'-*'
    print 30*'--'
    print('p,split; Iters       ;Trn Rand; Trn Weak; Trn Boost; Tst Rand; Tst Weak; Tst Boost')
    for p in [0.2, 0.4, 0.6, 0.8]:
        iter_res = [part_2a(p, 10, s, False) for s in range(5)]
        np.set_printoptions(precision=5)
        print(np.array(iter_res).mean(axis = 0))
    print 30*'-*'
    part_3a()
    part_4_a_b()
    part_4_c()
    print 'Compare integral to numpy sum'
    def np_sum(img):
        im = np.zeros(img.shape)
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                im[i,j] = img[:(i+1), :(j+1)].sum()
        return im.astype(np.int32)
    def convert_image_to_integral_images(image):    
        return np.cumsum(np.cumsum(image, axis=0), axis=1).astype(np.int32)
    for sz in [20,40,80,160]:
        size = (sz,sz)
        images_files = [f for f in os.listdir(YALE_FACES_DIR) if f.endswith(".png")]
        imgs = [cv2.imread(os.path.join(YALE_FACES_DIR, img)) for img in images_files]
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
        imgs = [cv2.resize(img, tuple(size), interpolation = cv2.INTER_CUBIC) for img in imgs]
        print '----------------Image Size : %s-----------------'%(sz)
        %time A = np_sum(imgs[0])
        %time B = convert_image_to_integral_images(imgs[0])
        print('Both arrays equal - ' + str(np.array_equal(A, B)))

    
    
    