"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os
import sys
sys.path.append('/home/darragh/omscs6476/ps06')

from helper_classes import WeakClassifier, VJ_Classifier


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   ([int]): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """
    # folder=YALE_FACES_DIR; size=(32, 32)

    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]
    
    imgs = [cv2.imread(os.path.join(folder, img)) for img in images_files]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    imgs = [cv2.resize(img, tuple(size), interpolation = cv2.INTER_CUBIC) for img in imgs]
    imgs = [img.flatten() for img in imgs]
    labels = [int(f.split('.')[0][-2:]) for f in images_files ]
    imgs = np.vstack(imgs)
    labels = np.vstack(labels)
    
    return (imgs, labels)


def split_dataset(X, y, p, seed = 1):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    n = y.shape[0]
    np.random.seed(seed)
    idxtmp = np.random.choice(range(n), int(n*p), replace=False) 
    idx    = np.array([False]*n)
    idx[idxtmp] = True
    return X[idx], y[idx], X[~idx], y[~idx]
    # raise NotImplementedError


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """
    return np.mean(x, axis=0)
    # raise NotImplementedError


def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """
    
    M= X-np.array(X.mean(0),ndmin=2)
    eigenval, eigenvec = np.linalg.eigh(np.dot(M.T,M))
    eigenval = eigenval[::-1][:k]
    eigenvec = eigenvec.T[::-1][:k].T

    return (eigenvec, eigenval)
    # raise NotImplementedError


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
            
        for i in range(0, self.num_iterations):
            mod = WeakClassifier(X=self.Xtrain, y=self.ytrain, weights=self.weights)
            mod.train()
            mod_j = mod.predict(np.transpose(self.Xtrain))
            erridx = self.ytrain != mod_j
            err_sum = np.sum(self.weights[erridx])/np.sum(self.weights)
            alpha = 0.5 * np.log((1. - err_sum)/err_sum)
            self.weakClassifiers = np.append(self.weakClassifiers, mod)
            self.alphas = np.append(self.alphas, alpha)
            if err_sum > self.eps:
                self.weights[erridx] = self.weights[erridx] * np.exp(-alpha * mod_j[erridx] * self.ytrain[erridx])
            else:
                break
        

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the traini                                                            ng data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        y_trn_pred = self.predict(self.Xtrain)
        y_trn_act  = self.ytrain

        return ((y_trn_act==y_trn_pred).sum(), (y_trn_act!=y_trn_pred).sum())

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        
        
        y_pred = [[m.predict(np.transpose(X))] for m in self.weakClassifiers]
        
        for i in range(0, len(self.alphas)):
            y_pred[i] = np.array(y_pred[i]) * self.alphas[i]
        y_pred = np.sum(y_pred, axis=0)
        y_pred = y_pred[0]
        return np.sign(y_pred)


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        
        img = np.zeros(shape)
        y, x = self.position
        h,w    = self.size
        img[y:y+int(h/2)  , x:x+w] = 255
        img[y+int(h/2):y+h, x:x+w] = 126
        
        return img
    
    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)
        y, x = self.position
        h,w    = self.size

        img[y:y+h, x:x+int(w/2)]   = 255
        img[y:y+h, x+int(w/2):x+w] = 126
        return img

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        
        img = np.zeros(shape)

        y1, x1 = self.position

        h,w = self.size
        
        strip = int(h/3)

        #left right 1/3 is white, mid 1/3 is gray
        img[y1:y1+strip, x1:x1+w] = 255
        img[y1+strip:y1+strip+strip, x1:x1+w] = 126
        img[y1+strip+strip:y1+h, x1:x1+w] = 255

        return img
        
        img = np.zeros(shape)
        y, x = self.position
        h,w = self.size
        
        img[y:y+int(h/3),            x:x+w] = 255
        img[y+int(h/3):y+int(2*h/3), x:x+w] = 126
        img[y+int(2*h/3):y+h,        x:x+w] = 255
        

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        img = np.zeros(shape)
        y, x = self.position
        h,w = self.size
        
        img[y:y+h, x:x+int(w/3)]            = 255
        img[y:y+h, x+int(w/3):x+int(2*w/3)] = 126
        img[y:y+h, x+int(2*w/3):x+w]        = 255
        
        return img

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        
        img = np.zeros(shape)
        y, x = self.position
        h,w = self.size

        s1, s2 = int(h/2), int(w/2)

        #left right 1/3 is white, mid 1/3 is gray
        img[y:y+s1, x:x+s2] = 126
        img[y:y+s1, x+s2:x+w] = 255
        img[y+s1:y+h, x:x+s2] = 255
        img[y+s1:y+h, x+s2:x+w] = 126

        return img

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        


        h,w = self.size
        ii = ii.astype(np.float32)
        
        y, x = self.position[:2]
        y1, x1 = self.position[:2]

        if self.feat_type == (2, 1):
            y2, x2 = y+int(h/2)-1,  x1+w-1
            A = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=y+int(h/2)
            x1=x
            y2, x2 = y+int(h)-1, x1+w-1
            B = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            score = A-B


        if self.feat_type == (1, 2):
            y2, x2 = y1+int(h)-1,  x1+int(w/2)-1
            A = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=y
            x1=x+int(w/2)
            y2, x2 =y1+int(h)-1, x+int(w)-1
            B = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            score = A-B


        #left right 1/3 is white, mid 1/3 is gray

        if self.feat_type == (3, 1):
            y2 = y+int(h/3)-1
            x2 = x1+w-1
            A = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=y+int(h/3)
            x1=x
            y2, x2 =y+int(h/3)+int(h/3)-1, x1+w-1
            B = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=y+int(h/3)+int(h/3)
            x1=x
            y2, x2 =y+h-1, x1+w-1
            C = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            score = A-B+C


        if self.feat_type == (1, 3):
            y2 = y1+int(h)-1
            x2 = x1+int(w/3)-1
            A = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=y
            x1=x+int(w/3)
            y2, x2 = y1+int(h)-1, x+int(w/3)+int(w/3)-1
            B = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=y
            x1=x+int(w/3)+int(w/3)
            y2, x2 =y+h-1, x+int(w)-1
            C = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            score = A-B+C


        if self.feat_type == (2, 2):
            y2 = y1+int(h/2)-1
            x2 = x1+int(w/2)-1
            A = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=y
            x1=x+int(w/2)
            y2, x2 =y1+int(h/2)-1, x+w-1
            B = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=y+int(h/2)
            x1=x
            y2, x2 =y+h-1, x1+int(w/2)-1
            C = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=y+int(h/2)
            x1=x+int(w/2)
            y2, x2 =y+h-1, x+w-1
            D = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]
            score = -A+B+C-D

        return score


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """
    
    return [np.cumsum(np.cumsum(i, axis=0), axis=1) for i in images]



class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.iteritems():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei-1, sizej-1]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):
        
        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print " -- compute all scores --"
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print " -- select classifiers --"

        for i in range(num_classifiers):
            
            weights = weights / np.sum(weights)
            VJ = VJ_Classifier(scores,self.labels,weights)
            VJ.train()
            self.classifiers.append(VJ)
            
            B = VJ.error / (1.0 - VJ.error)
            alpha = np.log(1.0/B)
            
            et = [0 if VJ.predict(st) == lt else 1 for i, (st, lt) in enumerate(zip(scores, self.labels))]
            weights = [ w * np.power(B,1-e) for (w, e) in zip(weights, et)]
            
            self.alphas.append(alpha)
        
        print " -- select classifiers done --"


    def predict(self, images, thresh = 0.55):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """
        ii = convert_images_to_integral_images(images)
        
        scores = np.zeros((len(ii), len(self.haarFeatures)),dtype=np.float64)
        
        p = np.zeros((len(ii),len(self.classifiers)))
        limit = thresh * np.sum(self.alphas)
        print " -- predict classifiers --"
        
        for i in range(0,len(self.classifiers)):
            idx = self.classifiers[i].feature
            for j in range(0,len(ii)):
                scores[j, idx]= self.haarFeatures[idx].evaluate(ii[j])    
        
        for i in range(0,len(self.classifiers)):
            p[:, i] = [ self.classifiers[i].predict(scores[j]) * self.alphas[i] for j in range(0,len(ii))]
        
        result = [1 if np.sum(x) >= limit else -1 for x in p]
        
        return result
    
    
    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        # image, filename
        frame = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Define the window size
        ws_r = 26
        ws_c = 26

        # Crop windows using a steps size
        x = []
        p1 = []
        p2 = []
        for r in range(0,gray.shape[0] - ws_r, 1):
            for c in range(0,gray.shape[1] - ws_c,1):
                p1.append([c, r])
                p2.append([c+ws_c, r+ws_r])
                window = gray[r:r+ws_r,c:c+ws_c]
                x.append(np.array(window))
        predictions = self.predict(x, thresh = 0.5)
        #predictions = VJ.predict(x)

        #Average positive predictions
        p1mean = (np.array(p1)[np.array(predictions) == 1]).mean(axis=0).astype(np.int)
        p2mean = (np.array(p2)[np.array(predictions) == 1]).mean(axis=0).astype(np.int)
        cv2.rectangle(frame, tuple(p1mean), tuple(p2mean), (0,255,255), 2)
        # Export
        cv2.imwrite(os.path.join("output", filename + '.png'), frame)
        