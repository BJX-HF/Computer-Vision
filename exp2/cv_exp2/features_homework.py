import scipy
from scipy import ndimage
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial

import transformations


def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    def saveHarrisImage(self, harrisImage, srcImage):
        '''
        Saves a visualization of the harrisImage, by overlaying the harris
        response image as red over the srcImage.

        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
            harrisImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        '''
        outshape = [harrisImage.shape[0], harrisImage.shape[1], 3]
        outImage = np.zeros(outshape)
        # Make a grayscale srcImage as a background
        srcNorm = srcImage * (0.3 * 255 / (np.max(srcImage) + 1e-50))
        outImage[:, :, :] = np.expand_dims(srcNorm, 2)

        # Add in the harris keypoints as red
        outImage[:, :, 2] += harrisImage * (4 * 255 / (np.max(harrisImage)) + 1e-50)
        cv2.imwrite("harris.png", outImage)

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'.  See the project page
        # for direction on how to do this. Also compute an orientation
        # for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN
        dx = ndimage.sobel(srcImage, axis=1, mode="reflect")
        dy = ndimage.sobel(srcImage, axis=0, mode="reflect")
        Ix2 = ndimage.gaussian_filter(dx*dx, 0.5)
        Iy2 = ndimage.gaussian_filter(dy*dy, 0.5)
        Ixy = ndimage.gaussian_filter(dx*dy, 0.5)

        for i in range(height):
            for j in range(width):
                H = np.array([[Ix2[i][j], Ixy[i][j]], [Ixy[i][j], Iy2[i][j]]])
                eig = np.linalg.eigvals(H)
                det = eig[0]*eig[1]
                trace = eig[0]+eig[1]
                R = det - 0.1 * trace ** 2
                harrisImage[i][j] = R
                rad = np.arctan2(dy[i][j], dx[i][j])
                degree = rad * 180 / np.pi
                orientationImage[i][j] = degree
        # raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        # Save the harris image as harris.png for the website assignment
        self.saveHarrisImage(harrisImage, srcImage)

        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        '''
        destImage = np.zeros_like(harrisImage, np.bool)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
        harrisDuplicate = harrisImage.copy()
        height = harrisDuplicate.shape[0]
        width = harrisDuplicate.shape[1]
        raw = np.zeros(width)
        col = np.zeros(height + 6)
        for i in range(3):
            harrisDuplicate = np.insert(harrisDuplicate, 0, values=raw, axis=0)
            harrisDuplicate = np.insert(harrisDuplicate, harrisDuplicate.shape[0], values=raw, axis=0)
        for j in range(3):
            harrisDuplicate = np.insert(harrisDuplicate, 0, values=col, axis=1)
            harrisDuplicate = np.insert(harrisDuplicate, harrisDuplicate.shape[1], values=col, axis=1)
        for i in range(height):
            for j in range(width):
                mat7x7 = harrisDuplicate[i:i + 7, j:j + 7]
                if (harrisImage[i][j] == np.max(mat7x7)): destImage[i][j] = True
        # raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return destImage

    #进行局部非极大值抑制
    def NotMaxSuppress(self,harrisImg,size):
        # d_mat = np.ones((5,5))
        # dilated = cv2.dilate(harrisImg,d_mat)
        # localMax = cv2.compare(harrisImg,dilated,cv2.CMP_EQ)
        # return localMax
        h = harrisImg.shape[0]
        w = harrisImg.shape[1]
        harrisImg1 = harrisImg.copy()
        rol = np.zeros(w)
        for i in range(size//2):
            harrisImg1 = np.insert(harrisImg1,0,rol,axis=0)
            harrisImg1 = np.insert(harrisImg1,harrisImg1.shape[0],rol,axis=0)
        col = np.zeros(harrisImg1.shape[0])
        for m in range(size//2):
            harrisImg1 = np.insert(harrisImg1,0,values=col,axis=1)
            harrisImg1 = np.insert(harrisImg1,harrisImg1.shape[1],values=col,axis=1)
        H = harrisImg.shape[0]
        W = harrisImg.shape[1]
        for i in range(H):
            for j in range(W):
                local_mat = harrisImg1[i:i+size,j:j+size]
                max = local_mat.max()
                if(harrisImg[i][j]==max and max!=0): continue
                else: harrisImg[i][j] = 0


    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()

                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                # TODO-BLOCK-BEGIN
                f.size = 10
                f.pt = (x, y)
                f.angle = orientationImage[y][x]
                f.response = harrisImage[y][x]
                # raise Exception("TODO in features.py not implemented")
                # TODO-BLOCK-END

                features.append(f)

        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
        detector = cv2.ORB_create()
        return detector.detect(image,None)

## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))
        image_extended = grayImage.copy()
        row = np.zeros(image.shape[1])
        vol = np.zeros(image.shape[0] + 4)
        for i in range(2):
            image_extended = np.insert(image_extended, 0, values=row, axis=0)
            image_extended = np.insert(image_extended, image_extended.shape[0], values=row, axis=0)
        for j in range(2):
            image_extended = np.insert(image_extended, 0, values=vol, axis=1)
            image_extended = np.insert(image_extended, image_extended.shape[1], values=vol, axis=1)

        for i, f in enumerate(keypoints):
            x, y = f.pt
            x, y = int(x), int(y)

            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            # TODO-BLOCK-BEGIN
            mat5x5 = image_extended[y:y+5,x:x+5]
            desc[i] = np.reshape(mat5x5,(1,25))
            # raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.


            # TODO-BLOCK-BEGIN
            x, y = f.pt
            x, y = int(x), int(y)

            t_mat = transformations.get_trans_mx(np.array([-x, -y]))
            rot_mat = transformations.get_rot_mx(-f.angle*np.pi / 180)
            scale_mat = transformations.get_scale_mx(0.2, 0.2)
            t_mat2 = np.float32([[1, 0, 4], [0, 1, 4], [0, 0, 1]])
            transMx1 = np.dot(rot_mat, t_mat)
            transMx2 = np.dot(scale_mat, transMx1)
            transMx3 = np.dot(t_mat2, transMx2)
            transMx = np.delete(transMx3, 2, axis=0)


            # raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix

            destImage = cv2.warpAffine(grayImage, transMx,
(windowSize, windowSize), flags=cv2.INTER_LINEAR)

            # TODO 6: Normalize the descriptor to have zero mean and unit
            # variance. If the variance is zero then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN
            destImage = np.reshape(destImage,(1,64))
            if np.std(destImage) >= 1e-5: destImage = (destImage - np.mean(destImage))/np.std(destImage)
            else: destImage = np.zeros(windowSize*windowSize)
            desc[i] = destImage

            # raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

        return desc

def get_rotmx(angle):
    rotmx = np.array([[math.cos(angle), -math.sin(angle), 0],
                      [math.sin(angle), math.cos(angle), 0],
                      [0, 0, 1]])
    return rotmx


def get_transmx(trans_x, trans_y):
    transmx = np.eye(3)
    transmx[0, 2] = trans_x
    transmx[1, 2] = trans_y
    return transmx


def get_scalemx(s_x, s_y):
    scalemx = np.eye(3)
    scalemx[0, 0] = s_x
    scalemx[1, 1] = s_y
    return scalemx

class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching.  This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN
        delta = cdist(desc1,desc2,metric="euclidean")
        min_index = np.argmin(delta,axis=1)
        for i,k in enumerate(min_index):
            dmatch = cv2.DMatch();
            dmatch.queryIdx = i
            dmatch.trainIdx = k
            dmatch.distance = delta[i][k]
            matches.append(dmatch)
        # raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # You don't need to threshold matches in this function
        # TODO-BLOCK-BEGIN
        delta = cdist(desc1, desc2, metric="euclidean")
        min_index = np.argmin(delta, axis=1)
        delta_dup = delta.copy()
        for i,row in delta_dup:
            row[min_index[i]] = row.max()
        second_min_index = np.argmin(delta_dup,axis=1)

        for i, k in enumerate(min_index):
            k_dup = second_min_index[i]
            dmatch = cv2.DMatch()
            dmatch.queryIdx = i
            dmatch.trainIdx = k
            dmatch.distance = delta[i][k]/delta[i][k_dup]
            matches.append(dmatch)
        # raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))

detector  = HarrisKeypointDetector()
picture = cv2.imread("resources/yosemite/yosemite1.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("gray",picture)
picture = np.float32(picture)
picture = (picture-picture.min())/(picture.max()-picture.min())
harrisimg,orientationimg = detector.computeHarrisValues(picture)
cv2.waitKey(0)
