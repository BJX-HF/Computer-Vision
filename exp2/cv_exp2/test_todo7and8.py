import unittest
import features_homework
import features
import numpy as np
import cv2

#test TODO7
class TestSSDFeatureMatcher(unittest.TestCase):
    def setUp(self):
        self.small_height = 10
        self.small_width = 8
        self.big_height = 50
        self.big_width = 40
        HKD = features.HarrisKeypointDetector()
        MOPS = features.MOPSFeatureDescriptor()
        self.big_img1 = np.random.rand(self.big_height, self.big_width,3)*255
        self.big_img2 = np.random.rand(self.big_height, self.big_width,3)*255
        self.small_img1 = np.random.rand(self.small_height, self.small_width,3)*255
        self.small_img2 = np.random.rand(self.small_height, self.small_width,3)*255
        self.big_img1_keypoints = HKD.detectKeypoints(self.big_img1)
        self.big_img2_keypoints = HKD.detectKeypoints(self.big_img2)
        self.big1_desc = MOPS.describeFeatures(self.big_img1,self.big_img1_keypoints)
        self.big2_desc = MOPS.describeFeatures(self.big_img2,self.big_img2_keypoints)
        self.small_img1_keypoints = HKD.detectKeypoints(self.small_img1)
        self.small_img2_keypoints = HKD.detectKeypoints(self.small_img2)
        self.small1_desc = MOPS.describeFeatures(self.small_img1, self.small_img1_keypoints)
        self.small2_desc = MOPS.describeFeatures(self.small_img2, self.small_img2_keypoints)
    def test_TODO7_big_img(self):
        teacherSSD = features.SSDFeatureMatcher()
        studentSSD = features_homework.SSDFeatureMatcher()
        teachermatches = teacherSSD.matchFeatures(self.big1_desc,self.big2_desc)
        studentmatches = studentSSD.matchFeatures(self.big1_desc,self.big2_desc)
        self.assertEqual(len(teachermatches),len(studentmatches))
        for i in range(len(teachermatches)):
            self.assertEqual(teachermatches[i].distance,studentmatches[i].distance)
            self.assertEqual(teachermatches[i].queryIdx,studentmatches[i].queryIdx)
            self.assertEqual(teachermatches[i].trainIdx,studentmatches[i].trainIdx)
    def test_TODO7_small_img(self):
        teacherSSD = features.SSDFeatureMatcher()
        studentSSD = features_homework.SSDFeatureMatcher()
        teachermatches = teacherSSD.matchFeatures(self.small1_desc,self.small2_desc)
        studentmatches = studentSSD.matchFeatures(self.small1_desc,self.small2_desc)
        self.assertEqual(len(teachermatches),len(studentmatches))
        for i in range(len(teachermatches)):
            self.assertEqual(teachermatches[i].distance,studentmatches[i].distance)
            self.assertEqual(teachermatches[i].queryIdx,studentmatches[i].queryIdx)
            self.assertEqual(teachermatches[i].trainIdx,studentmatches[i].trainIdx)

#test TODO8
class TestRatioFeatureMatcher(unittest.TestCase):
    def setUp(self):
        self.small_height = 10
        self.small_width = 8
        self.big_height = 50
        self.big_width = 40
        HKD = features.HarrisKeypointDetector()
        MOPS = features.MOPSFeatureDescriptor()
        self.big_img1 = np.random.rand(self.big_height, self.big_width,3)*255
        self.big_img2 = np.random.rand(self.big_height, self.big_width,3)*255
        self.small_img1 = np.random.rand(self.small_height, self.small_width,3)*255
        self.small_img2 = np.random.rand(self.small_height, self.small_width,3)*255
        self.big_img1_keypoints = HKD.detectKeypoints(self.big_img1)
        self.big_img2_keypoints = HKD.detectKeypoints(self.big_img2)
        self.big1_desc = MOPS.describeFeatures(self.big_img1,self.big_img1_keypoints)
        self.big2_desc = MOPS.describeFeatures(self.big_img2,self.big_img2_keypoints)
        self.small_img1_keypoints = HKD.detectKeypoints(self.small_img1)
        self.small_img2_keypoints = HKD.detectKeypoints(self.small_img2)
        self.small1_desc = MOPS.describeFeatures(self.small_img1, self.small_img1_keypoints)
        self.small2_desc = MOPS.describeFeatures(self.small_img2, self.small_img2_keypoints)
    def test_TODO8_big_img(self):
        teacherRatio = features.SSDFeatureMatcher()
        studentRatio = features_homework.SSDFeatureMatcher()
        teachermatches = teacherRatio.matchFeatures(self.big1_desc,self.big2_desc)
        studentmatches = studentRatio.matchFeatures(self.big1_desc,self.big2_desc)
        self.assertEqual(len(teachermatches),len(studentmatches))
        for i in range(len(teachermatches)):
            self.assertEqual(teachermatches[i].distance,studentmatches[i].distance)
            self.assertEqual(teachermatches[i].queryIdx,studentmatches[i].queryIdx)
            self.assertEqual(teachermatches[i].trainIdx,studentmatches[i].trainIdx)
    def test_TODO8_small_img(self):
        teacherRatio = features.SSDFeatureMatcher()
        studentRatio = features_homework.SSDFeatureMatcher()
        teachermatches = teacherRatio.matchFeatures(self.small1_desc,self.small2_desc)
        studentmatches = studentRatio.matchFeatures(self.small1_desc,self.small2_desc)
        self.assertEqual(len(teachermatches),len(studentmatches))
        for i in range(len(teachermatches)):
            self.assertEqual(teachermatches[i].distance,studentmatches[i].distance)
            self.assertEqual(teachermatches[i].queryIdx,studentmatches[i].queryIdx)
            self.assertEqual(teachermatches[i].trainIdx,studentmatches[i].trainIdx)

if __name__ == '__main__':
    unittest.main()
