/*

Copyright 2016 Mark Muth

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

#include "KittiDataset.h"

#include <string>
#include <vector>
#include <iterator>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/format.hpp>

#include <pcl/filters/crop_box.h>
#include <pcl/common/io.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/search/kdtree.h>

#include <eigen3/Eigen/Core>

 #include <QImageReader>

KittiDataset::KittiDataset(int dataset) :
    _dataset(dataset),
    _number_of_frames(0)
{
    if (!boost::filesystem::exists(KittiConfig::getPointCloudPath(_dataset)))
    {
        std::cerr << "Error in KittiDataset: Data set path "
                  << (KittiConfig::getPointCloudPath(_dataset)).string()
                  << " does not exist!" << std::endl;
        return;
    }
    if (!boost::filesystem::exists(KittiConfig::getPointCloudPath(_dataset, 0)))
    {
        std::cerr << "Error in KittiDataset: No point cloud was found at "
                  << (KittiConfig::getPointCloudPath(_dataset, 0)).string()
                  << std::endl;
        return;
    }
    if (!boost::filesystem::exists(KittiConfig::getTrackletsPath(_dataset)))
    {
        std::cerr << "Error in KittiDataset: No tracklets were found at "
                  << (KittiConfig::getTrackletsPath(_dataset)).string()
                  << std::endl;
        return;
    }
    if (!boost::filesystem::exists(KittiConfig::getImagePath(_dataset)))
    {
        std::cerr << "Error in KittiDataset: Data set path "
                  << (KittiConfig::getImagePath(_dataset)).string()
                  << " does not exist!" << std::endl;
        return;
    }
    if (!boost::filesystem::exists(KittiConfig::getImagePath(_dataset, 0)))
    {
        std::cerr << "Error in KittiDataset: No image was found at "
                  << (KittiConfig::getImagePath(_dataset, 0)).string()
                  << std::endl;
        return;
    }
    if (!boost::filesystem::exists(KittiConfig::getVeloToCameraCalibrationPath(_dataset)))
    {
        std::cerr << "Error in KittiDataset: Data set path "
                  << (KittiConfig::getImagePath(_dataset)).string()
                  << " does not exist!" << std::endl;
        return;
    }

    initNumberOfFrames();
    initTracklets();
}

int KittiDataset::getNumberOfFrames()
{
    return _number_of_frames;
}

KittiPointCloud::Ptr KittiDataset::getPointCloud(int frameId)
{
    KittiPointCloud::Ptr cloud(new KittiPointCloud);
    std::fstream file(KittiConfig::getPointCloudPath(_dataset, frameId).c_str(), std::ios::in | std::ios::binary);
    if(file.good()){
        file.seekg(0, std::ios::beg);
        int i;
        for (i = 0; file.good() && !file.eof(); i++) {
            KittiPoint point;
            file.read((char *) &point.x, 3*sizeof(float));
            file.read((char *) &point.intensity, sizeof(float));
            cloud->push_back(point);
        }
        file.close();
    }
    return cloud;
}

QImage KittiDataset::getImage(int frameId)
{
   std::string fileName = KittiConfig::getImagePath(_dataset, frameId).string();
   QString qFileName = QString::fromStdString(fileName);
   QImageReader imageReader(qFileName);
   return imageReader.read();
}

void KittiDataset::getCalibration()
{
    std::string data;
    std::ifstream file(KittiConfig::getVeloToCameraCalibrationPath(_dataset).c_str());
    int lineNumber = 0;
    Eigen::Matrix3f rotation;
    Eigen::Vector4f translation;
    if(file.good()){
         std::string data;

         while( std::getline( file, data ) )
         {
             if(lineNumber == 1)
             {
                 // rotationmatrix
                 std::stringstream ss(data);
                 std::istream_iterator<std::string> begin(ss);
                 std::istream_iterator<std::string> end;
                 std::vector<std::string> tokens(begin, end);

                 std::stringstream conversionToFloat;
                 conversionToFloat.str(tokens[1]);
                 conversionToFloat >> rotation(0, 0);
                 conversionToFloat.clear();

                 conversionToFloat.str(tokens[2]);
                 conversionToFloat >> rotation(0, 1);
                 conversionToFloat.clear();

                 conversionToFloat.str(tokens[3]);
                 conversionToFloat >> rotation(0, 2);
                 conversionToFloat.clear();

                 conversionToFloat.str(tokens[4]);
                 conversionToFloat >> rotation(1, 0);
                 conversionToFloat.clear();

                 conversionToFloat.str(tokens[5]);
                 conversionToFloat >> rotation(1, 1);
                 conversionToFloat.clear();

                 conversionToFloat.str(tokens[6]);
                 conversionToFloat >> rotation(1, 2);
                 conversionToFloat.clear();

                 conversionToFloat.str(tokens[7]);
                 conversionToFloat >> rotation(2, 0);
                 conversionToFloat.clear();

                 conversionToFloat.str(tokens[8]);
                 conversionToFloat >> rotation(2, 1);
                 conversionToFloat.clear();

                 conversionToFloat.str(tokens[9]);
                 conversionToFloat >> rotation(2, 2);
                 conversionToFloat.clear();
             }
             else if(lineNumber == 2)
             {
                 // translation vector
                 std::stringstream ss(data);
                 std::istream_iterator<std::string> begin(ss);
                 std::istream_iterator<std::string> end;
                 std::vector<std::string> tokens(begin, end);

                 std::stringstream conversionToFloat;
                 conversionToFloat.str(tokens[1]);
                 conversionToFloat >> translation(0);
                 conversionToFloat.clear();

                 conversionToFloat.str(tokens[2]);
                 conversionToFloat >> translation(1);
                 conversionToFloat.clear();

                 conversionToFloat.str(tokens[3]);
                 conversionToFloat >> translation(2);
                 conversionToFloat.clear();

                 translation(3) = 1.0;
             }
             lineNumber++;
         }

        }
    file.close();

    _camera0UnrectToVelodyne.setIdentity();
    _camera0UnrectToVelodyne.block<3,3>(0,0) = rotation;
    _camera0UnrectToVelodyne.rightCols<1>() = translation;

}

std::vector<std::vector<Eigen::Vector4f> > KittiDataset::getTrackletBoundingBoxes(const KittiTracklet& tracklet, int frameId)
{
    std::vector<std::vector<Eigen::Vector4f> > boundingBoxes;

    double d = std::sqrt(std::pow(tracklet.l, 2) + std::pow(tracklet.w, 2)) / 2.0;

    for(int pose_number = 0; pose_number < tracklet.poses.size(); pose_number++)
    {
        std::vector<Eigen::Vector4f> poses;
        Tracklets::tPose tpose = tracklet.poses.at(pose_number);
        Eigen::Vector4f point1((float) tpose.tx + d * std::sin((float) tpose.rz),  (float) tpose.ty + d * std::cos((float) tpose.rz), (float) tpose.tz + tracklet.h, 1);
        Eigen::Vector4f point2((float) tpose.tx - d * std::sin((float) tpose.rz),  (float) tpose.ty + d * std::cos((float) tpose.rz), (float) tpose.tz + tracklet.h, 1);
        Eigen::Vector4f point3((float) tpose.tx - d * std::sin((float) tpose.rz),  (float) tpose.ty - d * std::cos((float) tpose.rz), (float) tpose.tz + tracklet.h, 1);
        Eigen::Vector4f point4((float) tpose.tx + d * std::sin((float) tpose.rz),  (float) tpose.ty - d * std::cos((float) tpose.rz), (float) tpose.tz + tracklet.h, 1);
        Eigen::Vector4f point5((float) tpose.tx + d * std::sin((float) tpose.rz),  (float) tpose.ty + d * std::cos((float) tpose.rz), (float) tpose.tz, 1);
        Eigen::Vector4f point6((float) tpose.tx - d * std::sin((float) tpose.rz),  (float) tpose.ty + d * std::cos((float) tpose.rz), (float) tpose.tz, 1);
        Eigen::Vector4f point7((float) tpose.tx - d * std::sin((float) tpose.rz),  (float) tpose.ty - d * std::cos((float) tpose.rz), (float) tpose.tz, 1);
        Eigen::Vector4f point8((float) tpose.tx + d * std::sin((float) tpose.rz),  (float) tpose.ty - d * std::cos((float) tpose.rz), (float) tpose.tz, 1);

        poses.push_back(point1);
        poses.push_back(point2);
        poses.push_back(point3);
        poses.push_back(point4);
        poses.push_back(point5);
        poses.push_back(point6);
        poses.push_back(point7);
        poses.push_back(point8);
        boundingBoxes.push_back(poses);
    }

    return boundingBoxes;
}

KittiPointCloud::Ptr KittiDataset::getTrackletPointCloud(KittiPointCloud::Ptr& pointCloud, const KittiTracklet& tracklet, int frameId)
{
    int pose_number = frameId - tracklet.first_frame;
    Tracklets::tPose tpose = tracklet.poses.at(pose_number);

    Eigen::Vector4f minPoint(-tracklet.l / 2.0f, -tracklet.w / 2.0, -tracklet.h / 2.0, 1.0f);
    Eigen::Vector4f maxPoint( tracklet.l / 2.0f,  tracklet.w / 2.0,  tracklet.h / 2.0, 1.0f);
    Eigen::Vector3f boxTranslation((float) tpose.tx, (float) tpose.ty, (float) tpose.tz + tracklet.h / 2.0f);
    Eigen::Vector3f boxRotation((float) tpose.rx, (float) tpose.ry, (float) tpose.rz);

    KittiPointCloud::Ptr trackletPointCloud(new KittiPointCloud());
    pcl::CropBox<KittiPoint> cropFilter;
    cropFilter.setInputCloud(pointCloud);
    cropFilter.setMin(minPoint);
    cropFilter.setMax(maxPoint);
    cropFilter.setTranslation(boxTranslation);
    cropFilter.setRotation(boxRotation);
    cropFilter.filter(*trackletPointCloud);

    return trackletPointCloud;
}

Tracklets& KittiDataset::getTracklets()
{
    return _tracklets;
}

int KittiDataset::getLabel(const char* labelString)
{
    if (strcmp(labelString, "Car") == 0)
    {
        return 0;
    }
    if (strcmp(labelString, "Van") == 0)
    {
        return 1;
    }
    if (strcmp(labelString, "Truck") == 0)
    {
        return 2;
    }
    if (strcmp(labelString, "Pedestrian") == 0)
    {
        return 3;
    }
    if (strcmp(labelString, "Person (sitting)") == 0)
    {
        return 4;
    }
    if (strcmp(labelString, "Cyclist") == 0)
    {
        return 5;
    }
    if (strcmp(labelString, "Tram") == 0)
    {
        return 6;
    }
    if (strcmp(labelString, "Misc") == 0)
    {
        return 7;
    }
    std::cerr << "Error in KittiDataset:getLabel(): Not a valid label string: "
              << labelString << std::endl;
    return -1;
}

void KittiDataset::getColor(const char* labelString, int& r, int& g, int& b)
{
    if (strcmp(labelString, "Car") == 0)
    {
        r = 255; g = 0; b = 0;
        return;
    }
    if (strcmp(labelString, "Van") == 0)
    {
        r = 191; g = 0; b = 0;
        return;
    }
    if (strcmp(labelString, "Truck") == 0)
    {
        r = 127; g = 0; b = 0;
        return;
    }
    if (strcmp(labelString, "Pedestrian") == 0)
    {
        r = 0; g = 255; b = 0;
        return;
    }
    if (strcmp(labelString, "Person (sitting)") == 0)
    {
        r = 0; g = 128; b = 0;
        return;
    }
    if (strcmp(labelString, "Cyclist") == 0)
    {
        r = 0; g = 0; b = 255;
        return;
    }
    if (strcmp(labelString, "Tram") == 0)
    {
        r = 0; g = 255; b = 255;
        return;
    }
    if (strcmp(labelString, "Misc") == 0)
    {
        r = 255; g = 255; b = 0;
        return;
    }
    std::cerr << "Error in KittiDataset:getColor(): Not a valid label string: "
              << labelString << std::endl;
    r = 128; g = 128; b = 128;
}

void KittiDataset::getColor(int label, int& r, int& g, int& b)
{
    if (label == 0)
    {
        r = 255; g = 0; b = 0;
        return;
    }
    if (label == 1)
    {
        r = 191; g = 0; b = 0;
        return;
    }
    if (label == 2)
    {
        r = 127; g = 0; b = 0;
        return;
    }
    if (label == 3)
    {
        r = 0; g = 255; b = 0;
        return;
    }
    if (label == 4)
    {
        r = 0; g = 128; b = 0;
        return;
    }
    if (label == 5)
    {
        r = 0; g = 0; b = 255;
        return;
    }
    if (label == 6)
    {
        r = 0; g = 255; b = 255;
        return;
    }
    if (label == 7)
    {
        r = 255; g = 255; b = 0;
        return;
    }
    std::cerr << "Error in KittiDataset:getColor(): Not a valid label: "
              << label << std::endl;
    r = 128; g = 128; b = 128;
}

std::string KittiDataset::getLabelString(int label)
{
    if (label == 0)
    {
        return "Car";
    }
    if (label == 1)
    {
        return "Van";
    }
    if (label == 2)
    {
        return "Truck";
    }
    if (label == 3)
    {
        return "Pedestrian";
    }
    if (label == 4)
    {
        return "Person (sitting)";
    }
    if (label == 5)
    {
        return "Cyclist";
    }
    if (label == 6)
    {
        return "Tram";
    }
    if (label == 7)
    {
        return "Misc";
    }
    std::cerr << "Error in KittiDataset:getLabelString(): Not a valid label: "
              << label << std::endl;
    return "Unknown";
}

void KittiDataset::initNumberOfFrames()
{
    boost::filesystem::directory_iterator dit(KittiConfig::getPointCloudPath(_dataset));
    boost::filesystem::directory_iterator eit;

    while(dit != eit)
    {
        if(boost::filesystem::is_regular_file(*dit) && dit->path().extension() == ".bin")
        {
            _number_of_frames++;
        }
        ++dit;
    }
}

void KittiDataset::initTracklets()
{
    boost::filesystem::path trackletsPath = KittiConfig::getTrackletsPath(_dataset);
    _tracklets.loadFromFile(trackletsPath.string());
}

Eigen::Vector3f KittiDataset::transformPointFromVeloToImage(const Eigen::Vector4f& point){
    Eigen::Matrix4f R_rect0;
    R_rect0.setIdentity(4,4);

    R_rect0(0,0) = 9.998817e-01;
    R_rect0(0,1) = 9.837760e-03;
    R_rect0(0,2) =  -7.445048e-03;
    R_rect0(1,0) = -9.869795e-03;
    R_rect0(1,1) = 9.999421e-01;
    R_rect0(1,2) =  -4.278459e-03;
    R_rect0(2,0) = 7.402527e-03;
    R_rect0(2,1) = 4.351614e-03;
    R_rect0(2,2) =  9.999631e-01;

    Eigen::Matrix4f R_rect2;
    R_rect2.setIdentity(4,4);

    R_rect2(0,0) = 9.998817e-01;
    R_rect2(0,1) = 1.511453e-02;
    R_rect2(0,2) =  -2.841595e-03;
    R_rect2(1,0) = -1.511724e-02;
    R_rect2(1,1) = 9.998853e-01;
    R_rect2(1,2) = -9.338510e-04;
    R_rect2(2,0) = 2.827154e-03;
    R_rect2(2,1) = 9.766976e-04;
    R_rect2(2,2) =  9.999955e-01;

    Eigen::MatrixXf P_rect2(3,4);
    P_rect2(0,0) = 7.215377e+02;
    P_rect2(0,1) = 0;
    P_rect2(0,2) = 6.095593e+02;
    P_rect2(0,3) = 4.485728e+01;
    P_rect2(1,0) = 0.000000e+00;
    P_rect2(1,1) = 7.215377e+02;
    P_rect2(1,2) = 1.728540e+02;
    P_rect2(1,3) = 2.163791e-01;
    P_rect2(2,0) = 0.000000e+00;
    P_rect2(2,1) = 0.000000e+00;
    P_rect2(2,2) = 1.000000e+00;
    P_rect2(2,3) = 2.745884e-03;

    Eigen::Matrix4f T0;
    T0.setIdentity(4,4);
    T0(0,3) = P_rect2(0,3) / P_rect2(0,0);

    Eigen::Matrix4f cameraToVelodyne = T0 * R_rect0 * _camera0UnrectToVelodyne;

    Eigen::Vector3f pointImage = P_rect2 * R_rect2 * cameraToVelodyne * point;
    pointImage /= pointImage(2);

    Eigen::Vector4f pointCameraCoordinates = R_rect2 * cameraToVelodyne * point;

    return pointImage;

}

//#undef DEBUG_OUTPUT_ENABLED
