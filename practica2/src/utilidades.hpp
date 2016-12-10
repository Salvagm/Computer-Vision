#ifndef UTILIDADES_H_
#define UTILIDADES_H_


#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET


#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <boost/thread/thread.hpp>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


// PCL specific includes
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>

#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/agast_2d.h>
#include <pcl/keypoints/harris_3d.h>

#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/pfh.h>
#include <pcl/features/narf_descriptor.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/range_image_border_extractor.h>

enum PCL_DETECTORS
	{
		K_ISS=0,
		K_SIFT=1,
		K_NARF=2,
		K_HARRIS3D=3,
	};
 
enum PCL_DESCRIPTORS
	{
   	 F_FPFH=0,
   	 F_PFH=1,
   	 F_SHOT=2,
   	 F_SHOTCOLOR=3,
   	 F_NARF=4
	};

using namespace cv;
using namespace std;

typedef pcl::PointXYZRGB Rgbpoint;
typedef pcl::PointCloud<pcl::PointXYZRGB> Cloudrgb;
typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr Cloudrgbptr;

typedef pcl::PointCloud<pcl::Narf36>::Ptr narfDesriptorptr;
typedef pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhDescriptorptr;
typedef pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhDescriptorptr;
typedef pcl::PointCloud<pcl::SHOT352>::Ptr shotDescriptorptr;
typedef pcl::PointCloud<pcl::SHOT1344>::Ptr shotColorDescriptorptr;

void tick(struct timespec &ts1)
{
	clock_gettime( CLOCK_REALTIME, &ts1 );
}

void tack(struct timespec &ts1, struct timespec &ts2)
{
	clock_gettime( CLOCK_REALTIME, &ts2 );
	float tiempo = (float) ( 1.0*(1.0*ts2.tv_nsec - ts1.tv_nsec*1.0)*1e-9+ 1.0*ts2.tv_sec - 1.0*ts1.tv_sec );

	PCL_INFO("TIEMPO EN REALIZAR OPERACION : %f\n",tiempo);
}
class ImageCloudContainer{
	public:

		ImageCloudContainer()
		{
			ready= false;
		}

		void storeCloud(const Cloudrgbptr & cloudPoints){this->cloudPoints= cloudPoints;}
		void setTransformada(const Cloudrgbptr & transformada){this->transformada= transformada;}
		void setReady(){ready= true;}
		void setKeypointsIndices(const pcl::PointCloud<int> keypointsIndice){keypoints_indices=keypointsIndice;}
		void setRangeImage(const pcl::RangeImage &range_image){this->range_image=range_image;}
		void setKeypoints(const Cloudrgbptr &keyPointsXYZ){	this->keyPointsXYZ= keyPointsXYZ;}
		void setNARFDescriptor(const narfDesriptorptr &narf_descriptors ){this->narf_descriptors= narf_descriptors;	}
		void setFPFHDescriptor(const fpfhDescriptorptr &fpfhDescriptors){this->fpfh_descriptors=fpfhDescriptors; 	}
		void setPFHDescriptor(const pfhDescriptorptr &pfhDescriptors){this->pfh_descriptors=pfhDescriptors; }
		void setSHOTDescriptor(const shotDescriptorptr &shotDesriptors){this->shot_descriptors=shotDesriptors; }
		void setSHOTColorDescriptor(const shotColorDescriptorptr &shotColorDesriptors){this->shotColor_descriptors= shotColorDesriptors;}

		Cloudrgbptr &getTransformada() { return this->transformada;}
		pfhDescriptorptr  &getPFHDescriptor()	{	return this->pfh_descriptors;}
		fpfhDescriptorptr &getFPFHDescriptor()	{return this->fpfh_descriptors;	}
		narfDesriptorptr &getNARFDescriptor() 	{return this->narf_descriptors;	}
		shotDescriptorptr &getSHOTDescriptor()	{return this->shot_descriptors;	}
		shotColorDescriptorptr &getSHOTColorDescriptor(){return this->shotColor_descriptors;}
		pcl::RangeImage &getRangeImage(){return this->range_image;}
		pcl::PointCloud<int> &getKeypointsIndices(){return this->keypoints_indices;}
		Cloudrgbptr &getKeypoints(){return this->keyPointsXYZ;	}
		Cloudrgbptr &getCloudPoint(){return this->cloudPoints;}
		bool isReady(){return this->ready;}



		ImageCloudContainer &operator=(const ImageCloudContainer &imgCloudCont)
		{
			if(this!=&imgCloudCont)
			{
				this->range_image= imgCloudCont.range_image;
				this->keyPointsXYZ= imgCloudCont.keyPointsXYZ;
				this->narf_descriptors= imgCloudCont.narf_descriptors;
				this->fpfh_descriptors= imgCloudCont.fpfh_descriptors;
				this->cloudPoints = imgCloudCont.cloudPoints;
				this->pfh_descriptors= imgCloudCont.pfh_descriptors;
				this->ready= imgCloudCont.ready;
				this->shot_descriptors= imgCloudCont.shot_descriptors;
				this->shotColor_descriptors= imgCloudCont.shotColor_descriptors;
				keypoints_indices= imgCloudCont.keypoints_indices;
			}

			return (*this);

		}
	private:
		Cloudrgbptr cloudPoints;
		Cloudrgbptr transformada;
		pcl::RangeImage range_image;
		Cloudrgbptr keyPointsXYZ; 	//Cloud Keypoints
		pcl::PointCloud<int> keypoints_indices;
		bool ready ;
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr 	fpfh_descriptors;		// descriptores FPFH
		pcl::PointCloud<pcl::Narf36>::Ptr 			narf_descriptors;		//descriptores NARF
		pcl::PointCloud<pcl::PFHSignature125>::Ptr 	pfh_descriptors;		// descriptores PFH
		pcl::PointCloud<pcl::SHOT352>::Ptr 			shot_descriptors; 		//descriptores SHOT
		shotColorDescriptorptr 						shotColor_descriptors;	//descriptores SHOTColor
	};

void extractRangeImage(ImageCloudContainer &currentImage)
{
	Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());
	boost::shared_ptr<pcl::RangeImage> range_image_ptr (new pcl::RangeImage);
	Cloudrgbptr currentPntCLoudPtr = currentImage.getCloudPoint();

	float noise_level = 0.0;
	float min_range = 2.0f;
	int border_size = 1;
	
	scene_sensor_pose = 
				Eigen::Affine3f (Eigen::Translation3f (currentPntCLoudPtr->sensor_origin_[0],
                  currentPntCLoudPtr->sensor_origin_[1],
                  currentPntCLoudPtr->sensor_origin_[2]))
					* Eigen::Affine3f (currentPntCLoudPtr->sensor_orientation_);
	
	range_image_ptr->createFromPointCloud ((*currentPntCLoudPtr),  pcl::deg2rad (0.5f), pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),scene_sensor_pose, pcl::RangeImage::CAMERA_FRAME, noise_level, min_range, border_size);

	range_image_ptr->setUnseenToMaxRange(); 

	currentImage.setRangeImage(*range_image_ptr);

}
double computeCloudResolution (const Cloudrgbptr &cloud)
{
	double res = 0.0;
	int n_points = 0;
	int nres;
	std::vector<int> indices (2);
	std::vector<float> sqr_distances (2);
	pcl::search::KdTree<Rgbpoint> tree;
	tree.setInputCloud (cloud);

	for (size_t i = 0; i < cloud->size (); ++i)
	{
		if (! pcl_isfinite ((*cloud)[i].x))
		{
		  continue;
		}
		//Considering the second neighbor since the first is the point itself.
		nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
		if (nres == 2)
		{
			res += sqrt (sqr_distances[1]);
			++n_points;
		}
	}
	if (n_points != 0)
	{
		res /= n_points;
	}
		return res;
}


void computeNormal(const  Cloudrgbptr &imgKeypoints, pcl::PointCloud<pcl::Normal>::Ptr &cloud_normals)
{
	try{
		pcl::NormalEstimationOMP<Rgbpoint,pcl::Normal> nest;
		// pcl::search::Search<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
		// pcl::search::Search<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
		nest.setKSearch (10);
		// nest.setSearchMethod (tree);
		// nest.setRadiusSearch (0.5);
		nest.setInputCloud (imgKeypoints);
		// nest.setSearchSurface(imgKeypoints);
		
		nest.compute (*cloud_normals);
		
	}catch(std::exception e)  
    {
    	cerr << "FALLO EN OBTENCION LA NORMAL"<< endl;
    }
}


bool shotDescriptorMatching(const  Cloudrgbptr &imgKeypoints, ImageCloudContainer &currentImage,ImageCloudContainer  &lastImage, boost::shared_ptr<pcl::Correspondences> &correspondences)
{
	pcl::PointCloud<pcl::SHOT352>::Ptr shot_descriptors (new pcl::PointCloud<pcl::SHOT352> ());
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal> ());
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal> ());
	std::vector<int> indices;

	pcl::SHOTEstimationOMP<Rgbpoint, pcl::Normal, pcl::SHOT352>::Ptr descr_est(new pcl::SHOTEstimationOMP<Rgbpoint, pcl::Normal, pcl::SHOT352>);
	pcl::FeatureFromNormals<pcl::PointXYZRGB, pcl::Normal,pcl::SHOT352>::Ptr feature_from_normals = boost::dynamic_pointer_cast<pcl::FeatureFromNormals<pcl::PointXYZRGB, pcl::Normal,pcl::SHOT352> > (descr_est);
	

	computeNormal(currentImage.getCloudPoint(), cloud_normals2);
	pcl::removeNaNNormalsFromPointCloud(*cloud_normals2,*cloud_normals,indices);
	feature_from_normals->setInputNormals(cloud_normals);
	PCL_INFO("CLOUD NORMAL SIZE: %d\n",cloud_normals->size());
	descr_est->setSearchSurface (currentImage.getCloudPoint());
	descr_est->setInputCloud (imgKeypoints);
	descr_est->setRadiusSearch (0.04);
	// descr_est->setRadius(0.01);
	// descr_est->setLRFRadius(1.8f);
	
	descr_est->compute (*shot_descriptors);

	currentImage.setSHOTDescriptor(shot_descriptors);
	PCL_INFO("SHOT FEATURES SIZE %d \n", shot_descriptors->size());
	if(lastImage.isReady())
	{
		pcl::registration::CorrespondenceEstimation<pcl::SHOT352, pcl::SHOT352> corEst;
		pcl::search::KdTree<pcl::SHOT352>::Ptr corTree (new pcl::search::KdTree<pcl::SHOT352>);
		corEst.setSearchMethodSource(corTree);
	    corEst.setInputSource (shot_descriptors); 
		corEst.setInputTarget (lastImage.getSHOTDescriptor());
		
		boost::shared_ptr<pcl::Correspondences> cor_all_ptr (new pcl::Correspondences);
		corEst.determineCorrespondences (*cor_all_ptr);
	    PCL_INFO (" Correspondence Estimation - Found %d Correspondences\n", cor_all_ptr->size());
		correspondences= cor_all_ptr;

	    return true;
	}
	return false;
}

bool shotColorDescriptorMatching(const  Cloudrgbptr &imgKeypoints, ImageCloudContainer &currentImage,ImageCloudContainer  &lastImage, boost::shared_ptr<pcl::Correspondences> &correspondences)
{
	pcl::PointCloud<pcl::SHOT1344>::Ptr shot_descriptors (new pcl::PointCloud<pcl::SHOT1344> ());
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal> ());
	std::vector<int> indices;
	pcl::SHOTColorEstimationOMP<Rgbpoint, pcl::Normal, pcl::SHOT1344>::Ptr descr_est(new pcl::SHOTColorEstimationOMP<Rgbpoint, pcl::Normal, pcl::SHOT1344>);

	pcl::FeatureFromNormals<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344>::Ptr feature_from_normals = boost::dynamic_pointer_cast<pcl::FeatureFromNormals<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> > (descr_est);


	computeNormal(currentImage.getCloudPoint(), cloud_normals2);
	pcl::removeNaNNormalsFromPointCloud(*cloud_normals2,*cloud_normals,indices);
	feature_from_normals->setInputNormals(cloud_normals);
	PCL_INFO("CLOUD NORMAL SIZE: %d\n",cloud_normals->size());
	descr_est->setSearchSurface (currentImage.getCloudPoint());
	descr_est->setInputCloud (imgKeypoints);
	descr_est->setRadiusSearch (0.04);
	descr_est->compute (*shot_descriptors);

	currentImage.setSHOTColorDescriptor(shot_descriptors);
	PCL_INFO("SHOT Color FEATURES SIZE %d \n", shot_descriptors->size());
	if(lastImage.isReady())
	{
		pcl::registration::CorrespondenceEstimation<pcl::SHOT1344, pcl::SHOT1344> corEst;
		pcl::search::KdTree<pcl::SHOT1344>::Ptr corTree (new pcl::search::KdTree<pcl::SHOT1344>);
		corEst.setSearchMethodSource(corTree);
	    corEst.setInputSource (shot_descriptors); 
		corEst.setInputTarget (lastImage.getSHOTColorDescriptor());
		
		boost::shared_ptr<pcl::Correspondences> cor_all_ptr (new pcl::Correspondences);
		corEst.determineReciprocalCorrespondences (*cor_all_ptr);
	    PCL_INFO (" Correspondence Estimation - Found %d Correspondences\n", cor_all_ptr->size());
		correspondences= cor_all_ptr;

	    return true;
	}
	return false;
}

bool pfhDescriptorMatching(const  Cloudrgbptr &imgKeypoints, ImageCloudContainer &currentImage,ImageCloudContainer  &lastImage, boost::shared_ptr<pcl::Correspondences> &correspondences)
{
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
	
	computeNormal(imgKeypoints,cloud_normals);

	pcl::PFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PFHSignature125> pfh_est_src;
    
    pfh_est_src.setInputCloud (imgKeypoints);
    pfh_est_src.setInputNormals (cloud_normals);
    pfh_est_src.setRadiusSearch (0.08);
    pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_src (new pcl::PointCloud<pcl::PFHSignature125>);
    PCL_INFO ("PFH - Compute Source\n");
    pfh_est_src.compute (*pfh_src); 
    PCL_INFO("PFH - Size of descriptors %d\n",pfh_src->size());
    currentImage.setPFHDescriptor(pfh_src);
    
    if(lastImage.isReady())
	{
		
	    //// PFH CORRESPONDENCES
		pcl::registration::CorrespondenceEstimation<pcl::PFHSignature125, pcl::PFHSignature125> corEst;
		pcl::search::KdTree<pcl::PFHSignature125>::Ptr corTree (new pcl::search::KdTree<pcl::PFHSignature125>);
		corEst.setSearchMethodSource(corTree);
	    corEst.setInputSource (pfh_src); 
		corEst.setInputTarget (lastImage.getPFHDescriptor());
		
		boost::shared_ptr<pcl::Correspondences> cor_all_ptr (new pcl::Correspondences);
		corEst.determineReciprocalCorrespondences (*cor_all_ptr);
	    PCL_INFO (" Correspondence Estimation - Found %d Correspondences\n", cor_all_ptr->size());
		correspondences= cor_all_ptr;

	    return true;
	    // viewCorrespondences(*cor_all_ptr, currentImage,lastImage);
	}

	return false;
	// viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (imgKeypoints, cloud_normals, 100, 0.5, "normals");
}
bool fpfhDescriptorMatching(const Cloudrgbptr &imgKeypoints, ImageCloudContainer &currentImage,ImageCloudContainer  &lastImage, boost::shared_ptr<pcl::Correspondences> &correspondences )
{
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
	
	computeNormal(imgKeypoints,cloud_normals);

	pcl::FPFHEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> fpfh_est_src;
    
    // fpfh_est_src.setNumberOfThreads(4);
    fpfh_est_src.setInputCloud (imgKeypoints);
    fpfh_est_src.setInputNormals (cloud_normals);
    fpfh_est_src.setRadiusSearch (0.08);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_src (new pcl::PointCloud<pcl::FPFHSignature33>);
    PCL_INFO (" FPFH - Compute Source\n");
    fpfh_est_src.compute (*fpfh_src); 
    PCL_INFO("FPFH - Size of descriptors %d\n",fpfh_src->size());
    currentImage.setFPFHDescriptor(fpfh_src);
    if(lastImage.isReady())
	{
		
	      //// FPFH CORRESPONDENCES
		
		pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> corEst;
		
		pcl::search::KdTree<pcl::FPFHSignature33>::Ptr corTree (new pcl::search::KdTree<pcl::FPFHSignature33>);
	    corEst.setInputSource (currentImage.getFPFHDescriptor()); 
		corEst.setInputTarget (lastImage.getFPFHDescriptor());
		boost::shared_ptr<pcl::Correspondences> cor_all_ptr (new pcl::Correspondences);
		corEst.determineReciprocalCorrespondences (*cor_all_ptr);
	    PCL_INFO (" Correspondence Estimation - Found %d Correspondences\n", cor_all_ptr->size()); 
	    correspondences= cor_all_ptr;
	    return true;	    

	}

	return false;
	// viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (imgKeypoints, cloud_normals, 100, 0.5, "normals");
}

bool narfFeaturesMatching(const Cloudrgbptr &imgKeypoints, ImageCloudContainer &currentImage,ImageCloudContainer  &lastImage, boost::shared_ptr<pcl::Correspondences> &correspondences )
		{
			pcl::PointCloud<pcl::Narf36>::Ptr narf_descriptors( new pcl::PointCloud<pcl::Narf36> );
			
			pcl::PointCloud<int> keypoint_indices=currentImage.getKeypointsIndices();

			/////////FEATURES////////

			std::vector<int> feature_indice;		//descriptores
			feature_indice.resize (keypoint_indices.points.size ());
			 // This step is necessary to get the right vector type
			for (unsigned int i=0; i<keypoint_indices.size (); ++i)
				feature_indice[i]=keypoint_indices.points[i];
			
			pcl::NarfDescriptor narf_descriptor (&currentImage.getRangeImage(), &feature_indice);
			narf_descriptor.getParameters ().support_size = 0.2f;
			narf_descriptor.getParameters ().rotation_invariant = true;
			
			narf_descriptor.compute (*narf_descriptors);

			
			cout << "Extracted "<<narf_descriptors->size ()<<" descriptors for "
                  <<keypoint_indices.points.size ()<< " keypoints.\n";

			currentImage.setNARFDescriptor(narf_descriptors);

            if(lastImage.isReady())
			{
				
			      //// NARF CORRESPONDENCES
				pcl::registration::CorrespondenceEstimation<pcl::Narf36,pcl::Narf36> corEst;
				
			    corEst.setInputSource (currentImage.getNARFDescriptor()); 
				corEst.setInputTarget (lastImage.getNARFDescriptor());
				 boost::shared_ptr<pcl::Correspondences> cor_all_ptr (new pcl::Correspondences); 
				corEst.determineCorrespondences (*cor_all_ptr);
			    PCL_INFO (" Correspondence Estimation - Found %d Correspondences\n", cor_all_ptr->size()); 
			    correspondences= cor_all_ptr;
			    return true;
			}
			return false;
		}

void narfKeypoints(ImageCloudContainer &currentImage)
{
	pcl::PointCloud<int> keypoint_indices;	//detectores
	Cloudrgbptr keypoints_narf( new Cloudrgb);
	pcl::PointCloud<pcl::Narf36>::Ptr narf_descriptors( new pcl::PointCloud<pcl::Narf36> );
	pcl::RangeImage range_image= currentImage.getRangeImage();
	///////////KEYPOINT////////
	pcl::RangeImageBorderExtractor range_image_border_extractor;
	pcl::NarfKeypoint narf_keypoint_detector;

	narf_keypoint_detector.setRangeImageBorderExtractor(&range_image_border_extractor);
	narf_keypoint_detector.setRangeImage (&range_image);
	narf_keypoint_detector.getParameters ().support_size = 0.5f;
	narf_keypoint_detector.setRadiusSearch(1.3);			
	narf_keypoint_detector.compute (keypoint_indices);

	keypoints_narf->width = keypoint_indices.points.size();
    keypoints_narf->height = 1;
    keypoints_narf->is_dense = false;
    keypoints_narf->points.resize (keypoints_narf->width * keypoints_narf->height);
           
    int ind_count=0;
    //source XYZ-CLoud
    for (size_t i = 0; i < keypoint_indices.points.size(); i++)
    {
            ind_count = keypoint_indices.points[i];
                               
            keypoints_narf->points[i].x = range_image.points[ind_count].x;
            keypoints_narf->points[i].y = range_image.points[ind_count].y;
            keypoints_narf->points[i].z = range_image.points[ind_count].z;
    } 

    currentImage.setKeypoints(keypoints_narf);
}

void iss3DKeypoints(ImageCloudContainer &currentImage)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_keypoints (new pcl::PointCloud<pcl::PointXYZRGB> ());
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
	const Cloudrgbptr src = currentImage.getCloudPoint();
	std::vector<int> indices1;
	//
	//  ISS3D parameters
	//
	double iss_salient_radius_;
	double iss_non_max_radius_;
	double iss_normal_radius_;
	double iss_border_radius_;
	double iss_gamma_21_ (0.975);
	double iss_gamma_32_ (0.975);
	double iss_min_neighbors_ (5);
	

	// Fill in the model cloud

	double model_resolution= computeCloudResolution(currentImage.getCloudPoint());

	// Compute model_resolution

	iss_salient_radius_ = 5 * model_resolution;
	iss_non_max_radius_ = 4 * model_resolution;
	iss_normal_radius_ = 3.3 * model_resolution;
	iss_border_radius_ = 0.6 * model_resolution;
	//
	// Compute keypoints
	//
	pcl::ISSKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZRGB> iss_detector;

	iss_detector.setSearchMethod (tree);
	iss_detector.setSalientRadius (iss_salient_radius_);
	iss_detector.setNonMaxRadius (iss_non_max_radius_);

	iss_detector.setNormalRadius (iss_normal_radius_);
	iss_detector.setBorderRadius (iss_border_radius_);

	iss_detector.setThreshold21 (iss_gamma_21_);
	iss_detector.setThreshold32 (iss_gamma_32_);
	iss_detector.setMinNeighbors (iss_min_neighbors_);
	
	iss_detector.setInputCloud (src);
	iss_detector.compute (*model_keypoints);

	currentImage.setKeypoints(model_keypoints);
}

void siftKeypoints(ImageCloudContainer &currentImage)
{

	const float min_scale = 0.01;
	const int nr_octaves = 2;
	const int nr_scales_per_octave = 3;
	const float min_contrast = 1;
	const Cloudrgbptr src= currentImage.getCloudPoint();
	Cloudrgbptr keypoints_src (new Cloudrgb);
    pcl::SIFTKeypoint<Rgbpoint, pcl::PointWithScale> sift_detect;

    if(src->isOrganized() ){
   	 pcl::search::OrganizedNeighbor<Rgbpoint>::Ptr on(new pcl::search::OrganizedNeighbor<Rgbpoint>());
   	 sift_detect.setSearchMethod(on);
    }else{
   	 pcl::search::KdTree<Rgbpoint>::Ptr tree(new pcl::search::KdTree<Rgbpoint> ());
   	 sift_detect.setSearchMethod(tree);
    }
    sift_detect.setScales (min_scale, nr_octaves, nr_scales_per_octave);
    sift_detect.setMinimumContrast (min_contrast);

    sift_detect.setInputCloud (src);
    pcl::PointCloud<pcl::PointWithScale> keypoints_temp;
    sift_detect.compute (keypoints_temp);

    pcl::copyPointCloud (keypoints_temp,*keypoints_src);

    currentImage.setKeypoints(keypoints_src);
}

void harris3DKeypoints(ImageCloudContainer &currentImage)
{
	pcl::PointCloud<pcl::PointXYZI>::Ptr harris_keypoints( new pcl::PointCloud<pcl::PointXYZI>);
	Cloudrgbptr keyPoints(new Cloudrgb);
	pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI,pcl::Normal >::Ptr harris_detector(new pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI,pcl::Normal >);

	harris_detector->setNonMaxSupression(true);
	harris_detector->setSearchSurface (currentImage.getCloudPoint());
	harris_detector->setInputCloud (currentImage.getCloudPoint());
	harris_detector->setRadiusSearch(0.01);
	harris_detector->setRadius(0.004);
	// harris_detector->setThreshold(0.002f);

	harris_detector->compute (*harris_keypoints);


	pcl::copyPointCloud(*harris_keypoints,*keyPoints);

	currentImage.setKeypoints(keyPoints);
}

bool applyDescriptor(const int &descriptor, ImageCloudContainer &currentImage,ImageCloudContainer &lastImage, boost::shared_ptr<pcl::Correspondences> &correspondences )
{
	Cloudrgbptr imgKeypoints = currentImage.getKeypoints();
	struct timespec ts1, ts2;
	bool resultado;
	tick(ts1);
	switch(descriptor)
	{
		case F_PFH:
				resultado= pfhDescriptorMatching(imgKeypoints,currentImage,lastImage,correspondences);
				break;
		case F_FPFH:
				resultado= fpfhDescriptorMatching(imgKeypoints,currentImage,lastImage,correspondences); 
				break;
		case F_SHOT:
				resultado= shotDescriptorMatching(imgKeypoints,currentImage,lastImage,correspondences);
				break;
		case F_SHOTCOLOR:
				resultado= shotColorDescriptorMatching(imgKeypoints,currentImage,lastImage,correspondences);
				break;
		case F_NARF:
				resultado= narfFeaturesMatching(imgKeypoints,currentImage,lastImage,correspondences);
				break;
		default:
			PCL_INFO("DESCRIPTOR NO ENCONTRADO\n");
				return false;
	}
	tack(ts1,ts2);
	return resultado;
}

void applyDetector(const int &detector, ImageCloudContainer &currentImage)
{
	Cloudrgbptr imgKeypoints (new Cloudrgb);
	struct timespec ts1, ts2;
	tick(ts1);
	switch(detector)
	{
		 case K_ISS:
		 	PCL_INFO("ISS KEYPOINTS\n");
		 	iss3DKeypoints(currentImage);
		 	break;
		 case K_SIFT:
		 	PCL_INFO("SIFT KEYPOINTS\n");
		 	siftKeypoints(currentImage);
		 	break;
		 case K_NARF:
		 	PCL_INFO("NARF KEYPOINTS\n");
		 	extractRangeImage(currentImage);
		 	narfKeypoints(currentImage);
		 	break;
		 case K_HARRIS3D:
		 	PCL_INFO("HARRIS3D KEYPOINTS\n");
		 	harris3DKeypoints(currentImage);
		 	break;
		 default:
		 	PCL_INFO("DETECTOR NO ENCONTRADO\n");
	}
	tack(ts1,ts2);

}





#endif /* UTILIDADES_H_ */