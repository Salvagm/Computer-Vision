#include "utilidades.hpp"




class Egomotion{

	protected:

		ros::NodeHandle nhImg;
		ros::NodeHandle nhDepht;

		image_transport::ImageTransport itImg;
		image_transport::Subscriber imageSub;

		image_transport::ImageTransport itDepth;
		image_transport::Subscriber imgDephtSub;

	private:

		sensor_msgs::ImageConstPtr depthmsg;
		cv_bridge::CvImagePtr imageColormsg;
		ImageCloudContainer currentImage;
		ImageCloudContainer lastImage;
		
		Eigen::Matrix4f transformation; 
		pcl::visualization::PCLVisualizer::Ptr viewer;//objeto viewer
		pcl::visualization::PCLVisualizer::Ptr viewerNubeCompleta;//objeto viewer
		pcl::visualization::PCLVisualizer::Ptr viewerNube;//objeto viewer
		bool prevImageReady;
		bool depthreceived,imagereceived;
		int detector, descriptor;
		const float* depthImageFloat;

		Cloudrgbptr mergedCloud;

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

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr getCloudfromColorAndDepth(const Mat imageColor, const float* depthImage)
		{
	    	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud <pcl::PointXYZRGB>);
		    cloud->height = 480;
		    cloud->width = 640;
		    cloud->is_dense = false;


		    cloud->points.resize(cloud->height * cloud->width);

		    register float constant = 0.0019047619;
		    cloud->header.frame_id = "/openni_rgb_optical_frame";

		    register int centerX = (cloud->width >> 1);
		    int centerY = (cloud->height >> 1);

		    float bad_point = std::numeric_limits<float>::quiet_NaN();
			
		    register int depth_idx = 0;
		    int i,j;
		    for (int v = -centerY,j=0; v < centerY; ++v,++j)
	        {
		        for (register int u = -centerX,i=0; u < centerX; ++u, ++depth_idx,++i)
		        {
		          	pcl::PointXYZRGB& pt = cloud->points[depth_idx];
			        float depthimagevalue=depthImage[depth_idx];


			        if (depthimagevalue == 0)
			        {
				        // not valid
				        pt.x = pt.y = pt.z = bad_point;
				        continue;
				    }
				        pt.z = depthimagevalue;
				          pt.x = u * pt.z * constant;
				          pt.y = v * pt.z * constant;

				        const Point3_<uchar>* p = imageColor.ptr<Point3_<uchar> >(j,i);
				        pt.r=p->z;
				        pt.g=p->y;
				        pt.b=p->x;
	        	}
		        	
		    }
		    return cloud;
		}

		
		void extractRangeImage(pcl::RangeImage& range_image, const Cloudrgbptr &currentPntCLoudPtr)
		{
			Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());
			boost::shared_ptr<pcl::RangeImage> range_image_ptr (new pcl::RangeImage);
			range_image = *range_image_ptr; 
			float noise_level = 0.0;
			float min_range = 2.0f;
			int border_size = 1;
			
			scene_sensor_pose = 
						Eigen::Affine3f (Eigen::Translation3f (currentPntCLoudPtr->sensor_origin_[0],
                          currentPntCLoudPtr->sensor_origin_[1],
                          currentPntCLoudPtr->sensor_origin_[2]))
							* Eigen::Affine3f (currentPntCLoudPtr->sensor_orientation_);
			
			range_image.createFromPointCloud ((*currentPntCLoudPtr),  pcl::deg2rad (0.5f), pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),scene_sensor_pose, pcl::RangeImage::CAMERA_FRAME, noise_level, min_range, border_size);

			range_image.setUnseenToMaxRange(); 
		}

		void mostrarMatriz(const Eigen::Matrix4f &matriz4, const Eigen::Affine3f &matriz3)
		{
			cout << "MATRIZ 4F"<< endl;
			for(int i=0; i<4; ++i)
			{
				for(int j=0; j<4;++j)
				{
					cout << matriz4(i,j)<<"\t";					
				}
				cout << endl;
			}

			cout <<"MATRIZ 3F "<< endl;
			for(int i=0; i<4; ++i)
			{
				for(int j=0; j<4;++j)
				{
					cout << matriz3(i,j)<<"\t";					
				}
				cout << endl;
			}			
		}
		
		double getRotationDistance(const Eigen::Affine3f &matrizAnt, const Eigen::Affine3f &matrizActu)
		{
			float rollAnt,pitchAnt,yawAnt;
			float rollAct,pitchAct,yawAct;

			pcl::getEulerAngles (matrizAnt, rollAnt, pitchAnt, yawAnt);
			pcl::getEulerAngles (matrizActu,rollAct, pitchAct, yawAct);

			return sqrt((rollAct-rollAnt)*(rollAct-rollAnt)
							+(pitchAct-pitchAnt)*(pitchAct-pitchAnt)
							+(yawAct-yawAct)*(yawAct-yawAct));
		}
		void applyRANSAC(const boost::shared_ptr<pcl::Correspondences> &cor_all_ptr)
		{
			boost::shared_ptr<pcl::Correspondences> cor_inliers(new pcl::Correspondences);
			///// RANSAC//////
			PCL_INFO ("Correspondence Rejection Features\n");
			pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB> sac;
			sac.setInputSource(currentImage.getKeypoints());
			sac.setInputTarget (lastImage.getKeypoints());
			sac.setInlierThreshold (0.2);
			sac.setMaximumIterations (1000);
			sac.setInputCorrespondences (cor_all_ptr);
			sac.getCorrespondences (*cor_inliers); 
			PCL_INFO ("CORRESPONDENCIAS TOTALES %d ELIMINADAS %d \n ",cor_inliers->size(), 
				cor_all_ptr->size()-cor_inliers->size());

			Eigen::Matrix4f transformation_tmp;
			Eigen::Matrix4f tranfTemporal ;
			tranfTemporal.setIdentity();
			Eigen::Matrix4f transfomationAnt;
			Cloudrgbptr cloud_tmp (new Cloudrgb);
			Cloudrgbptr cloud_Transformada( new Cloudrgb);
			transformation_tmp = sac.getBestTransformation();

			transfomationAnt = transformation;

			tranfTemporal=transformation * transformation_tmp;


			Eigen::Matrix3f rotacionAnterior, rotacionActual;
			Eigen::Affine3f rotAnt, rotAct;
			rotAnt= transfomationAnt;
			rotAct= transformation;
			// rotacionAnterior.setValue(transfomationAnt(0,0),transfomationAnt(0,1),transfomationAnt(0,2),transfomationAnt(1,0),transfomationAnt(1,1),transfomationAnt(1,2),transfomationAnt(2,0),transfomationAnt(2,1),transfomationAnt(2,2));
			// rotacionActual.setValue(transformation(0,0),transformation(0,1),transformation(0,2),transformation(1,0),transformation(1,1),transformation(1,2),transformation(2,0),transformation(2,1),transformation(2,2));
			

			if(getRotationDistance(rotAnt,rotAct)<0.4)
			{
				//// MUESTRA LA UNION ENTRE DOS NUBES ///
				transformation= tranfTemporal;	
				pcl::transformPointCloud (*currentImage.getCloudPoint(), *cloud_tmp, transformation_tmp); 
				viewAcumulada(cloud_tmp);
				/// FIN MUESTRA ///
				
				// PCL_INFO("INFORMACION DE ANGULOS %f",angulos);
				pcl::transformPointCloud(*currentImage.getCloudPoint(),*cloud_Transformada, transformation);
				currentImage.setTransformada(cloud_Transformada);
				viewCorrespondences(*cor_inliers, currentImage, lastImage);
			}

		}
		void iss3DKeypoints()
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
			int iss_threads_ (4);

			// Fill in the model cloud

			double model_resolution= computeCloudResolution(currentImage.getCloudPoint());

			// Compute model_resolution

			iss_salient_radius_ = 6 * model_resolution;
			iss_non_max_radius_ = 4 * model_resolution;
			iss_normal_radius_ = 4 * model_resolution;
			iss_border_radius_ = 1 * model_resolution;
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
			iss_detector.setNumberOfThreads (iss_threads_);
			iss_detector.setInputCloud (src);
			iss_detector.compute (*model_keypoints);

			currentImage.setKeypoints(model_keypoints);
		}
		
		
		void estimateKeypointsSIFT()
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


		void narfFeaturesMatching(const Cloudrgbptr &imgKeypoints, const pcl::RangeImage &range_image, const pcl::PointCloud<int> &keypoint_indices)
		{
			pcl::PointCloud<pcl::Narf36>::Ptr narf_descriptors( new pcl::PointCloud<pcl::Narf36> );

			/////////FEATURES////////

			std::vector<int> feature_indice;		//descriptores
			feature_indice.resize (keypoint_indices.points.size ());
			 // This step is necessary to get the right vector type
			for (unsigned int i=0; i<keypoint_indices.size (); ++i)
				feature_indice[i]=keypoint_indices.points[i];
			
			pcl::NarfDescriptor narf_descriptor (&range_image, &feature_indice);
			narf_descriptor.getParameters ().support_size = 0.02f;
			narf_descriptor.getParameters ().rotation_invariant = true;
			
			narf_descriptor.compute (*narf_descriptors);


			cout << "Extracted "<<narf_descriptors->size ()<<" descriptors for "
                  <<keypoint_indices.points.size ()<< " keypoints.\n";

			currentImage.setNARFDescriptor(narf_descriptors);
            if(prevImageReady)
			{
				
			      //// NARF CORRESPONDENCES
				pcl::registration::CorrespondenceEstimation<pcl::Narf36,pcl::Narf36> corEst;
				
			    corEst.setInputSource (currentImage.getNARFDescriptor()); 
				corEst.setInputTarget (lastImage.getNARFDescriptor());
				 boost::shared_ptr<pcl::Correspondences> cor_all_ptr (new pcl::Correspondences); 
				corEst.determineCorrespondences (*cor_all_ptr);
			    PCL_INFO (" Correspondence Estimation - Found %d Correspondences\n", cor_all_ptr->size()); 
			    viewCorrespondences(*cor_all_ptr, currentImage,lastImage);
			}


		}
		pcl::PointCloud<int> narfKeypoints(const pcl::RangeImage &range_image)
		{
			pcl::PointCloud<int> keypoint_indices;	//detectores
			Cloudrgbptr keypoints_narf( new Cloudrgb);
			pcl::PointCloud<pcl::Narf36>::Ptr narf_descriptors( new pcl::PointCloud<pcl::Narf36> );
			///////////KEYPOINT////////
			pcl::RangeImageBorderExtractor range_image_border_extractor;
			pcl::NarfKeypoint narf_keypoint_detector;

			narf_keypoint_detector.setRangeImageBorderExtractor(&range_image_border_extractor);
			narf_keypoint_detector.setRangeImage (&range_image);
			narf_keypoint_detector.getParameters ().support_size = 0.5f;
			narf_keypoint_detector.setRadiusSearch(0.5);			
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

            return keypoint_indices;

		}

		void showKeypoint(const Cloudrgbptr &keypoint_indices)
		{
			pcl::PointCloud<Rgbpoint>::Ptr keypoints_ptr (new pcl::PointCloud<Rgbpoint>);
			pcl::PointCloud<Rgbpoint>& keypoints = *keypoints_ptr;

			keypoints.points.resize (keypoint_indices->points.size ());
			for (size_t i=0; i<keypoint_indices->points.size (); ++i)
			keypoints.points[i].getVector3fMap () = keypoint_indices->points[i].getVector3fMap ();

			pcl::visualization::PointCloudColorHandlerCustom<Rgbpoint> keypoints_color_handler (keypoint_indices, 0, 255, 0);
  			viewer->addPointCloud<Rgbpoint> (keypoint_indices, keypoints_color_handler, "keypoints");
  			viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");
		}
		
		void generateCloud()
		{
			Cloudrgbptr cloud_trans (new Cloudrgb);
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
			*cloud_trans= *currentImage.getTransformada();


    	     for(int i=0;i<currentImage.getCloudPoint()->size();i++)
       			mergedCloud->push_back(cloud_trans->at(i));
    		 float voxel_side_size=0.02f; // me quedaria con un representante para cada 2cm cúbicos de la nube
    
	        pcl::VoxelGrid <pcl::PointXYZRGB> sor;
	        sor.setInputCloud (mergedCloud);    //la nube mapa necesita ser filtrada
	        sor.setLeafSize (voxel_side_size, voxel_side_size, voxel_side_size);
	        sor.filter (*cloud_filtered);    //la filtramos en otra nube auxiliar

	        mergedCloud->clear();            //borramos la nube mapa anterior
	        std::swap(mergedCloud,cloud_filtered);    //e intercambiamos los punteros pa
            

			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbcloud(mergedCloud);
		    if (!viewerNubeCompleta->updatePointCloud (mergedCloud,rgbcloud, "cloudn4")) //intento actualizar la nube y si no existe la creo.
		        viewerNubeCompleta->addPointCloud(mergedCloud,rgbcloud,"cloudn4");

		}


		void viewCorrespondences(const pcl::Correspondences &cor_all , ImageCloudContainer &currentImage, ImageCloudContainer  &lastImage)
		{
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ant= lastImage.getCloudPoint();
		    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoint_ant= lastImage.getKeypoints();
			Eigen::Affine3f  transfrom_translation=pcl::getTransformation (5.0, 0, 0, 0, 0, 0);



		    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ant_transformed (new pcl::PointCloud<pcl::PointXYZRGB>);
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr keyPoint_ant_transformed (new pcl::PointCloud<pcl::PointXYZRGB>);

		    pcl::transformPointCloud (*cloud_ant, *cloud_ant_transformed,transfrom_translation);
		    pcl::transformPointCloud (*keypoint_ant, *keyPoint_ant_transformed,transfrom_translation);
		    // pcl::transformPointCloud (*cloud_ant, *n_ant_transformed,transfrom_translation);


		     //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> green(cloud, 0, 255, 0);
		    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbcloud(currentImage.getCloudPoint());
		    if (!viewer->updatePointCloud (currentImage.getCloudPoint(),rgbcloud, "cloudn1")) //intento actualizar la nube y si no existe la creo.
		        viewer->addPointCloud(currentImage.getCloudPoint(),rgbcloud,"cloudn1");

		    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> red(cloud_ant, 255, 0, 0);
		    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbcloud_ant(cloud_ant_transformed);
		    if (!viewer->updatePointCloud (cloud_ant_transformed,rgbcloud_ant, "cloudn2")) //intento actualizar la nube y si no existe la creo.
		        viewer->addPointCloud(cloud_ant_transformed,rgbcloud_ant,"cloudn2");
		    
		    string corresname="correspondences";
		    if (!viewer->updateCorrespondences<pcl::PointXYZRGB>(currentImage.getKeypoints(),keyPoint_ant_transformed,cor_all,1)) //intento actualizar la nube y si no existe la creo.
		        viewer->addCorrespondences<pcl::PointXYZRGB>(currentImage.getKeypoints(),keyPoint_ant_transformed,cor_all,1, corresname);

			////// MOSTRAR KEYPOINT EN IMAGEN ///////
			// showKeypoint(currentImage.getKeypoints(),range_image);
			// showKeypoint(keypoint_ant,lastImage.getRangeImage());
		}

		void viewAcumulada(const Cloudrgbptr &cloudTemp)
		{
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ant= lastImage.getCloudPoint();

			// Eigen::Affine3f  transfrom_translation=pcl::getTransformation (0.0, -5.0, 0, 0, 0, 0);
			// pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ant_transformed (new pcl::PointCloud<pcl::PointXYZRGB>);
			// pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudTemp_ant_transformed (new pcl::PointCloud<pcl::PointXYZRGB>);
			// pcl::transformPointCloud (*cloud_ant, *cloud_ant_transformed,transfrom_translation);
		 //    pcl::transformPointCloud (*cloudTemp, *cloudTemp_ant_transformed,transfrom_translation);

			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> green(cloudTemp, 0, 255, 0);

			// pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbcloud(cloudTemp_ant_transformed);
		    if (!viewerNube->updatePointCloud (cloudTemp,green, "cloudn3")) //intento actualizar la nube y si no existe la creo.
		        viewerNube->addPointCloud(cloudTemp,green,"cloudn3");

		    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> red(cloud_ant, 255, 0, 0);
		    // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbcloud_ant(cloud_ant_transformed);
		    if (!viewerNube->updatePointCloud (cloud_ant,red, "cloudn4")) //intento actualizar la nube y si no existe la creo.
		        viewerNube->addPointCloud(cloud_ant,red,"cloudn4");

			

		}
	public:

		void imageCbdepth(const sensor_msgs::ImageConstPtr& msg)
	  	{
		    depthmsg=msg;
		     // std::cerr<<" depthcb: "<<msg->header.frame_id<<" : "<<msg->header.seq<<" : "<<msg->header.stamp<<std::endl;
		    //depthImagemsg = cv_bridge::toCvCopy(msg, enc::TYPE_32FC1);

		    depthreceived=true;
		    if(imagereceived && depthreceived)
		        processRegistration();

		}
		void imageCb(const sensor_msgs::ImageConstPtr& msg)
		{
			try
			{
				imageColormsg = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

			}
			catch (cv_bridge::Exception& e)
			{
				ROS_ERROR("cv_bridge exception: %s", e.what());
				return;
			}

			// std::cerr<<" imagecb: "<<msg->header.frame_id<<" : "<<msg->header.seq<<" : "<<msg->header.stamp<<std::endl;
			imagereceived=true;
			if(imagereceived && depthreceived)
			    processRegistration();
		}
		void processRegistration()
		{
			viewer->removeAllPointClouds();
			viewer->removeShape();
			imagereceived=depthreceived=false;

			pcl::PointCloud<int> keypoint_indices;	//detectores
			pcl::PointCloud<pcl::Narf36> narf_descriptors;
			pcl::RangeImage range_image;
			std::vector<int> indices;
			Cloudrgbptr currentPntCloudPtr(new Cloudrgb);
			boost::shared_ptr<pcl::Correspondences> allCorrespondences(new pcl::Correspondences);
			depthImageFloat = reinterpret_cast<const float*>(&depthmsg->data[0]);
			
			pcl::removeNaNFromPointCloud (*getCloudfromColorAndDepth(imageColormsg->image,depthImageFloat), *currentPntCloudPtr, indices); 
			//COLOCAMOS LA NUBE DE PUNTOS
			currentImage.storeCloud(currentPntCloudPtr);

			//////////////////////////
			// MOSTAR NUBE DE PUNTOS//
			//////////////////////////
			// pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(currentImage.getCloudPoint());   //esto es el manejador de color de la nube "cloud"
			// if (!viewer->updatePointCloud (currentImage.getCloudPoint(),rgb, "cloud")) //intento actualizar la nube y si no existe la creo.
   //  		viewer->addPointCloud(currentImage.getCloudPoint(),rgb,"cloud");
 
 


			//EXTRAYENDO KEYPOINTS Y FEATURES
			// -----------------------------------------------------------------
			// -----Extract NARF descriptors / features for interest points-----
			// -----------------------------------------------------------------
    		//// Obteniendo la Imagen a partir de la nube ///
			// extractRangeImage(range_image,currentImage.getCloudPoint());
			// keypoint_indices = narfKeypoints(range_image);
			// if(prevImageReady==true)
			// 	PCL_INFO("LastImage keypoint: %d, CurrentImagen keypoints: %d\n",lastImage.getKeypoints()->size(),currentImage.getKeypoints()->size());
			// narfFeaturesMatching(currentImage.getKeypoints(),range_image,keypoint_indices);

			iss3DKeypoints();
			// estimateKeypointsSIFT();
			if(applyDescriptor(descriptor,currentImage,lastImage,allCorrespondences))
			{
				applyRANSAC(allCorrespondences);

				generateCloud();	
			}
			// fpfhDescriptor(currentImage.getKeypoints());
			// pfhDescriptor(currentImage.getKeypoints());

			currentImage.setReady();
			lastImage= currentImage;
			prevImageReady=true;

		}

		void bucle_eventos()
		{

		    while (ros::ok())
		    {

		      ros::spinOnce();       // Handle ROS events
		      // cloud_viewer_regi.spinOnce(1);  //update viewers
		      // cloud_viewer.spinOnce(1);
		      viewer->spinOnce(1);
		    }
		}


		Egomotion(int detector, int descriptor): itImg(nhImg), itDepth(nhDepht)
		{
			depthreceived=false;
			imagereceived= false;
			prevImageReady= false;
			this->detector= detector;
			this->descriptor= descriptor;
			Cloudrgbptr temp(new Cloudrgb);
			mergedCloud =temp;
			transformation.setIdentity();
			//Inicializamos el viewer
			viewer= pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer ("3D Viewer"));
			viewerNube= pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer ("Emparejada 3D"));
			viewerNubeCompleta= pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer ("Acumulada 3D"));
			viewer->setBackgroundColor (0, 0, 0);
			viewer->initCameraParameters ();
			viewerNube->setBackgroundColor(0,0,0);
			viewerNube->initCameraParameters();
			viewerNubeCompleta->setBackgroundColor(0,0,0);
			viewerNubeCompleta->initCameraParameters();
 			imageSub = itImg.subscribe("/camera/rgb/image_color", 1, &Egomotion::imageCb,this);

 			imgDephtSub= itDepth.subscribe("/camera/depth/image",1,&Egomotion::imageCbdepth,this);
		}
};

int main(int argc, char **argv)
{
	if(argc <3)
	{
		cout << "---------Comandos introducidos invalidos---------"<< endl;
		cout << "--- Introduce en primer lugar el numero asociado a los keypoints [0-5] -----"<< endl;
		cout << "[0]->SIFT\t[1]->ISS3D\t"<< endl;
		cout << "--- Introduce en segundo lugar el numero asociado a los descriptores [0-4] -----"<<endl;
		cout << "[0]->FPFH\t[1]->PFH\t[2]->SHOT\t[3]->SHOTCOLOR\t[4]->NARF\t"<< endl;
	}
	else
	{
		ros::init(argc, argv, "egomotion");
		int detector = atoi(argv[1]);
		int descriptor = atoi(argv[2]);

		Egomotion ego(detector,descriptor);
		// ros::spin();
		ego.bucle_eventos();
	}
	return 0;
}