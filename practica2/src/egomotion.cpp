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

		struct timespec ts1, ts2;
		sensor_msgs::ImageConstPtr depthmsg;
		cv_bridge::CvImagePtr imageColormsg;
		ImageCloudContainer currentImage;
		ImageCloudContainer lastImage;
		
		Eigen::Matrix4f transformation; 
		pcl::visualization::PCLVisualizer::Ptr viewer;//objeto viewer
		pcl::visualization::PCLVisualizer::Ptr viewerNubeCompleta;//objeto viewer
		pcl::visualization::PCLVisualizer::Ptr viewerNube;//objeto viewer
		bool actualizaLastImage;
		bool depthreceived,imagereceived;
		int detector, descriptor;
		const float* depthImageFloat;
		ofstream fichero;

		Cloudrgbptr mergedCloud;

		
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
			
			PCL_INFO("Matriz anteriro roll: %f, pitch: %f, yaw: %f\n",rollAnt,pitchAnt,yawAnt);
			PCL_INFO("Matriz actual roll: %f, pitch: %f, yaw: %f\n",rollAct,pitchAct,yawAct);

			return sqrt((rollAct-rollAnt)*(rollAct-rollAnt)
							+(pitchAct-pitchAnt)*(pitchAct-pitchAnt)
							+(yawAct-yawAct)*(yawAct-yawAct));
		}
		bool applyRANSAC(const boost::shared_ptr<pcl::Correspondences> &cor_all_ptr)
		{
			struct timespec ts1, ts2;
			boost::shared_ptr<pcl::Correspondences> cor_inliers(new pcl::Correspondences);
			tick(ts1);
			///// RANSAC//////
			PCL_INFO ("Correspondence Rejection Features RANSAC\n");
			pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB> sac;
			sac.setInputSource(currentImage.getKeypoints());
			sac.setInputTarget (lastImage.getKeypoints());
			sac.setInlierThreshold (0.095);
			sac.setMaximumIterations (1300);
			sac.setInputCorrespondences (cor_all_ptr);
			sac.getCorrespondences (*cor_inliers); 
			PCL_INFO ("CORRESPONDENCIAS TOTALES %d ELIMINADAS %d \n ",cor_inliers->size(), 
				cor_all_ptr->size()-cor_inliers->size());
			tack(ts1,ts2);
			Eigen::Matrix4f transformation_tmp;
			Eigen::Matrix4f tranfTemporal ;
			tranfTemporal.setIdentity();
			Eigen::Matrix4f transfomationAnt;
			Cloudrgbptr cloud_tmp (new Cloudrgb);
			Cloudrgbptr cloud_Transformada( new Cloudrgb);
			
			transformation_tmp = sac.getBestTransformation();

			transfomationAnt = transformation;

			tranfTemporal=transformation * transformation_tmp;

			Eigen::Affine3f rotacionAnterior, rotacionActual;
			rotacionAnterior= transfomationAnt;
			rotacionActual= tranfTemporal;

			double distanciaAngular= getRotationDistance(rotacionAnterior,rotacionActual);
			PCL_INFO("DISTANCIA ENTRE NUBES %f \n",distanciaAngular);
			if(distanciaAngular<0.3 || transformation== Eigen::Matrix4f::Identity () )
			{
				PCL_INFO("APLICA TRANSFORAMCION\n");
				//// MUESTRA LA UNION ENTRE DOS NUBES ///
				transformation= tranfTemporal;	
				Eigen::Affine3f transTotal;
				transTotal= transformation;
				pcl::PointXYZRGB p0; //point at zero reference
        		p0.x=0; p0.y=0; p0.z=0; p0.r=255; p0.g=0; p0.b=0;
          		pcl::PointXYZRGB pt_trans=pcl::transformPoint<pcl::PointXYZRGB>(p0,transTotal); //estimated position of the camera
        		Eigen::Quaternion<float> rot2D( (transTotal).rotation());
            	fichero <<""<<depthmsg->header.stamp<<" "<<pt_trans.x<<" "<<pt_trans.y<<" "<<pt_trans.z<<" "<<rot2D.x()<<" "<<rot2D.y()<<" "<<rot2D.z()<<" "<<rot2D.w()<<std::endl;
				pcl::transformPointCloud (*currentImage.getCloudPoint(), *cloud_tmp, transformation_tmp); 
				viewAcumulada(cloud_tmp);
				/// FIN MUESTRA ///
				
				// PCL_INFO("INFORMACION DE ANGULOS %f",angulos);
				pcl::transformPointCloud(*currentImage.getCloudPoint(),*cloud_Transformada, transformation);
				currentImage.setTransformada(cloud_Transformada);
				viewCorrespondences(*cor_inliers, currentImage, lastImage);
				return true;
			}
			return false;

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

		Cloudrgbptr removeNaN(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
		{
			std::vector<int> indices;
			Cloudrgbptr currentPntCloudPtr(new Cloudrgb);
			pcl::removeNaNFromPointCloud (*cloud, *currentPntCloudPtr, indices); 

			return currentPntCloudPtr;
		}

		void processRegistration()
		{
			viewer->removeAllPointClouds();
			viewer->removeShape();
			imagereceived=depthreceived=false;

			pcl::PointCloud<int> keypoint_indices;	//detectores
			pcl::PointCloud<pcl::Narf36> narf_descriptors;
			pcl::RangeImage range_image;
			Cloudrgbptr currentPntCloudPtr;
			Cloudrgbptr cleanKeypoints(new Cloudrgb);

			boost::shared_ptr<pcl::Correspondences> allCorrespondences(new pcl::Correspondences);
			depthImageFloat = reinterpret_cast<const float*>(&depthmsg->data[0]);
			currentPntCloudPtr= getCloudfromColorAndDepth(imageColormsg->image,depthImageFloat);
			currentPntCloudPtr = removeNaN(currentPntCloudPtr);

			//COLOCAMOS LA NUBE DE PUNTOS
			currentImage.storeCloud(currentPntCloudPtr);

			//EXTRAYENDO KEYPOINTS Y FEATURES

			applyDetector(detector,currentImage);
			currentImage.setKeypoints(removeNaN(currentImage.getKeypoints())); 
			PCL_INFO("KEYPOINTS FOUND %d\n",currentImage.getKeypoints()->size());
			// estimateKeypointsSIFT();
			if(applyDescriptor(descriptor,currentImage,lastImage,allCorrespondences) &&
				applyRANSAC(allCorrespondences))
			{
				generateCloud();	
			
			}

			currentImage.setReady();
			lastImage= currentImage;
			

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

		~Egomotion()
		{
			this->fichero.close();
		}

		Egomotion(int detector, int descriptor): itImg(nhImg), itDepth(nhDepht)
		{
			string detectores[4]= {"ISS3D","SIFT","NARF","HARRIS3D"};
			string descriptores[5] = {"FPFH","PFH","SHOT","SHOTCOLOR","NARF"};
			depthreceived=false;
			imagereceived= false;
			actualizaLastImage= true;;
			this->detector= detector;
			this->descriptor= descriptor;
			Cloudrgbptr temp(new Cloudrgb);
			mergedCloud =temp;
			fichero.open("traj_estimated.txt");
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
	string detectores[4]= {"ISS3D","SIFT","NARF","HARRIS3D"};
	string descriptores[5] = {"FPFH","PFH","SHOT","SHOTCOLOR","NARF"};
	if(argc <3)
	{
		cout << "---------Comandos introducidos invalidos---------"<< endl;
		cout << "--- Introduce en primer lugar el numero asociado a los keypoints [0-4] -----"<< endl;
		cout << "[0]->ISS3D\t[1]->SIFT\t[2]->NARF\t[3]->HARRIS3D\t"<< endl;
		cout << "--- Introduce en segundo lugar el numero asociado a los descriptores [0-4] -----"<<endl;
		cout << "[0]->FPFH\t[1]->PFH\t[2]->SHOT\t[3]->SHOTCOLOR\t[4]->NARF\t"<< endl;
	}
	else
	{
		ros::init(argc, argv, "egomotion");
		int detector = atoi(argv[1]);
		int descriptor = atoi(argv[2]);
		cout << "Detector seleccionado: "<<detectores[detector]<<endl;
		cout << "Descriptor seleccionado: "<< descriptores[descriptor]<< endl;

		Egomotion ego(detector,descriptor);
		// ros::spin();
		ego.bucle_eventos();
	}
	return 0;
}